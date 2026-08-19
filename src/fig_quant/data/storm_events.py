"""Rate-limited, retrying, cached ETL client for county-level storm events.

Demonstrates tooling independence: a pod researcher can build a compliant,
production-grade alt-data ingestion client without waiting on the
platform team, given only a callable data source.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Callable

import numpy as np


@dataclasses.dataclass(slots=True)
class RateLimiter:
    """Token-bucket rate limiter.

    Attributes:
      max_calls_per_sec: Maximum sustained call rate.
    """

    max_calls_per_sec: float
    _tokens: float = dataclasses.field(init=False, default=0.0)
    _last_refill: float = dataclasses.field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._tokens = self.max_calls_per_sec
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        """Blocks, if necessary, until a call token is available."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_calls_per_sec, self._tokens + elapsed * self.max_calls_per_sec)
        self._last_refill = now
        if self._tokens < 1.0:
            wait = (1.0 - self._tokens) / self.max_calls_per_sec
            time.sleep(wait)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0


SourceFn = Callable[[str, int], dict]


@dataclasses.dataclass(slots=True)
class StormEventClient:
    """Fetches and caches county-month storm event summaries.

    Attributes:
      source_fn: Dependency-injected data source callable
        ``(county_fips, year_month) -> payload dict``, allowing the real
        network client to be swapped for a synthetic/test source.
      rate_limiter: Token-bucket limiter guarding ``source_fn`` calls.
      max_retries: Maximum retry attempts on transient failure.
      backoff_base_sec: Base for exponential backoff (``base * 2**attempt``).
    """

    source_fn: SourceFn
    rate_limiter: RateLimiter = dataclasses.field(
        default_factory=lambda: RateLimiter(max_calls_per_sec=5.0)
    )
    max_retries: int = 3
    backoff_base_sec: float = 0.1
    _cache: dict[str, dict] = dataclasses.field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._cache = {}

    @staticmethod
    def _cache_key(county_fips: str, year_month: int) -> str:
        raw = f"{county_fips}:{year_month}".encode()
        return hashlib.sha1(raw).hexdigest()

    def fetch_county_month(self, county_fips: str, year_month: int) -> dict:
        """Fetches (with caching + retry/backoff) one county-month record.

        Args:
          county_fips: 5-digit county FIPS code.
          year_month: Year-month key as ``YYYYMM`` integer.

        Returns:
          The source payload dict for that county-month.

        Raises:
          RuntimeError: If all retry attempts are exhausted.
        """
        key = self._cache_key(county_fips, year_month)
        if key in self._cache:
            return self._cache[key]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.rate_limiter.acquire()
            try:
                payload = self.source_fn(county_fips, year_month)
                self._cache[key] = payload
                return payload
            except Exception as exc:  # noqa: BLE001 - retry on any transient failure
                last_error = exc
                time.sleep(self.backoff_base_sec * (2**attempt))
        raise RuntimeError(
            f"Failed to fetch {county_fips}/{year_month} after {self.max_retries} attempts"
        ) from last_error


def synthetic_storm_source(county_fips: str, year_month: int, seed: int | None = None) -> dict:
    """Generates a reproducible synthetic county-month storm event record.

    Reproduces Poisson event counts with a seasonal (Gaussian, centered on
    June) peak and Pareto-tailed property damage, for offline testing and
    demos without a live network dependency.

    Args:
      county_fips: 5-digit county FIPS code (used to seed determinism).
      year_month: Year-month key as ``YYYYMM``.
      seed: Optional explicit seed override.

    Returns:
      A payload dict with ``event_count``, ``mean_magnitude``,
      ``max_magnitude``, and ``total_damage`` fields.
    """
    month = year_month % 100
    base_seed = seed if seed is not None else int(hashlib.sha1(county_fips.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(base_seed + year_month)

    seasonal_bump = 3.0 * np.exp(-((month - 6) ** 2) / (2 * 2.0**2))
    lam = max(0.2 + seasonal_bump, 0.05)
    event_count = int(rng.poisson(lam))

    if event_count == 0:
        return {
            "county_fips": county_fips,
            "year_month": year_month,
            "event_count": 0,
            "mean_magnitude": 0.0,
            "max_magnitude": 0.0,
            "total_damage": 0.0,
        }
    magnitudes = rng.gamma(shape=2.0, scale=1.5, size=event_count)
    damages = (rng.pareto(a=1.5, size=event_count) + 1.0) * 5000.0
    return {
        "county_fips": county_fips,
        "year_month": year_month,
        "event_count": event_count,
        "mean_magnitude": float(np.mean(magnitudes)),
        "max_magnitude": float(np.max(magnitudes)),
        "total_damage": float(np.sum(damages)),
    }
