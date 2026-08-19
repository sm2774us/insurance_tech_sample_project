"""Tests for the rate-limited storm event ETL client."""

from __future__ import annotations

from fig_quant.data.storm_events import RateLimiter, StormEventClient, synthetic_storm_source


def test_synthetic_storm_source_reproducible() -> None:
    a = synthetic_storm_source("06037", 202406, seed=42)
    b = synthetic_storm_source("06037", 202406, seed=42)
    assert a == b


def test_synthetic_storm_source_zero_events_no_damage() -> None:
    payload = synthetic_storm_source("00000", 202401, seed=1)
    if payload["event_count"] == 0:
        assert payload["total_damage"] == 0.0


def test_storm_event_client_caches() -> None:
    calls = {"count": 0}

    def source(county: str, ym: int) -> dict:
        calls["count"] += 1
        return {"county": county, "ym": ym}

    client = StormEventClient(source_fn=source, rate_limiter=RateLimiter(max_calls_per_sec=1000.0))
    client.fetch_county_month("06037", 202401)
    client.fetch_county_month("06037", 202401)
    assert calls["count"] == 1


def test_storm_event_client_retries_then_raises() -> None:
    def failing_source(county: str, ym: int) -> dict:
        raise ValueError("transient")

    client = StormEventClient(
        source_fn=failing_source,
        rate_limiter=RateLimiter(max_calls_per_sec=1000.0),
        max_retries=2,
        backoff_base_sec=0.001,
    )
    try:
        client.fetch_county_month("06037", 202401)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Failed to fetch" in str(exc)


def test_rate_limiter_acquire_does_not_raise() -> None:
    limiter = RateLimiter(max_calls_per_sec=1000.0)
    for _ in range(5):
        limiter.acquire()
