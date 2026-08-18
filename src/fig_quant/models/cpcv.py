"""Combinatorial Purged Cross-Validation (CPCV) with embargo purging.

Standard k-fold CV on temporally clustered data leaks information: adjacent
train/test years share autocorrelated shocks. CPCV enumerates every
size-``r`` combination of held-out year-groups as a test fold, and purges
training years within an embargo window ``e`` of any test year, so no
training observation is temporally adjacent to the evaluation set.
"""

from __future__ import annotations

import dataclasses
import itertools


@dataclasses.dataclass(frozen=True, slots=True)
class CpcvFold:
    """A single combinatorial purged fold.

    Attributes:
      test_years: Year-groups held out for evaluation.
      train_years: Year-groups available for training, after purging any
        year within ``embargo`` of a test year.
    """

    test_years: tuple[int, ...]
    train_years: tuple[int, ...]


def cpcv_year_splits(
    years: list[int], n_test_groups: int = 2, embargo: int = 0
) -> list[CpcvFold]:
    """Generates all C(n, n_test_groups) combinatorial purged folds.

    Args:
      years: All available year-groups, e.g. ``[2021, 2022, 2023, 2024]``.
      n_test_groups: Size of each held-out combination (``r`` in ``C(n, r)``).
      embargo: Purge window; any training year within ``embargo`` of a test
        year (by absolute distance) is dropped from that fold's training
        set.

    Returns:
      One :class:`CpcvFold` per combination, in itertools combination
      order.
    """
    folds: list[CpcvFold] = []
    for test_combo in itertools.combinations(sorted(years), n_test_groups):
        train_years = tuple(
            y
            for y in years
            if y not in test_combo
            and min(abs(y - t) for t in test_combo) > embargo
        )
        folds.append(CpcvFold(test_years=test_combo, train_years=train_years))
    return folds
