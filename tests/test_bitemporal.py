"""Tests for the bi-temporal point-in-time data store."""

from __future__ import annotations

import datetime as dt

from fig_quant.pit.bitemporal import BitemporalRecord, BitemporalStore


def _dt(y: int, m: int, d: int) -> dt.datetime:
    return dt.datetime(y, m, d, tzinfo=dt.UTC)


def test_as_of_blocks_ingestion_lookahead(tmp_path) -> None:
    store = BitemporalStore(uri=f"file://{tmp_path}/facts.parquet")
    store.append(
        [
            BitemporalRecord(
                entity_id="C001",
                t_event=_dt(2024, 1, 1),
                t_ingest_start=_dt(2024, 1, 5),
                t_ingest_end=_dt(9999, 12, 31),
                payload={"value": 1.0},
            )
        ]
    )
    before_ingest = store.as_of(_dt(2024, 1, 2))
    assert before_ingest.num_rows == 0
    after_ingest = store.as_of(_dt(2024, 1, 10))
    assert after_ingest.num_rows == 1
    store.close()


def test_as_of_blocks_event_lookahead(tmp_path) -> None:
    store = BitemporalStore(uri=f"file://{tmp_path}/facts.parquet")
    store.append(
        [
            BitemporalRecord(
                entity_id="C002",
                t_event=_dt(2024, 6, 1),
                t_ingest_start=_dt(2024, 1, 1),
                t_ingest_end=_dt(9999, 12, 31),
                payload={"value": 2.0},
            )
        ]
    )
    result = store.as_of(_dt(2024, 3, 1))
    assert result.num_rows == 0
    store.close()


def test_restatement_closes_prior_version(tmp_path) -> None:
    store = BitemporalStore(uri=f"file://{tmp_path}/facts.parquet")
    store.append(
        [
            BitemporalRecord(
                entity_id="C003",
                t_event=_dt(2024, 1, 1),
                t_ingest_start=_dt(2024, 1, 1),
                t_ingest_end=_dt(9999, 12, 31),
                payload={"value": 10.0},
            )
        ]
    )
    store.append(
        [
            BitemporalRecord(
                entity_id="C003",
                t_event=_dt(2024, 1, 1),
                t_ingest_start=_dt(2024, 2, 1),
                t_ingest_end=_dt(9999, 12, 31),
                payload={"value": 20.0},
            )
        ]
    )
    as_of_original = store.as_of(_dt(2024, 1, 15))
    assert as_of_original.num_rows == 1
    as_of_restated = store.as_of(_dt(2024, 2, 15))
    assert as_of_restated.num_rows == 1
    store.close()
