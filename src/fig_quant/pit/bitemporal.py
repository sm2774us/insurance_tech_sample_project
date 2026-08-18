"""Bi-temporal point-in-time (PIT) data access layer.

Enforces the invariant that no query may observe information that would
not have been knowable at a given decision time `T_decision`. Every fact
carries two independent timelines:

  * Event time (``t_event``): when the real-world event occurred.
  * Ingestion validity (``t_ingest_start``, ``t_ingest_end``): the half-open
    interval during which a database record was the current belief.

A row is visible to a query at ``T_decision`` iff::

    t_event <= T_decision AND t_ingest_start <= T_decision < t_ingest_end

This blocks two distinct leakage modes: event-time lookahead (using a
claim that had not yet happened) and ingestion-time lookahead (using a
value that was later corrected/restated but was not yet known at
``T_decision``).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib

import duckdb
import pyarrow as pa

_END_OF_TIME = dt.datetime(9999, 12, 31, tzinfo=dt.timezone.utc)


@dataclasses.dataclass(frozen=True, slots=True)
class BitemporalRecord:
    """A single versioned fact in the bi-temporal store.

    Attributes:
        entity_id: Identifier of the entity the fact describes (e.g. a
          policy ID or county-quarter key).
        t_event: Wall-clock time the underlying real-world event occurred.
        t_ingest_start: Time this version became the current belief.
        t_ingest_end: Time this version was superseded (``_END_OF_TIME`` if
          still current).
        payload: Arrow-serializable field values for this fact version.
    """

    entity_id: str
    t_event: dt.datetime
    t_ingest_start: dt.datetime
    t_ingest_end: dt.datetime
    payload: dict[str, float | int | str]


class BitemporalStore:
    """DuckDB-backed bi-temporal fact store with a point-in-time gate.

    Swappable storage backend: pass ``uri="file://..."`` for local Parquet
    or ``uri="s3://..."`` for a Ceph/S3-compatible object store; DuckDB's
    httpfs/S3 extension handles the URI transparently so downstream query
    code is identical between standalone and cluster execution modes.
    """

    def __init__(self, uri: str = "file://./data/bitemporal.parquet") -> None:
        """Initializes the store and its DuckDB connection.

        Args:
          uri: Storage location URI. ``file://`` resolves to local Parquet
            for standalone mode; ``s3://`` targets cluster object storage.
        """
        self._uri = uri
        self._conn = duckdb.connect(database=":memory:")
        if uri.startswith("s3://"):
            # Only load the httpfs extension for cluster-mode object storage;
            # standalone local-Parquet mode has no network dependency.
            self._conn.execute("INSTALL httpfs; LOAD httpfs;")
        self._path = self._resolve_path(uri)
        self._ensure_schema()

    @staticmethod
    def _resolve_path(uri: str) -> str:
        if uri.startswith("file://"):
            local = pathlib.Path(uri[len("file://") :])
            local.parent.mkdir(parents=True, exist_ok=True)
            return str(local)
        return uri

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                entity_id VARCHAR,
                t_event TIMESTAMP,
                t_ingest_start TIMESTAMP,
                t_ingest_end TIMESTAMP,
                payload_json VARCHAR
            )
            """
        )

    def append(self, records: list[BitemporalRecord]) -> int:
        """Appends new fact versions, closing out any superseded versions.

        Args:
          records: New fact versions to insert. Each is assumed to be the
            new current version for its ``entity_id``; any prior open
            (``t_ingest_end == _END_OF_TIME``) row for the same entity is
            closed at the new row's ``t_ingest_start``.

        Returns:
          Count of rows appended.
        """
        import json

        for rec in records:
            self._conn.execute(
                """
                UPDATE facts
                SET t_ingest_end = ?
                WHERE entity_id = ? AND t_ingest_end >= ?
                """,
                [rec.t_ingest_start, rec.entity_id, _END_OF_TIME],
            )
            self._conn.execute(
                "INSERT INTO facts VALUES (?, ?, ?, ?, ?)",
                [
                    rec.entity_id,
                    rec.t_event,
                    rec.t_ingest_start,
                    rec.t_ingest_end,
                    json.dumps(rec.payload),
                ],
            )
        self._persist()
        return len(records)

    def _persist(self) -> None:
        self._conn.execute(
            f"COPY facts TO '{self._path}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE true)"
        )

    def as_of(self, decision_time: dt.datetime) -> pa.Table:
        """Returns the valid information set F_T as of ``decision_time``.

        Args:
          decision_time: The ``T_decision`` cutoff. Rows with
            ``t_event <= decision_time`` and
            ``t_ingest_start <= decision_time < t_ingest_end`` pass the
            point-in-time gate; all others are excluded (lookahead leakage
            blocked by construction rather than by post-hoc filtering).

        Returns:
          An Arrow table of the visible fact rows, one row per entity
          reflecting the belief state at ``decision_time``.
        """
        result = self._conn.execute(
            """
            SELECT entity_id, t_event, t_ingest_start, t_ingest_end, payload_json
            FROM facts
            WHERE t_event <= ?
              AND t_ingest_start <= ?
              AND t_ingest_end > ?
            ORDER BY entity_id
            """,
            [decision_time, decision_time, decision_time],
        )
        return result.fetch_arrow_table()

    def close(self) -> None:
        """Closes the underlying DuckDB connection."""
        self._conn.close()
