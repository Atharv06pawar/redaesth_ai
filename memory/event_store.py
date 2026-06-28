from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .event_schema import Event, EventType


def _normalize_value(value: Any) -> Any:
    """Convert dataclasses and datetimes into JSON-friendly structures."""

    if is_dataclass(value):
        return _normalize_value(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _payload_text(value: Any) -> str:
    """Flatten payload content into searchable text."""

    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_payload_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_payload_text(item) for item in value)
    return str(value)


class EventStore:
    """SQLite-backed append-only event store for durable user memory."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    conversation_id TEXT,
                    tags_json TEXT NOT NULL,
                    superseded_by TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_events_user_time
                    ON events (user_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_events_type_time
                    ON events (event_type, timestamp DESC);

                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
                    USING fts5(event_id UNINDEXED, user_id UNINDEXED, payload_text);
                """
            )
            connection.commit()

    def append(self, event: Event) -> str:
        """Append an event and return its persisted identifier."""

        event_id = event.event_id or str(uuid.uuid4())
        payload = _normalize_value(event.payload)
        payload_json = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        payload_text = _payload_text(payload)
        tags_json = json.dumps(event.tags, ensure_ascii=True)

        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO events (
                    event_id, user_id, event_type, timestamp, payload_json, payload_text,
                    source, confidence, conversation_id, tags_json, superseded_by
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event.user_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    payload_json,
                    payload_text,
                    event.source,
                    event.confidence,
                    event.conversation_id,
                    tags_json,
                    event.superseded_by,
                ),
            )
            connection.execute(
                """
                INSERT INTO events_fts (event_id, user_id, payload_text)
                VALUES (?, ?, ?)
                """,
                (event_id, event.user_id, payload_text),
            )
            connection.commit()
        return event_id

    def get_events(
        self,
        user_id: str,
        event_types: list[EventType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Return events for a user filtered by type and time."""

        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]

        if event_types:
            placeholders = ", ".join("?" for _ in event_types)
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(item.value for item in event_types)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until.isoformat())

        query = f"""
            SELECT *
            FROM events
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp DESC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_latest(self, user_id: str, event_type: EventType) -> Event | None:
        """Return the most recent event of a given type for a user."""

        events = self.get_events(user_id=user_id, event_types=[event_type], limit=1)
        return events[0] if events else None

    def supersede(self, event_id: str, new_event_id: str) -> None:
        """Mark an existing event as superseded by a newer one."""

        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE events SET superseded_by = ? WHERE event_id = ?",
                (new_event_id, event_id),
            )
            connection.commit()

    def search(self, user_id: str, query: str) -> list[Event]:
        """Perform an FTS-backed search over a user's event payload text."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT events.*
                FROM events_fts
                JOIN events ON events.event_id = events_fts.event_id
                WHERE events_fts.user_id = ?
                  AND events_fts.payload_text MATCH ?
                ORDER BY events.timestamp DESC
                """,
                (user_id, query),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert a database row into an Event instance."""

        return Event(
            event_id=row["event_id"],
            user_id=row["user_id"],
            event_type=EventType(row["event_type"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            payload=json.loads(row["payload_json"]),
            source=row["source"],
            confidence=float(row["confidence"]),
            conversation_id=row["conversation_id"],
            tags=json.loads(row["tags_json"]),
            superseded_by=row["superseded_by"],
        )
