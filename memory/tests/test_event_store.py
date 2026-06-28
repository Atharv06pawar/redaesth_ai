from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from memory.event_schema import Event, EventType
from memory.event_store import EventStore


class EventStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "events.sqlite3"
        self.store = EventStore(self.db_path)
        self.base_time = datetime(2026, 6, 28, 12, 0, 0)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _event(self, *, event_id: str, event_type: EventType, payload: dict[str, object]) -> Event:
        return Event(
            event_id=event_id,
            user_id="user-1",
            event_type=event_type,
            timestamp=self.base_time,
            payload=payload,
            source="user_explicit",
            confidence=0.98,
            tags=["test"],
        )

    def test_append_and_get_events(self) -> None:
        self.store.append(
            self._event(
                event_id="event-1",
                event_type=EventType.WEIGHT_RECORDED,
                payload={"weight_kg": 82.4, "note": "morning weigh-in"},
            )
        )

        events = self.store.get_events("user-1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["weight_kg"], 82.4)

    def test_get_latest_returns_most_recent_event(self) -> None:
        older = self._event(
            event_id="event-older",
            event_type=EventType.GOAL_CHANGED,
            payload={"previous_goal": "cutting", "new_goal": "maintenance"},
        )
        older.timestamp = self.base_time

        newer = self._event(
            event_id="event-newer",
            event_type=EventType.GOAL_CHANGED,
            payload={"previous_goal": "maintenance", "new_goal": "lean bulk"},
        )
        newer.timestamp = self.base_time + timedelta(hours=1)

        self.store.append(older)
        self.store.append(newer)

        latest = self.store.get_latest("user-1", EventType.GOAL_CHANGED)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.event_id, "event-newer")

    def test_supersede_marks_prior_event(self) -> None:
        self.store.append(
            self._event(
                event_id="event-1",
                event_type=EventType.SLEEP_PATTERN_CHANGED,
                payload={"previous_average_hours": 7.5, "new_average_hours": 6.0},
            )
        )
        self.store.supersede("event-1", "event-2")

        event = self.store.get_events("user-1")[0]
        self.assertEqual(event.superseded_by, "event-2")

    def test_search_uses_payload_text(self) -> None:
        self.store.append(
            self._event(
                event_id="event-1",
                event_type=EventType.COACH_NOTE_ADDED,
                payload={"note": "Shoulder tweak on incline pressing", "priority": "high"},
            )
        )

        results = self.store.search("user-1", "Shoulder")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].event_type, EventType.COACH_NOTE_ADDED)


if __name__ == "__main__":
    unittest.main()
