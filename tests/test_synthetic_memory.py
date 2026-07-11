from __future__ import annotations

import unittest

from memory.event_schema import EventType
from redaesth.synthetic_memory import (
    MemoryCategory,
    MemoryEventSpecification,
    MemoryPriority,
    MemoryUsageExample,
    get_memory_event_specifications,
    validate_memory_specifications,
)


class SyntheticMemorySpecificationTests(unittest.TestCase):
    def test_memory_specifications_are_valid(self) -> None:
        specifications = get_memory_event_specifications()
        validate_memory_specifications(specifications)
        self.assertGreaterEqual(len(specifications), 8)

    def test_missing_creation_rules_are_rejected(self) -> None:
        broken = MemoryEventSpecification(
            event_type=EventType.GOAL_CHANGED,
            category=MemoryCategory.GOAL,
            priority=MemoryPriority.HIGH,
            creation_rules=[],
            retrieval_rules=["retrieve on planning questions"],
            behavioral_adaptation_rules=["update the plan"],
            invalid_memory_usage=["quote raw schema"],
            expiration_policy="Until superseded.",
            valid_usage_examples=[MemoryUsageExample(label="valid", example="Use the new goal.")],
            invalid_usage_examples=[MemoryUsageExample(label="invalid", example="Use the old goal.")],
        )

        with self.assertRaisesRegex(ValueError, "missing creation rules"):
            validate_memory_specifications(get_memory_event_specifications() + (broken,))


if __name__ == "__main__":
    unittest.main()
