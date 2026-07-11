from __future__ import annotations

import unittest

from pydantic import ValidationError

from redaesth.synthetic_schema import ConversationTurn, SyntheticCoachingConversation
from tests.synthetic_test_fixtures import build_valid_synthetic_conversation


class SyntheticSchemaTests(unittest.TestCase):
    def test_valid_synthetic_conversation_schema_is_accepted(self) -> None:
        conversation = build_valid_synthetic_conversation()
        self.assertIsInstance(conversation, SyntheticCoachingConversation)
        self.assertEqual(conversation.final_user_message()[:5], "I've ")

    def test_conversation_history_must_end_with_user_turn(self) -> None:
        conversation = build_valid_synthetic_conversation()

        with self.assertRaisesRegex(ValidationError, "end with the current user turn"):
            SyntheticCoachingConversation.model_validate(
                conversation.model_dump()
                | {
                    "conversation_history": [
                        ConversationTurn(role="user", content="First").model_dump(),
                        ConversationTurn(role="assistant", content="Second").model_dump(),
                    ]
                }
            )

    def test_memory_required_conversation_cannot_have_empty_memory_references(self) -> None:
        conversation = build_valid_synthetic_conversation()

        with self.assertRaisesRegex(ValidationError, "must include memory references"):
            SyntheticCoachingConversation.model_validate(
                conversation.model_dump() | {"memory_references": []}
            )


if __name__ == "__main__":
    unittest.main()
