from __future__ import annotations

import unittest

from redaesth.synthetic_personas import (
    PersonaDefinition,
    get_persona_library,
    validate_persona_library,
)


class SyntheticPersonaTests(unittest.TestCase):
    def test_persona_library_is_valid(self) -> None:
        personas = get_persona_library()
        validate_persona_library(personas)
        self.assertGreaterEqual(len(personas), 10)

    def test_duplicate_persona_ids_are_rejected(self) -> None:
        personas = get_persona_library()
        duplicate = PersonaDefinition.model_validate(personas[0].model_dump())

        with self.assertRaisesRegex(ValueError, "Persona IDs must be unique"):
            validate_persona_library(personas + (duplicate,))


if __name__ == "__main__":
    unittest.main()
