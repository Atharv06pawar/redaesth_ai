from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redaesth.config import RedAesthConfig
from redaesth.final_split import spot_check_training_split


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize, add_generation_prompt
        rendered = []
        for message in messages:
            rendered.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
        return "".join(rendered)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        del add_special_tokens
        return text.split()


def build_config(project_root: Path) -> RedAesthConfig:
    return RedAesthConfig(
        project_root=project_root,
        base_model_id="test/model",
        embedding_model_id="test/embedding",
        max_seq_length=10,
        final_spot_check_sample_size=3,
    )


def write_train_split(project_root: Path, records: list[dict[str, object]]) -> Path:
    path = project_root / "data" / "final" / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


class FinalSpotCheckTests(unittest.TestCase):
    def test_spot_check_flags_malformed_and_over_length_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            tokenizer = FakeTokenizer()

            good_conversation = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "plan three sessions"},
            ]
            long_conversation = [
                {"role": "user", "content": "one two three four five six"},
                {"role": "assistant", "content": "seven eight nine ten eleven twelve"},
            ]
            records = [
                {
                    "record_id": "good",
                    "conversations": good_conversation,
                    "text": tokenizer.apply_chat_template(
                        good_conversation,
                        tokenize=False,
                        add_generation_prompt=False,
                    ),
                },
                {
                    "record_id": "malformed",
                    "conversations": good_conversation,
                    "text": "",
                },
                {
                    "record_id": "long",
                    "conversations": long_conversation,
                    "text": tokenizer.apply_chat_template(
                        long_conversation,
                        tokenize=False,
                        add_generation_prompt=False,
                    ),
                },
            ]
            write_train_split(project_root, records)

            _, report = spot_check_training_split(config=config, tokenizer=tokenizer)

            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["malformed_record_count"], 1)
            self.assertEqual(report["over_length_record_count"], 1)
            self.assertAlmostEqual(report["truncation_rate"], 1 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
