import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_runtime import (
    DEFAULT_TEXT_MODEL_PROVIDER,
    DEFAULT_TEXT_THINKING_MODE,
    get_model_runtime_config,
    runtime_model_config,
    thinking_extra_body,
)
from scripts.run_csbench import _run_signature


class ModelRuntimeTests(unittest.TestCase):
    def test_default_contract_is_glm52_without_thinking(self):
        with patch.dict(os.environ, {}, clear=True):
            config = runtime_model_config()
        self.assertEqual(DEFAULT_TEXT_MODEL_PROVIDER, "glm5")
        self.assertEqual(DEFAULT_TEXT_THINKING_MODE, "disabled")
        self.assertEqual(config["text_model"], "glm-5.2")
        self.assertEqual(config["text_thinking"], "disabled")
        self.assertEqual(config["vlm_model"], "glm-4.6v")

    def test_public_runtime_config_alias_matches_primary_helper(self):
        self.assertEqual(get_model_runtime_config(), runtime_model_config())

    def test_environment_override_is_reflected_in_signature(self):
        with patch.dict(
            os.environ,
            {
                "REFGRADER_TEXT_MODEL_PROVIDER": "glm47",
                "REFGRADER_TEXT_THINKING": "enabled",
            },
            clear=True,
        ):
            config = runtime_model_config()
        self.assertEqual(config["text_model"], "glm-4.7")
        self.assertEqual(config["text_thinking"], "enabled")

    def test_thinking_payload_is_explicit(self):
        self.assertEqual(
            thinking_extra_body("disabled"),
            {"thinking": {"type": "disabled"}},
        )

    def test_invalid_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported text model"):
            runtime_model_config(text_provider="unknown")

    def test_run_signature_changes_with_model_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            rubric = root / "rubric.json"
            split.write_text('{"test":["A"]}', encoding="utf-8")
            rubric.write_text('[{"id":"s1","points":1}]', encoding="utf-8")
            context = SimpleNamespace(
                question_id="CO_1",
                split_file=split,
                optimized_rubric=rubric,
            )
            no_thinking = runtime_model_config(
                text_provider="glm47", thinking_mode="disabled"
            )
            thinking = runtime_model_config(
                text_provider="glm47", thinking_mode="enabled"
            )
            first = _run_signature([context], "test", None, no_thinking)
            second = _run_signature([context], "test", None, thinking)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
