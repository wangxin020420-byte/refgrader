import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from model_runtime import (
    DEEPSEEK_BASE_URL_ENV,
    DEEPSEEK_MODEL_ENV,
    DEFAULT_TEXT_MODEL_PROVIDER,
    DEFAULT_TEXT_THINKING_MODE,
    TEXT_TEMPERATURE_OVERRIDE_ENV,
    effective_text_temperature,
    get_model_runtime_config,
    model_environment,
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

    def test_deepseek_endpoint_and_model_can_be_frozen_by_environment(self):
        with patch.dict(
            os.environ,
            {
                DEEPSEEK_MODEL_ENV: "deepseek-chat",
                DEEPSEEK_BASE_URL_ENV: "https://example.invalid/v1",
            },
            clear=True,
        ):
            config = runtime_model_config(text_provider="deepseek")
        self.assertEqual(config["text_model"], "deepseek-chat")
        self.assertEqual(config["text_base_url"], "https://example.invalid/v1")

    def test_zero_temperature_override_is_recorded_and_propagated(self):
        with patch.dict(
            os.environ,
            {TEXT_TEMPERATURE_OVERRIDE_ENV: "0"},
            clear=True,
        ):
            config = runtime_model_config(text_provider="deepseek")
        self.assertEqual(config["text_temperature_override"], 0.0)
        self.assertEqual(
            model_environment(config)[TEXT_TEMPERATURE_OVERRIDE_ENV],
            "0",
        )
        self.assertEqual(
            effective_text_temperature(0.35, override=0.0),
            0.0,
        )

    def test_native_temperature_is_preserved_without_override(self):
        with patch.dict(os.environ, {}, clear=True):
            config = runtime_model_config()
            effective = effective_text_temperature(0.35)
        self.assertNotIn("text_temperature_override", config)
        self.assertEqual(effective, 0.35)

    def test_invalid_temperature_override_is_rejected(self):
        for value in ("not-a-number", "nan", -0.1, 2.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "temperature override"):
                    runtime_model_config(text_temperature_override=value)

    def test_text_model_request_uses_zero_temperature_override(self):
        import step4_vlm_grader as grader

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with (
            patch.object(grader, "TEXT_MODEL_FAMILY", "deepseek"),
            patch.object(grader, "TEXT_MODEL_NAME", "deepseek-chat"),
            patch.object(grader, "TEXT_TEMPERATURE_OVERRIDE", 0.0),
            patch.object(grader, "DEEPSEEK_API_KEY", "test-key"),
            patch.object(grader, "DEEPSEEK_BASE_URL", "https://example.invalid"),
            patch.object(grader, "OpenAI", return_value=client),
        ):
            result = grader.call_text_model([], temperature=0.35)
        self.assertEqual(result, "ok")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["temperature"],
            0.0,
        )

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
            with patch.dict(os.environ, {}, clear=True):
                no_thinking = runtime_model_config(
                    text_provider="glm47", thinking_mode="disabled"
                )
                thinking = runtime_model_config(
                    text_provider="glm47", thinking_mode="enabled"
                )
                fixed_temperature = runtime_model_config(
                    text_provider="glm47",
                    thinking_mode="disabled",
                    text_temperature_override=0,
                )
            first = _run_signature([context], "test", None, no_thinking)
            second = _run_signature([context], "test", None, thinking)
            third = _run_signature(
                [context], "test", None, fixed_temperature
            )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
