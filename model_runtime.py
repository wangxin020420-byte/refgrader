"""Shared model selection and thinking-mode configuration.

The experiment runner passes these values through environment variables so
every subprocess in one run uses the same model contract.
"""

from __future__ import annotations

import math
import os
from typing import Any


TEXT_MODEL_PROFILES = {
    "glm": {"model": "glm-4.5-air", "family": "glm"},
    "glm5": {"model": "glm-5.2", "family": "glm"},
    "glm47": {"model": "glm-4.7", "family": "glm"},
    "deepseek": {"model": "deepseek-v4-flash", "family": "deepseek"},
    "deepseek_v4pro": {"model": "deepseek-v4-pro", "family": "deepseek"},
}

VLM_MODEL_PROFILES = {
    "glm4v": "glm-4.6v",
    "glm5v": "glm-5v-turbo",
}

DEFAULT_TEXT_MODEL_PROVIDER = "glm5"
DEFAULT_TEXT_THINKING_MODE = "disabled"
DEFAULT_VLM_MODEL_PROVIDER = "glm4v"

TEXT_PROVIDER_ENV = "REFGRADER_TEXT_MODEL_PROVIDER"
TEXT_THINKING_ENV = "REFGRADER_TEXT_THINKING"
VLM_PROVIDER_ENV = "REFGRADER_VLM_MODEL_PROVIDER"
DEEPSEEK_MODEL_ENV = "REFGRADER_DEEPSEEK_MODEL"
DEEPSEEK_BASE_URL_ENV = "REFGRADER_DEEPSEEK_BASE_URL"
TEXT_TEMPERATURE_OVERRIDE_ENV = "REFGRADER_TEXT_TEMPERATURE_OVERRIDE"
DEFAULT_DEEPSEEK_BASE_URL = "https://gpt-agent.cc/v1"


def _choice(value: str, choices: dict[str, Any] | set[str], label: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"Unsupported {label} {value!r}; choose one of: {allowed}")
    return normalized


def resolve_text_provider(value: str | None = None) -> str:
    return _choice(
        value or os.getenv(TEXT_PROVIDER_ENV, DEFAULT_TEXT_MODEL_PROVIDER),
        TEXT_MODEL_PROFILES,
        "text model provider",
    )


def resolve_thinking_mode(value: str | None = None) -> str:
    return _choice(
        value or os.getenv(TEXT_THINKING_ENV, DEFAULT_TEXT_THINKING_MODE),
        {"enabled", "disabled"},
        "thinking mode",
    )


def resolve_vlm_provider(value: str | None = None) -> str:
    return _choice(
        value or os.getenv(VLM_PROVIDER_ENV, DEFAULT_VLM_MODEL_PROVIDER),
        VLM_MODEL_PROFILES,
        "VLM provider",
    )


def resolve_text_temperature_override(
    value: str | float | int | None = None,
) -> float | None:
    raw_value = (
        os.getenv(TEXT_TEMPERATURE_OVERRIDE_ENV)
        if value is None
        else value
    )
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        temperature = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid text temperature override: {raw_value!r}"
        ) from exc
    if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
        raise ValueError(
            "Text temperature override must be finite and between 0 and 2: "
            f"{raw_value!r}"
        )
    return temperature


def effective_text_temperature(
    requested: str | float | int,
    *,
    override: str | float | int | None = None,
) -> float:
    resolved_override = resolve_text_temperature_override(override)
    if resolved_override is not None:
        return resolved_override
    try:
        temperature = float(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid requested text temperature: {requested!r}") from exc
    if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
        raise ValueError(
            "Requested text temperature must be finite and between 0 and 2: "
            f"{requested!r}"
        )
    return temperature


def runtime_model_config(
    *,
    text_provider: str | None = None,
    thinking_mode: str | None = None,
    vlm_provider: str | None = None,
    text_temperature_override: str | float | int | None = None,
) -> dict[str, Any]:
    text_provider = resolve_text_provider(text_provider)
    thinking_mode = resolve_thinking_mode(thinking_mode)
    vlm_provider = resolve_vlm_provider(vlm_provider)
    text_profile = TEXT_MODEL_PROFILES[text_provider]
    text_model = text_profile["model"]
    config = {
        "text_provider": text_provider,
        "text_model": text_model,
        "text_family": text_profile["family"],
        "text_thinking": thinking_mode,
        "vlm_provider": vlm_provider,
        "vlm_model": VLM_MODEL_PROFILES[vlm_provider],
    }
    temperature_override = resolve_text_temperature_override(
        text_temperature_override
    )
    if temperature_override is not None:
        config["text_temperature_override"] = temperature_override
    if text_profile["family"] == "deepseek":
        config["text_model"] = (
            os.getenv(DEEPSEEK_MODEL_ENV, text_model).strip() or text_model
        )
        config["text_base_url"] = (
            os.getenv(DEEPSEEK_BASE_URL_ENV, DEFAULT_DEEPSEEK_BASE_URL).strip()
            or DEFAULT_DEEPSEEK_BASE_URL
        )
    return config


def get_model_runtime_config(
    *,
    text_provider: str | None = None,
    thinking_mode: str | None = None,
    vlm_provider: str | None = None,
    text_temperature_override: str | float | int | None = None,
) -> dict[str, Any]:
    """Return the shared runtime contract through a stable public name."""
    return runtime_model_config(
        text_provider=text_provider,
        thinking_mode=thinking_mode,
        vlm_provider=vlm_provider,
        text_temperature_override=text_temperature_override,
    )


def model_environment(config: dict[str, Any]) -> dict[str, str]:
    environment = {
        TEXT_PROVIDER_ENV: config["text_provider"],
        TEXT_THINKING_ENV: config["text_thinking"],
        VLM_PROVIDER_ENV: config["vlm_provider"],
    }
    if config.get("text_family") == "deepseek":
        environment[DEEPSEEK_MODEL_ENV] = config["text_model"]
        environment[DEEPSEEK_BASE_URL_ENV] = config["text_base_url"]
    if "text_temperature_override" in config:
        environment[TEXT_TEMPERATURE_OVERRIDE_ENV] = format(
            float(config["text_temperature_override"]), ".12g"
        )
    return environment


def thinking_extra_body(mode: str | None = None) -> dict[str, dict[str, str]]:
    return {"thinking": {"type": resolve_thinking_mode(mode)}}
