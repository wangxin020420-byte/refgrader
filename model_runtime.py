"""Shared model selection and thinking-mode configuration.

The experiment runner passes these values through environment variables so
every subprocess in one run uses the same model contract.
"""

from __future__ import annotations

import os
from typing import Any


TEXT_MODEL_PROFILES = {
    "glm": {"model": "glm-4.5-air", "family": "glm"},
    "glm5": {"model": "glm-5.1", "family": "glm"},
    "glm47": {"model": "glm-4.7", "family": "glm"},
    "deepseek": {"model": "deepseek-v4-flash", "family": "deepseek"},
    "deepseek_v4pro": {"model": "deepseek-v4-pro", "family": "deepseek"},
}

VLM_MODEL_PROFILES = {
    "glm4v": "glm-4.6v",
    "glm5v": "glm-5v-turbo",
}

DEFAULT_TEXT_MODEL_PROVIDER = "glm47"
DEFAULT_TEXT_THINKING_MODE = "disabled"
DEFAULT_VLM_MODEL_PROVIDER = "glm4v"

TEXT_PROVIDER_ENV = "REFGRADER_TEXT_MODEL_PROVIDER"
TEXT_THINKING_ENV = "REFGRADER_TEXT_THINKING"
VLM_PROVIDER_ENV = "REFGRADER_VLM_MODEL_PROVIDER"


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


def runtime_model_config(
    *,
    text_provider: str | None = None,
    thinking_mode: str | None = None,
    vlm_provider: str | None = None,
) -> dict[str, str]:
    text_provider = resolve_text_provider(text_provider)
    thinking_mode = resolve_thinking_mode(thinking_mode)
    vlm_provider = resolve_vlm_provider(vlm_provider)
    text_profile = TEXT_MODEL_PROFILES[text_provider]
    return {
        "text_provider": text_provider,
        "text_model": text_profile["model"],
        "text_family": text_profile["family"],
        "text_thinking": thinking_mode,
        "vlm_provider": vlm_provider,
        "vlm_model": VLM_MODEL_PROFILES[vlm_provider],
    }


def get_model_runtime_config(
    *,
    text_provider: str | None = None,
    thinking_mode: str | None = None,
    vlm_provider: str | None = None,
) -> dict[str, str]:
    """Return the shared runtime contract through a stable public name."""
    return runtime_model_config(
        text_provider=text_provider,
        thinking_mode=thinking_mode,
        vlm_provider=vlm_provider,
    )


def model_environment(config: dict[str, str]) -> dict[str, str]:
    return {
        TEXT_PROVIDER_ENV: config["text_provider"],
        TEXT_THINKING_ENV: config["text_thinking"],
        VLM_PROVIDER_ENV: config["vlm_provider"],
    }


def thinking_extra_body(mode: str | None = None) -> dict[str, dict[str, str]]:
    return {"thinking": {"type": resolve_thinking_mode(mode)}}
