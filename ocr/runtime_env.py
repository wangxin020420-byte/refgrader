"""Deterministic runtime settings for the isolated PaddleX/PaddleOCR worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


PADDLEX_CACHE_ENV = "PADDLE_PDX_CACHE_HOME"
PADDLEX_SOURCE_ENV = "PADDLE_PDX_MODEL_SOURCE"
PADDLEX_SOURCE_CHECK_ENV = "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"


def resolve_paddlex_cache_home(
    project_root: str | os.PathLike[str],
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Return the single authoritative PaddleX model cache for this device.

    An explicit environment setting always wins. Otherwise Windows uses an
    ASCII path at the root of the drive containing the checkout, avoiding
    native-library issues seen with non-ASCII user profiles. Other platforms
    use one stable per-user cache shared by all RefGrader checkouts.
    """
    env = os.environ if environment is None else environment
    configured = str(env.get(PADDLEX_CACHE_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser().absolute()

    root = Path(project_root).absolute()
    if os.name == "nt":
        drive_root = Path(root.anchor)
        return drive_root / "paddlex_cache"
    return Path.home() / ".cache" / "refgrader" / "paddlex"


def build_paddlex_environment(
    project_root: str | os.PathLike[str],
    base_environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], Path]:
    """Build the worker environment and create its authoritative cache."""
    env = dict(os.environ if base_environment is None else base_environment)
    cache_home = resolve_paddlex_cache_home(project_root, env)
    try:
        cache_home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create the authoritative PaddleX cache: {cache_home}. "
            f"Set {PADDLEX_CACHE_ENV} to a writable ASCII path."
        ) from exc

    env[PADDLEX_CACHE_ENV] = str(cache_home)
    env.setdefault(PADDLEX_SOURCE_ENV, "modelscope")
    env.setdefault(PADDLEX_SOURCE_CHECK_ENV, "True")
    return env, cache_home


def configure_paddlex_process(
    project_root: str | os.PathLike[str],
) -> Path:
    """Apply deterministic PaddleX settings before importing PaddleOCR."""
    env, cache_home = build_paddlex_environment(project_root)
    for name in (
        PADDLEX_CACHE_ENV,
        PADDLEX_SOURCE_ENV,
        PADDLEX_SOURCE_CHECK_ENV,
    ):
        os.environ[name] = env[name]
    return cache_home
