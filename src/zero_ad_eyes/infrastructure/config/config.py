"""Configuration loading & persistence (REQUIREMENTS.md X3 / NF7).

The *types* — :class:`Config` and its parts — are pure value objects and live in
``application.settings`` (policy). This module is the *infrastructure* around them:
it *loads* a config from a JSON file, *overrides* it from environment variables, and
*round-trips* it back to disk (calibration-profile store, X3). That is the I/O the
application ring must not depend on, so it lives here and depends inward on the types.

Layering (lowest precedence first): a supplied ``base`` config < JSON file <
environment. The types are defaultless (see ``application.settings``), so the loader
never conjures values on its own: the caller injects the base — the canonical set
from ``interface.default_config.default_config`` — and a file/env only overrides the
leaves it names. Environment keys are prefixed (``ZAE_``) and address nested fields
with ``__``, e.g. ``ZAE_THRESHOLDS__MIN_CONFIDENCE=0.7``. Nothing here imports
OpenCV, so importing the config is always headless-safe.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zero_ad_eyes.application.settings import Config

DEFAULT_ENV_PREFIX = "ZAE_"
_NESTED_DELIMITER = "__"


def save_config(config: Config, path: str | os.PathLike[str]) -> None:
    """Serialise ``config`` to a JSON file (round-trips with :func:`load_config`)."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_config(
    base: Config,
    path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> Config:
    """Layer an optional JSON file and env overrides onto a ``base`` config.

    Precedence (ascending): ``base`` < JSON file (if it exists) < environment
    overrides. The models are defaultless, so ``base`` (the caller-supplied canonical
    default, :func:`interface.default_config.default_config`) provides every value a
    partial file or env leaves unset; passing only ``base`` returns it unchanged.
    """

    data: dict[str, Any] = base.model_dump(mode="json")
    if path is not None:
        source = Path(path)
        if source.exists():
            loaded = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = _deep_merge(data, loaded)

    overrides = _env_overrides(os.environ if env is None else env, prefix)
    if overrides:
        data = _deep_merge(data, overrides)

    return Config.model_validate(data)


def _env_overrides(env: Mapping[str, str], prefix: str) -> dict[str, Any]:
    """Turn ``PREFIX_A__B=value`` variables into a nested override dict."""

    overrides: dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(prefix):
            continue
        path = raw_key[len(prefix) :].lower().split(_NESTED_DELIMITER)
        if not path or "" in path:
            continue
        cursor = overrides
        for part in path[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _parse_scalar(raw_value)
    return overrides


def _parse_scalar(value: str) -> Any:
    """Best-effort JSON decode of an env value, falling back to the raw string."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""

    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged
