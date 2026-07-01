"""Typed configuration system (REQUIREMENTS.md X3 / NF7).

A single, typed source of truth for thresholds, filesystem paths, and rendering
palettes so no magic number is scattered across the codebase. The config is a
pure ``pydantic`` value object with sensible defaults; it can be *loaded* from a
JSON file, then *overridden* by environment variables, and *round-tripped* back to
disk without loss (calibration-profile store, X3).

Layering (lowest precedence first): built-in defaults < JSON file < environment.
Environment keys are prefixed (``ZAE_``) and address nested fields with ``__``,
e.g. ``ZAE_THRESHOLDS__MIN_CONFIDENCE=0.7``.

Colours are stored as RGB triples (matching ``HudState.self_player_color``); the
overlay converts them to OpenCV's BGR order at draw time. Nothing here imports
OpenCV, so importing the config is always headless-safe.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.minimap import FogState
from zero_ad_eyes.domain.taxonomy import Ownership

RGB = tuple[int, int, int]

DEFAULT_ENV_PREFIX = "ZAE_"
_NESTED_DELIMITER = "__"


def _default_owner_colors() -> dict[Ownership, RGB]:
    return {
        Ownership.SELF: (60, 180, 75),  # green
        Ownership.ALLY: (0, 130, 200),  # blue
        Ownership.ENEMY: (230, 25, 75),  # red
        Ownership.GAIA: (170, 170, 170),  # grey
        Ownership.UNKNOWN: (255, 255, 255),  # white
    }


def _default_fog_colors() -> dict[FogState, RGB]:
    return {
        FogState.UNEXPLORED: (20, 20, 20),  # near-black
        FogState.EXPLORED: (90, 90, 90),  # shroud grey
        FogState.VISIBLE: (60, 120, 60),  # lit green tint
    }


class OwnerPalette(BaseModel):
    """RGB colour per :class:`Ownership`, used to tint entities and blips (E3)."""

    model_config = ConfigDict(frozen=True)

    colors: dict[Ownership, RGB] = Field(default_factory=_default_owner_colors)

    def for_ownership(self, ownership: Ownership) -> RGB:
        return self.colors.get(ownership, self.colors[Ownership.UNKNOWN])


class FogPalette(BaseModel):
    """RGB colour per :class:`FogState`, used to tint the fog panel (§4.5)."""

    model_config = ConfigDict(frozen=True)

    colors: dict[FogState, RGB] = Field(default_factory=_default_fog_colors)

    def for_state(self, state: FogState) -> RGB:
        return self.colors.get(state, self.colors[FogState.UNEXPLORED])


class OverlaySettings(BaseModel):
    """Everything the debug overlay (X1) needs to draw, with no magic numbers."""

    model_config = ConfigDict(frozen=True)

    owner_palette: OwnerPalette = Field(default_factory=OwnerPalette)
    fog_palette: FogPalette = Field(default_factory=FogPalette)

    health_good: RGB = (60, 180, 75)
    health_warn: RGB = (255, 200, 0)
    health_bad: RGB = (230, 25, 75)
    health_good_min: float = Field(default=0.5, ge=0.0, le=1.0)
    health_warn_min: float = Field(default=0.25, ge=0.0, le=1.0)

    hud_text_color: RGB = (255, 255, 255)
    hud_panel_color: RGB = (0, 0, 0)
    font_scale: float = Field(default=0.4, gt=0.0)
    box_thickness: int = Field(default=1, ge=1)
    minimap_fraction: float = Field(default=0.25, gt=0.0, le=1.0)

    def health_color(self, fraction: float) -> RGB:
        if fraction >= self.health_good_min:
            return self.health_good
        if fraction >= self.health_warn_min:
            return self.health_warn
        return self.health_bad


class Thresholds(BaseModel):
    """Perception thresholds and the NF3 accuracy targets, in one place (NF7)."""

    model_config = ConfigDict(frozen=True)

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_health: float = Field(default=0.0, ge=0.0, le=1.0)
    hud_read_max_error: float = Field(default=0.01, ge=0.0, le=1.0)  # NF3 <1%
    detection_map_target: float = Field(default=0.80, ge=0.0, le=1.0)  # NF3
    ownership_accuracy_target: float = Field(default=0.98, ge=0.0, le=1.0)  # NF3


class Paths(BaseModel):
    """Filesystem locations for recordings and calibration profiles (X2/X3)."""

    model_config = ConfigDict(frozen=True)

    recordings_dir: Path = Path("recordings")
    calibration_dir: Path = Path("calibration")


class Config(BaseModel):
    """The single, typed configuration root (X3)."""

    model_config = ConfigDict(frozen=True)

    thresholds: Thresholds = Field(default_factory=Thresholds)
    paths: Paths = Field(default_factory=Paths)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)


def save_config(config: Config, path: str | os.PathLike[str]) -> None:
    """Serialise ``config`` to a JSON file (round-trips with :func:`load_config`)."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> Config:
    """Build a :class:`Config` from defaults, an optional JSON file, and the env.

    Precedence (ascending): built-in defaults < JSON file (if it exists) <
    environment overrides. Passing no arguments yields the pure defaults.
    """

    data: dict[str, Any] = {}
    if path is not None:
        source = Path(path)
        if source.exists():
            loaded = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded

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
