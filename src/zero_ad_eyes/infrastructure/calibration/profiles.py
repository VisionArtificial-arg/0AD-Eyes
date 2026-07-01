"""Calibration profile persistence (EPIC B — B3).

A ``CalibrationProfileStore`` persists and reuses ``Calibration`` value objects
keyed by *resolution + theme*, so a resolution/theme calibrated once need not be
re-detected on every session (REQUIREMENTS.md B3, X3 calibration-profile store).

Storage is a directory of JSON files, one per profile, named by the key. The
directory is configurable; the default is a CWD-relative hidden folder so tests and
tools never write outside the repository. ``Calibration`` is a pydantic model, so
serialisation is its own responsibility — this store only owns *where* and *when*.
"""

from __future__ import annotations

import re
from pathlib import Path

from zero_ad_eyes.domain.calibration import Calibration

DEFAULT_PROFILE_DIRNAME = ".zae_calibration_profiles"

# Themes are free-form labels; collapse anything unsafe for a filename to keep the
# key a stable, portable slug.
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(theme: str) -> str:
    cleaned = _SLUG_UNSAFE.sub("_", theme.strip())
    return cleaned or "default"


class CalibrationProfileStore:
    """A JSON-backed store of calibration profiles keyed by resolution+theme."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = (
            Path(directory) if directory is not None else Path.cwd() / DEFAULT_PROFILE_DIRNAME
        )

    @property
    def directory(self) -> Path:
        return self._dir

    def key(self, width: int, height: int, theme: str) -> str:
        return f"{int(width)}x{int(height)}@{_slug(theme)}"

    def load(self, width: int, height: int, theme: str) -> Calibration | None:
        path = self._path(width, height, theme)
        if not path.is_file():
            return None
        return Calibration.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, calibration: Calibration, theme: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(calibration.width, calibration.height, theme)
        path.write_text(calibration.model_dump_json(), encoding="utf-8")

    def _path(self, width: int, height: int, theme: str) -> Path:
        return self._dir / f"{self.key(width, height, theme)}.json"
