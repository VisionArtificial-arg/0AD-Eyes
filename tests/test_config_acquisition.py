"""Config guards for offline acquisition (Approach B, P3)."""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import AcquisitionSettings
from zero_ad_eyes.infrastructure.acquisition import ImageFolderSource
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.interface.cli import _offline_source


def test_defaults_match_historical() -> None:
    a = AcquisitionSettings()
    assert a.offline_fps == 30.0
    assert a.image_extensions[0] == ".png"


def test_config_file_threads_offline_fps(tmp_path: Path) -> None:
    (tmp_path / "frames").mkdir()
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('{"acquisition": {"offline_fps": 10.0}}', encoding="utf-8")

    config = load_config(cfg_path, env={})
    source = _offline_source(str(tmp_path / "frames"), config.acquisition)

    assert isinstance(source, ImageFolderSource)
    assert source._fps == 10.0
