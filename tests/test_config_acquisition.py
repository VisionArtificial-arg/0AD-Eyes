"""Config guards for acquisition — offline replay + live capture (Approach B, P3)."""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.infrastructure.acquisition import ImageFolderSource, ScreenCaptureSource
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.interface.cli import _offline_source
from zero_ad_eyes.interface.default_config import default_config


def test_defaults_match_historical() -> None:
    # Guard the generated acquisition defaults against the frozen historical values.
    a = default_config().acquisition
    assert a.offline_fps == 30.0
    assert a.image_extensions == (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    assert (a.live_monitor, a.live_fps) == (1, 30.0)


def test_config_file_threads_offline_fps(tmp_path: Path) -> None:
    (tmp_path / "frames").mkdir()
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('{"acquisition": {"offline_fps": 10.0}}', encoding="utf-8")

    config = load_config(default_config(), cfg_path, env={})
    source = _offline_source(str(tmp_path / "frames"), config.acquisition)

    assert isinstance(source, ImageFolderSource)
    assert source._fps == 10.0


def test_config_threads_live_capture_knobs_into_source() -> None:
    config = load_config(
        default_config(),
        env={"ZAE_ACQUISITION__LIVE_MONITOR": "2", "ZAE_ACQUISITION__LIVE_FPS": "12.0"},
    )
    source = ScreenCaptureSource.from_settings(config.acquisition)

    # Built from config, not in-code defaults: the pacer runs at the configured fps.
    assert source._pacer._interval == 1.0 / 12.0


def test_build_live_pipeline_wires_live_source_and_fuser() -> None:
    # The live composition root builds the classical chain over a screen-capture source
    # (constructed, not run — no display needed). mss is imported lazily on first grab.
    from zero_ad_eyes.application.pipeline import PerceptionPipeline
    from zero_ad_eyes.interface.cli import _build_live_pipeline

    pipeline = _build_live_pipeline(config=default_config(), max_frames=2)

    assert isinstance(pipeline, PerceptionPipeline)
    assert isinstance(pipeline._source, ScreenCaptureSource)
    assert pipeline._fuser is not None
