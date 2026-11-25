"""CAP-3 — replaying a plain video (e.g. OBS, no ``--record`` sidecar), optionally strided."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from zero_ad_eyes.application.settings import AcquisitionSettings
from zero_ad_eyes.infrastructure.acquisition import VideoFileSource
from zero_ad_eyes.interface.cli import _offline_source
from zero_ad_eyes.interface.default_config import default_config

FRAMES, FPS = 6, 30.0


def _write_video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"FFV1"), FPS, (32, 24))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write FFV1/MKV")
    for i in range(FRAMES):
        writer.write(np.full((24, 32, 3), i * 40, np.uint8))
    writer.release()
    return path


def _acquisition(stride: int = 1) -> AcquisitionSettings:
    acquisition = default_config().acquisition
    return acquisition.model_copy(update={"offline_video_stride": stride})


def test_sidecar_less_video_replays_on_its_own_clock(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "obs-match.mkv")
    source = _offline_source(str(video), _acquisition())
    assert isinstance(source, VideoFileSource)
    frames = list(source.frames())
    assert [f.meta.frame_id for f in frames] == list(range(FRAMES))
    stamps = [f.meta.timestamp for f in frames]
    assert stamps == sorted(stamps) and len(set(stamps)) == FRAMES
    assert frames[0].meta.source == "obs-match"
    assert (frames[0].meta.width, frames[0].meta.height) == (32, 24)


def test_stride_keeps_every_nth_frame_with_its_original_id(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "obs-match.mkv")
    frames = list(_offline_source(str(video), _acquisition(stride=2)).frames())
    assert [f.meta.frame_id for f in frames] == [0, 2, 4]
    # The decoded pixels are the frames at those indices, not the first three.
    assert [int(f.image[0, 0, 0]) for f in frames] == pytest.approx([0, 80, 160], abs=2)


def test_stride_must_be_positive() -> None:
    with pytest.raises(ValueError, match="stride"):
        VideoFileSource("x.mkv", stride=0)


def test_portal_config_sample_loads() -> None:
    from zero_ad_eyes.infrastructure.config import load_config

    sample = Path(__file__).resolve().parents[1] / "docs" / "config.linux-portal.json"
    cfg = load_config(default_config(), sample, env={})
    assert cfg.acquisition.capture_backend == "portal"
    assert json.loads(sample.read_text())["acquisition"]["portal_source_type"] == "window"
