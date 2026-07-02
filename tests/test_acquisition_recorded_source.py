"""Tests for RecordedVideoSource — replay a recording with its true clock (A5, #2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zero_ad_eyes.infrastructure.acquisition import (
    FrameStamp,
    InMemoryFrameSource,
    RecordedVideoSource,
    RecordingManifest,
    VideoFileSource,
)
from zero_ad_eyes.interface.cli import _offline_source
from zero_ad_eyes.interface.default_config import default_config

from .conftest import make_frame


def _manifest(stamps: list[tuple[int, float]], *, source: str = "live") -> RecordingManifest:
    return RecordingManifest(
        video="live.mkv",
        source=source,
        fps=30.0,
        stamps=tuple(FrameStamp(frame_id=fid, timestamp=ts) for fid, ts in stamps),
    )


def test_restamps_replayed_frames_with_the_sidecar_clock(tmp_path: Path) -> None:
    # The inner video (as VideoFileSource would) renumbers 0..N with PTS timestamps;
    # the sidecar restores the capture's real, non-zero-based clock and source name.
    inner = InMemoryFrameSource([make_frame(0), make_frame(1), make_frame(2)])
    manifest = _manifest([(100, 12.5), (101, 12.9), (102, 13.3)], source="live")

    source = RecordedVideoSource(tmp_path / "live.mkv", video=inner, manifest=manifest)
    out = list(source.frames())

    assert [(f.meta.frame_id, f.meta.timestamp) for f in out] == [
        (100, 12.5),
        (101, 12.9),
        (102, 13.3),
    ]
    assert all(f.meta.source == "live" for f in out)
    # Pixels + geometry are untouched — only the clock is restored.
    assert all(f.meta.width == 64 and f.meta.height == 48 for f in out)


def test_loads_the_sidecar_from_the_sibling_json_by_default(tmp_path: Path) -> None:
    video_path = tmp_path / "live.mkv"
    _manifest([(7, 0.7), (8, 0.8)]).save(video_path.with_suffix(".json"))
    inner = InMemoryFrameSource([make_frame(0), make_frame(1)])

    source = RecordedVideoSource(video_path, video=inner)  # manifest auto-loaded from disk
    out = list(source.frames())

    assert [f.meta.frame_id for f in out] == [7, 8]


def test_raises_when_video_has_more_frames_than_the_sidecar(tmp_path: Path) -> None:
    inner = InMemoryFrameSource([make_frame(0), make_frame(1), make_frame(2)])
    manifest = _manifest([(0, 0.0), (1, 0.1)])  # only two stamps for three frames

    source = RecordedVideoSource(tmp_path / "live.mkv", video=inner, manifest=manifest)
    with pytest.raises(OSError, match="out of step"):
        list(source.frames())


def test_raises_when_sidecar_has_more_frames_than_the_video(tmp_path: Path) -> None:
    inner = InMemoryFrameSource([make_frame(0)])
    manifest = _manifest([(0, 0.0), (1, 0.1)])  # sidecar lists two, video yields one

    source = RecordedVideoSource(tmp_path / "live.mkv", video=inner, manifest=manifest)
    with pytest.raises(OSError, match="out of step"):
        list(source.frames())


def test_offline_source_selects_recorded_source_when_a_sidecar_is_present(tmp_path: Path) -> None:
    # eval/run --recording auto-detects the sidecar and replays with the true clock.
    video = tmp_path / "live.mkv"
    _manifest([(0, 0.0)]).save(video.with_suffix(".json"))

    source = _offline_source(str(video), default_config().acquisition)

    assert isinstance(source, RecordedVideoSource)


def test_offline_source_falls_back_to_plain_video_without_a_sidecar(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"  # no sibling .json

    source = _offline_source(str(video), default_config().acquisition)

    assert isinstance(source, VideoFileSource)
