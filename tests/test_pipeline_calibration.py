"""Pipeline calibration reuse (B3): calibrate once per resolution, not per frame."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.pipeline import PerceptionPipeline
from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource
from zero_ad_eyes.infrastructure.model import StubPerceptionModel


class _CountingCalibrator:
    """A Calibrator that records how many times it actually re-detects the layout."""

    def __init__(self) -> None:
        self.calls = 0

    def calibrate(self, frame: Frame) -> Calibration:
        self.calls += 1
        height, width = frame.image.shape[:2]
        return Calibration(width=width, height=height)


def _frame(side: int, frame_id: int) -> Frame:
    return Frame(
        image=np.zeros((side, side, 3), dtype=np.uint8),
        meta=FrameMeta(
            frame_id=frame_id, timestamp=float(frame_id), source="t", width=side, height=side
        ),
    )


def _pipeline(frames: list[Frame], calibrator: _CountingCalibrator) -> PerceptionPipeline:
    return PerceptionPipeline(
        InMemoryFrameSource(frames), StubPerceptionModel(), calibrator=calibrator
    )


def test_calibration_is_reused_across_frames_of_the_same_resolution() -> None:
    calibrator = _CountingCalibrator()
    frames = [_frame(8, i) for i in range(4)]

    models = list(_pipeline(frames, calibrator).run())

    assert len(models) == 4
    assert calibrator.calls == 1  # detected once, reused for the remaining three


def test_calibration_recomputed_when_resolution_changes() -> None:
    calibrator = _CountingCalibrator()
    frames = [_frame(8, 0), _frame(8, 1), _frame(16, 2), _frame(16, 3)]

    list(_pipeline(frames, calibrator).run())

    assert calibrator.calls == 2  # once for 8x8, once when it switches to 16x16
