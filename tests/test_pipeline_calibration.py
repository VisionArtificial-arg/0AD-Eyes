"""Pipeline calibration reuse (B3): calibrate once per resolution, not per frame."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.pipeline import PerceptionPipeline
from zero_ad_eyes.domain.calibration import Calibration, CalibrationCheck
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


class _CountingSelfCheck:
    """A LayoutChecker that counts verifications and returns a fixed verdict."""

    def __init__(self, *, matches: bool = True) -> None:
        self.calls = 0
        self._matches = matches

    def verify(self, frame: Frame, calibration: Calibration) -> CalibrationCheck:
        self.calls += 1
        return CalibrationCheck(matches=self._matches, confidence=1.0)


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


def test_self_check_runs_on_interval_not_every_frame() -> None:
    calibrator = _CountingCalibrator()
    self_check = _CountingSelfCheck(matches=True)
    frames = [_frame(8, i) for i in range(10)]
    pipeline = PerceptionPipeline(
        InMemoryFrameSource(frames),
        StubPerceptionModel(),
        calibrator=calibrator,
        self_check=self_check,
        recalibrate_interval=3,
    )

    list(pipeline.run())

    assert calibrator.calls == 1  # detected once; every self-check passes → no re-detect
    assert self_check.calls == 3  # checked at frames 3, 6, 9 — not every frame


def test_pipeline_recalibrates_when_self_check_reports_drift() -> None:
    calibrator = _CountingCalibrator()
    self_check = _CountingSelfCheck(matches=False)  # layout has drifted
    frames = [_frame(8, i) for i in range(4)]
    pipeline = PerceptionPipeline(
        InMemoryFrameSource(frames),
        StubPerceptionModel(),
        calibrator=calibrator,
        self_check=self_check,
        recalibrate_interval=2,
    )

    list(pipeline.run())

    assert self_check.calls == 1  # first check fires at frame 2
    assert calibrator.calls == 2  # initial detect + one re-detect after drift


def test_calibration_check_is_a_domain_value_object() -> None:
    from zero_ad_eyes.infrastructure.calibration import CalibrationCheck as InfraCheck

    # Moved into the domain; the calibration package re-exports the same class.
    assert InfraCheck is CalibrationCheck
    assert CalibrationCheck(matches=True, confidence=0.9).matches
