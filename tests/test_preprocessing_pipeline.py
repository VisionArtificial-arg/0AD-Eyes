"""P1 — composable pipeline scaffold tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import Preprocessor
from zero_ad_eyes.infrastructure.preprocessing import PreprocessingPipeline

from .preprocessing_support import make_pattern_frame


class _AddOne:
    def __call__(self, frame: Frame) -> Frame:
        return replace(frame, image=frame.image + 1)


class _RecordFirstPixel:
    def __init__(self) -> None:
        self.seen: int | None = None

    def __call__(self, frame: Frame) -> Frame:
        self.seen = int(frame.image[0, 0, 0])
        return frame


def test_empty_pipeline_is_identity() -> None:
    frame = make_pattern_frame()
    out = PreprocessingPipeline().process(frame)
    assert out.meta == frame.meta
    assert np.array_equal(out.image, frame.image)


def test_pipeline_preserves_meta() -> None:
    frame = make_pattern_frame()
    out = PreprocessingPipeline([_AddOne()]).process(frame)
    assert out.meta == frame.meta
    assert out.image.shape == frame.image.shape


def test_pipeline_applies_steps_in_order() -> None:
    frame = make_pattern_frame()
    recorder = _RecordFirstPixel()
    base = int(frame.image[0, 0, 0])
    PreprocessingPipeline([_AddOne(), recorder]).process(frame)
    assert recorder.seen == base + 1


def test_pipeline_is_a_composable_step() -> None:
    frame = make_pattern_frame()
    inner = PreprocessingPipeline([_AddOne()])
    outer = PreprocessingPipeline([inner, _AddOne()])
    out = outer.process(frame)
    assert int(out.image[0, 0, 0]) == int(frame.image[0, 0, 0]) + 2


def test_then_appends_without_mutating() -> None:
    original = PreprocessingPipeline([_AddOne()])
    extended = original.then(_AddOne())
    assert len(original.steps) == 1
    assert len(extended.steps) == 2


def test_pipeline_satisfies_preprocessor_port() -> None:
    assert isinstance(PreprocessingPipeline(), Preprocessor)


def test_call_equals_process() -> None:
    frame = make_pattern_frame()
    pipeline = PreprocessingPipeline([_AddOne()])
    assert np.array_equal(pipeline(frame).image, pipeline.process(frame).image)
