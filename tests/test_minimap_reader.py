"""EPIC D — ClassicalMinimapReader port-conformance and orchestration tests."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import MinimapReader
from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.minimap import ClassicalMinimapReader


def _frame(image: np.ndarray, frame_id: int = 0) -> Frame:
    h, w = image.shape[:2]
    return Frame(
        image=image,
        meta=FrameMeta(
            frame_id=frame_id, timestamp=float(frame_id), source="test", width=w, height=h
        ),
    )


def test_reader_satisfies_the_minimap_reader_port() -> None:
    assert isinstance(ClassicalMinimapReader(), MinimapReader)


def test_returns_classical_model_when_region_present() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    calibration = Calibration(
        width=300, height=200, minimap=ScreenBBox(x=10, y=150, width=48, height=48)
    )

    model = ClassicalMinimapReader().read(_frame(image), calibration)

    assert model.confidence.provenance is Provenance.CLASSICAL
    assert model.confidence.value > 0.0


def test_read_populates_domain_fog_when_region_present() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[150:198, 10:58] = 220  # a bright (fully-visible) minimap crop
    calibration = Calibration(
        width=300, height=200, minimap=ScreenBBox(x=10, y=150, width=48, height=48)
    )

    model = ClassicalMinimapReader().read(_frame(image), calibration)

    assert model.fog is not None
    assert model.fog.rows * model.fog.cols == len(model.fog.cells) * len(model.fog.cells[0])


def test_returns_unknown_when_no_minimap_calibrated() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    calibration = Calibration(width=300, height=200, minimap=None)

    model = ClassicalMinimapReader().read(_frame(image), calibration)

    assert model.blips == ()
    assert model.viewport is None
    assert model.confidence.provenance is Provenance.UNKNOWN


def test_side_channels_return_none_without_calibration() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    calibration = Calibration(width=300, height=200, minimap=None)
    reader = ClassicalMinimapReader()

    frame = _frame(image)
    assert reader.read_territory(frame, calibration) is None
    assert reader.read_fog(frame, calibration) is None


def test_fog_side_channel_reads_the_calibrated_region() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[150:198, 10:58] = 220  # a fully-visible (bright) minimap crop
    calibration = Calibration(
        width=300, height=200, minimap=ScreenBBox(x=10, y=150, width=48, height=48)
    )

    grid = ClassicalMinimapReader().read_fog(_frame(image), calibration)

    assert grid is not None
    assert grid.rows * grid.cols > 0
