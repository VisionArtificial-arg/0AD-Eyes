"""EPIC D — ClassicalMinimapReader port-conformance and orchestration tests."""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import MinimapReader
from zero_ad_eyes.application.settings import (
    BgrColorSetting,
    BlipSettings,
    FogSettings,
    MinimapPaletteEntry,
    MinimapSettings,
    TerritorySettings,
    ViewportSettings,
    WorldExtentSettings,
)
from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.minimap import ClassicalMinimapReader


def _settings() -> MinimapSettings:
    return MinimapSettings(
        palette=(
            MinimapPaletteEntry(
                label="self", color=BgrColorSetting(b=235, g=90, r=40), ownership=Ownership.SELF
            ),
            MinimapPaletteEntry(
                label="ally", color=BgrColorSetting(b=60, g=200, r=60), ownership=Ownership.ALLY
            ),
            MinimapPaletteEntry(
                label="enemy", color=BgrColorSetting(b=40, g=40, r=220), ownership=Ownership.ENEMY
            ),
            MinimapPaletteEntry(
                label="gaia",
                color=BgrColorSetting(b=235, g=235, r=235),
                ownership=Ownership.GAIA,
            ),
        ),
        world_extent=WorldExtentSettings(
            origin_x=0.0, origin_y=0.0, width=1024.0, height=1024.0, flip_y=True
        ),
        fog=FogSettings(rows=16, cols=16, unexplored_max=25.0, visible_min=140.0),
        blips=BlipSettings(tolerance=70.0, min_area=1, max_area=60, confidence=0.8),
        territory=TerritorySettings(tolerance=90.0, min_area=64),
        viewport=ViewportSettings(white_min=200, min_area=64, min_side=8),
        disc_shape=False,
        region_confidence=0.9,
    )


def _reader() -> ClassicalMinimapReader:
    return ClassicalMinimapReader.from_settings(_settings())


def _frame(image: np.ndarray, frame_id: int = 0) -> Frame:
    h, w = image.shape[:2]
    return Frame(
        image=image,
        meta=FrameMeta(
            frame_id=frame_id, timestamp=float(frame_id), source="test", width=w, height=h
        ),
    )


def test_reader_satisfies_the_minimap_reader_port() -> None:
    assert isinstance(_reader(), MinimapReader)


def test_returns_classical_model_when_region_present() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    calibration = Calibration(
        width=300, height=200, minimap=ScreenBBox(x=10, y=150, width=48, height=48)
    )

    model = _reader().read(_frame(image), calibration)

    assert model.confidence.provenance is Provenance.CLASSICAL
    assert model.confidence.value > 0.0


def test_read_populates_domain_fog_when_region_present() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[150:198, 10:58] = 220  # a bright (fully-visible) minimap crop
    calibration = Calibration(
        width=300, height=200, minimap=ScreenBBox(x=10, y=150, width=48, height=48)
    )

    model = _reader().read(_frame(image), calibration)

    assert model.fog is not None
    assert model.fog.rows * model.fog.cols == len(model.fog.cells) * len(model.fog.cells[0])


def test_read_populates_domain_territory_when_region_present() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[150:198, 10:58] = 220
    calibration = Calibration(
        width=300, height=200, minimap=ScreenBBox(x=10, y=150, width=48, height=48)
    )

    model = _reader().read(_frame(image), calibration)

    # An empty territory map is valid; the contract is that the field is populated
    # (not None) whenever the minimap is calibrated, with coverage in [0, 1].
    assert model.territory is not None
    assert all(0.0 <= region.coverage <= 1.0 for region in model.territory.regions)


def test_returns_unknown_when_no_minimap_calibrated() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    calibration = Calibration(width=300, height=200, minimap=None)

    model = _reader().read(_frame(image), calibration)

    assert model.blips == ()
    assert model.viewport is None
    assert model.confidence.provenance is Provenance.UNKNOWN


def test_side_channels_return_none_without_calibration() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    calibration = Calibration(width=300, height=200, minimap=None)
    reader = _reader()

    frame = _frame(image)
    assert reader.read_territory(frame, calibration) is None
    assert reader.read_fog(frame, calibration) is None


def test_fog_side_channel_reads_the_calibrated_region() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[150:198, 10:58] = 220  # a fully-visible (bright) minimap crop
    calibration = Calibration(
        width=300, height=200, minimap=ScreenBBox(x=10, y=150, width=48, height=48)
    )

    grid = _reader().read_fog(_frame(image), calibration)

    assert grid is not None
    assert grid.rows * grid.cols > 0
