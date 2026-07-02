"""Reader wiring tests for resource counters (EPIC C — C1).

Uses the injected :class:`MarkerOcr` so no tesseract binary is needed; the reader
crops each region, the marked crop drives canned OCR text, and parsing yields the
stockpiles.
"""

from __future__ import annotations

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.hud import Population
from zero_ad_eyes.domain.taxonomy import Phase, ResourceType
from zero_ad_eyes.infrastructure.hud.layout import (
    FractionalRegion,
    SelectionPanelLayout,
    TopBarLayout,
)
from zero_ad_eyes.infrastructure.hud.reader import ClassicalHudReader

from .support import build_hud

_TOP_BAR = ScreenBBox(x=0, y=0, width=240, height=24)


# Former in-code defaults, now hardcoded as explicit test literals.
def _top_bar_layout() -> TopBarLayout:
    return TopBarLayout(
        food=FractionalRegion(x=0.03, y=0.0, width=0.10, height=1.0),
        wood=FractionalRegion(x=0.15, y=0.0, width=0.10, height=1.0),
        stone=FractionalRegion(x=0.27, y=0.0, width=0.10, height=1.0),
        metal=FractionalRegion(x=0.39, y=0.0, width=0.10, height=1.0),
        population=FractionalRegion(x=0.51, y=0.0, width=0.12, height=1.0),
        phase=FractionalRegion(x=0.80, y=0.0, width=0.18, height=1.0),
        swatch=FractionalRegion(x=0.0, y=0.2, width=0.025, height=0.6),
        civ=FractionalRegion(x=0.66, y=0.0, width=0.13, height=1.0),
    )


def _selection_layout() -> SelectionPanelLayout:
    return SelectionPanelLayout(
        entity_type=FractionalRegion(x=0.05, y=0.05, width=0.9, height=0.25),
        health=FractionalRegion(x=0.05, y=0.35, width=0.5, height=0.25),
        queue=FractionalRegion(x=0.05, y=0.7, width=0.9, height=0.28),
    )


def _calibration() -> Calibration:
    return Calibration(width=240, height=24, top_bar=_TOP_BAR)


def test_reads_all_four_resource_counters() -> None:
    layout = _top_bar_layout()
    frame, ocr = build_hud(
        _TOP_BAR,
        [
            (layout.food, "350"),
            (layout.wood, "1,200"),
            (layout.stone, "0"),
            (layout.metal, "75"),
            (layout.population, "15/20"),
            (layout.phase, "Town Phase"),
            (layout.civ, "Romans"),
        ],
    )
    reader = ClassicalHudReader(ocr, top_bar_layout=layout, selection_layout=_selection_layout())

    hud = reader.read(frame, _calibration())

    assert hud.stockpiles == {
        ResourceType.FOOD: 350,
        ResourceType.WOOD: 1200,
        ResourceType.STONE: 0,
        ResourceType.METAL: 75,
    }
    assert hud.population == Population(current=15, cap=20)
    assert hud.phase is Phase.TOWN
    assert hud.self_civ == "rome"
    assert hud.self_player_color is not None
    assert hud.confidence.value == 1.0
    assert hud.confidence.provenance is Provenance.CLASSICAL


def test_partial_read_lowers_confidence_and_omits_missing() -> None:
    layout = _top_bar_layout()
    frame, ocr = build_hud(
        _TOP_BAR,
        [
            (layout.food, "350"),
            (layout.wood, "200"),
            # stone/metal/population unpainted → OCR "" → parsed None → omitted
        ],
    )
    reader = ClassicalHudReader(ocr, top_bar_layout=layout, selection_layout=_selection_layout())

    hud = reader.read(frame, _calibration())

    assert set(hud.stockpiles) == {ResourceType.FOOD, ResourceType.WOOD}
    assert hud.population is None
    assert hud.phase is Phase.UNKNOWN
    assert hud.self_civ is None
    # 2 resources + a (default black) swatch colour are read; pop/phase/civ absent.
    assert hud.confidence.value == 3 / 8


def test_missing_calibration_yields_unknown() -> None:
    reader = ClassicalHudReader(
        build_hud(_TOP_BAR, [])[1],
        top_bar_layout=_top_bar_layout(),
        selection_layout=_selection_layout(),
    )
    frame = build_hud(_TOP_BAR, [])[0]

    hud = reader.read(frame, Calibration(width=240, height=24, top_bar=None))

    assert hud.stockpiles == {}
    assert hud.confidence.provenance is Provenance.UNKNOWN
