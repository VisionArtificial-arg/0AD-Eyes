"""Selection-panel reader tests (EPIC C — C5, best-effort).

Exercised via the injected :class:`MarkerOcr`; no tesseract binary required.
"""

from __future__ import annotations

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.hud.layout import (
    FractionalRegion,
    SelectionPanelLayout,
    TopBarLayout,
)
from zero_ad_eyes.infrastructure.hud.reader import ClassicalHudReader

from .support import build_hud

_PANEL = ScreenBBox(x=0, y=0, width=200, height=80)


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


def _calibration(panel: ScreenBBox | None) -> Calibration:
    return Calibration(width=200, height=80, selection_panel=panel)


def test_reads_type_health_and_queue() -> None:
    layout = _selection_layout()
    frame, ocr = build_hud(
        _PANEL,
        [
            (layout.entity_type, "Spearman"),
            (layout.health, "45/100"),
            (layout.queue, "House Barracks"),
        ],
        width=200,
        height=80,
    )
    reader = ClassicalHudReader(ocr, top_bar_layout=_top_bar_layout(), selection_layout=layout)

    sel = reader.read_selection(frame, _calibration(_PANEL))

    assert sel.entity_type == "Spearman"
    assert sel.health is not None
    assert sel.health.current == 45
    assert sel.health.fraction == 0.45
    assert sel.production_queue == ("House", "Barracks")
    assert sel.confidence.value == 1.0
    assert sel.confidence.provenance is Provenance.CLASSICAL


def test_read_folds_selection_into_hudstate_as_domain_value() -> None:
    # v0.2: the port's read() carries selection on HudState, health as a fraction.
    layout = _selection_layout()
    frame, ocr = build_hud(
        _PANEL,
        [(layout.entity_type, "Spearman"), (layout.health, "45/100")],
        width=200,
        height=80,
    )
    reader = ClassicalHudReader(ocr, top_bar_layout=_top_bar_layout(), selection_layout=layout)
    calibration = Calibration(
        width=200,
        height=80,
        top_bar=ScreenBBox(x=0, y=0, width=200, height=24),
        selection_panel=_PANEL,
    )

    hud = reader.read(frame, calibration)

    assert hud.selection is not None
    assert hud.selection.entity_type == "Spearman"
    assert hud.selection.health == 0.45  # domain contract: fraction, not raw HP


def test_read_without_selection_leaves_hudstate_selection_none() -> None:
    frame, ocr = build_hud(_PANEL, [], width=200, height=80)
    reader = ClassicalHudReader(
        ocr, top_bar_layout=_top_bar_layout(), selection_layout=_selection_layout()
    )
    calibration = Calibration(
        width=200, height=80, top_bar=ScreenBBox(x=0, y=0, width=200, height=24)
    )

    assert reader.read(frame, calibration).selection is None


def test_empty_selection_is_unknown() -> None:
    frame, ocr = build_hud(_PANEL, [], width=200, height=80)
    reader = ClassicalHudReader(
        ocr, top_bar_layout=_top_bar_layout(), selection_layout=_selection_layout()
    )

    sel = reader.read_selection(frame, _calibration(_PANEL))

    assert sel.entity_type is None
    assert sel.health is None
    assert sel.production_queue == ()
    assert sel.confidence.value == 0.0


def test_no_panel_calibration_returns_default() -> None:
    frame, ocr = build_hud(_PANEL, [], width=200, height=80)
    reader = ClassicalHudReader(
        ocr, top_bar_layout=_top_bar_layout(), selection_layout=_selection_layout()
    )

    sel = reader.read_selection(frame, _calibration(None))

    assert sel.entity_type is None
    assert sel.confidence.provenance is Provenance.UNKNOWN
