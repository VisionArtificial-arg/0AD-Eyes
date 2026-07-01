"""Selection-panel reader tests (EPIC C — C5, best-effort).

Exercised via the injected :class:`MarkerOcr`; no tesseract binary required.
"""

from __future__ import annotations

from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.hud.layout import SelectionPanelLayout
from zero_ad_eyes.infrastructure.hud.reader import ClassicalHudReader

from .support import build_hud

_PANEL = ScreenBBox(x=0, y=0, width=200, height=80)


def _calibration(panel: ScreenBBox | None) -> Calibration:
    return Calibration(width=200, height=80, selection_panel=panel)


def test_reads_type_health_and_queue() -> None:
    layout = SelectionPanelLayout()
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
    reader = ClassicalHudReader(ocr, selection_layout=layout)

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
    layout = SelectionPanelLayout()
    frame, ocr = build_hud(
        _PANEL,
        [(layout.entity_type, "Spearman"), (layout.health, "45/100")],
        width=200,
        height=80,
    )
    reader = ClassicalHudReader(ocr, selection_layout=layout)
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
    reader = ClassicalHudReader(ocr)
    calibration = Calibration(
        width=200, height=80, top_bar=ScreenBBox(x=0, y=0, width=200, height=24)
    )

    assert reader.read(frame, calibration).selection is None


def test_empty_selection_is_unknown() -> None:
    frame, ocr = build_hud(_PANEL, [], width=200, height=80)
    reader = ClassicalHudReader(ocr)

    sel = reader.read_selection(frame, _calibration(_PANEL))

    assert sel.entity_type is None
    assert sel.health is None
    assert sel.production_queue == ()
    assert sel.confidence.value == 0.0


def test_no_panel_calibration_returns_default() -> None:
    frame, ocr = build_hud(_PANEL, [], width=200, height=80)
    reader = ClassicalHudReader(ocr)

    sel = reader.read_selection(frame, _calibration(None))

    assert sel.entity_type is None
    assert sel.confidence.provenance is Provenance.UNKNOWN
