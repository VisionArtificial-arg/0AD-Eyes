"""``ClassicalHudReader`` — the EPIC C adapter (C1).

Implements the :class:`~zero_ad_eyes.application.ports.HudReader` port with a
purely classical, deterministic pipeline: crop each HUD sub-region from the
calibration, OCR it through the injected :class:`OcrEngine`, and parse the text
into domain values. Provenance is always ``CLASSICAL`` (this reader never uses the
learned model). It degrades gracefully — a missing calibration or unreadable crop
yields low confidence, never an exception (NF4).

This file grows one task at a time; C1 populates the four resource stockpiles.
"""

from __future__ import annotations

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.hud import HudState, Population
from zero_ad_eyes.domain.hud import SelectionState as DomainSelectionState
from zero_ad_eyes.domain.taxonomy import Phase, ResourceType

from .cropping import crop, sample_color_rgb
from .layout import FractionalRegion, SelectionPanelLayout, TopBarLayout
from .ocr import OcrEngine, TesseractOcrEngine
from .parsing import (
    normalize_civ,
    normalize_phase,
    parse_count,
    parse_health,
    parse_population,
)
from .selection import HealthReading, SelectionState


class ClassicalHudReader:
    """Reads the top bar / self-identification from pixels (satisfies HudReader)."""

    def __init__(
        self,
        ocr: OcrEngine | None = None,
        *,
        top_bar_layout: TopBarLayout | None = None,
        selection_layout: SelectionPanelLayout | None = None,
    ) -> None:
        self._ocr = ocr if ocr is not None else TesseractOcrEngine()
        self._top_bar = top_bar_layout if top_bar_layout is not None else TopBarLayout()
        self._selection = (
            selection_layout if selection_layout is not None else SelectionPanelLayout()
        )

    def read(self, frame: Frame, calibration: Calibration) -> HudState:
        top_bar = calibration.top_bar
        if top_bar is None:
            # No calibration for the bar → nothing observable (NF4).
            return HudState(confidence=Confidence.unknown())

        stockpiles = self._read_resources(frame, top_bar)
        population = self._read_population(frame, top_bar)
        phase = self._read_phase(frame, top_bar)
        self_color = sample_color_rgb(frame.image, self._top_bar.swatch.project(top_bar))
        self_civ = normalize_civ(self._ocr_region(frame, self._top_bar.civ.project(top_bar)))

        # Confidence is the fraction of the fields we attempted that we could read;
        # each task folds its field into this score.
        read = (
            len(stockpiles)
            + (1 if population is not None else 0)
            + (1 if phase is not Phase.UNKNOWN else 0)
            + (1 if self_color is not None else 0)
            + (1 if self_civ is not None else 0)
        )
        attempted = 4 + 1 + 1 + 1 + 1

        return HudState(
            stockpiles=stockpiles,
            population=population,
            phase=phase,
            self_player_color=self_color,
            self_civ=self_civ,
            selection=self._selection_domain(frame, calibration),
            confidence=Confidence(value=read / attempted, provenance=Provenance.CLASSICAL),
        )

    def _selection_domain(
        self, frame: Frame, calibration: Calibration
    ) -> DomainSelectionState | None:
        """v0.2: fold the selection-panel read into ``HudState.selection``.

        Maps the reader's local reading onto the domain value object: health is
        emitted as a fraction (the domain contract), not raw hit points. Returns
        ``None`` when nothing is selected (or the panel is uncalibrated) so a bare
        HUD carries no phantom selection.
        """

        reading = self.read_selection(frame, calibration)
        if reading.entity_type is None and reading.health is None and not reading.production_queue:
            return None
        return DomainSelectionState(
            entity_type=reading.entity_type,
            health=reading.health.fraction if reading.health is not None else None,
            production_queue=reading.production_queue,
            confidence=reading.confidence,
        )

    def read_selection(self, frame: Frame, calibration: Calibration) -> SelectionState:
        """Best-effort read of the selection panel (C5, ``[S]``).

        Returned *alongside* ``read`` because the ``HudState`` port type has no
        selection field; see the module docstring. Degrades to an empty, unknown
        state when the panel is not calibrated or nothing is selected (NF4).
        """

        panel = calibration.selection_panel
        if panel is None:
            return SelectionState()

        raw_type = self._ocr_region(frame, self._selection.entity_type.project(panel)).strip()
        entity_type = raw_type or None

        hp = parse_health(self._ocr_region(frame, self._selection.health.project(panel)))
        health = HealthReading(current=hp[0], maximum=hp[1]) if hp and hp[1] > 0 else None

        queue_text = self._ocr_region(frame, self._selection.queue.project(panel))
        production_queue = tuple(token for token in queue_text.split() if token)

        read = sum(
            (
                entity_type is not None,
                health is not None,
                bool(production_queue),
            )
        )
        confidence = Confidence(value=read / 3.0, provenance=Provenance.CLASSICAL)
        return SelectionState(
            entity_type=entity_type,
            health=health,
            production_queue=production_queue,
            confidence=confidence,
        )

    def _read_resources(self, frame: Frame, top_bar: ScreenBBox) -> dict[ResourceType, int]:
        regions: dict[ResourceType, FractionalRegion] = {
            ResourceType.FOOD: self._top_bar.food,
            ResourceType.WOOD: self._top_bar.wood,
            ResourceType.STONE: self._top_bar.stone,
            ResourceType.METAL: self._top_bar.metal,
        }
        stockpiles: dict[ResourceType, int] = {}
        for resource, region in regions.items():
            value = parse_count(self._ocr_region(frame, region.project(top_bar)))
            if value is not None:
                stockpiles[resource] = value
        return stockpiles

    def _read_population(self, frame: Frame, top_bar: ScreenBBox) -> Population | None:
        bbox = self._top_bar.population.project(top_bar)
        parsed = parse_population(self._ocr_region(frame, bbox))
        if parsed is None:
            return None
        current, cap = parsed
        return Population(current=current, cap=cap)

    def _read_phase(self, frame: Frame, top_bar: ScreenBBox) -> Phase:
        bbox = self._top_bar.phase.project(top_bar)
        return normalize_phase(self._ocr_region(frame, bbox))

    def _ocr_region(self, frame: Frame, bbox: ScreenBBox) -> str:
        return self._ocr.read_text(crop(frame.image, bbox))
