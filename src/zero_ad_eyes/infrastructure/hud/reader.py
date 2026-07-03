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

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import HudSettings
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

_DIGITS = re.compile(r"\d+")


class ClassicalHudReader:
    """Reads the top bar / self-identification from pixels (satisfies HudReader)."""

    def __init__(
        self,
        ocr: OcrEngine,
        *,
        top_bar_layout: TopBarLayout,
        selection_layout: SelectionPanelLayout,
    ) -> None:
        self._ocr = ocr
        self._top_bar = top_bar_layout
        self._selection = selection_layout

    @classmethod
    def from_settings(cls, settings: HudSettings) -> ClassicalHudReader:
        """Build from pure config (Approach B). Field names of the settings mirror the
        infra layout types 1:1, so the boundary mapping is a ``model_validate`` of the
        dumped data — no per-field wiring to drift."""

        return cls(
            TesseractOcrEngine(config=settings.ocr_config),
            top_bar_layout=TopBarLayout.model_validate(settings.top_bar.model_dump()),
            selection_layout=SelectionPanelLayout.model_validate(settings.selection.model_dump()),
        )

    def read(self, frame: Frame, calibration: Calibration) -> HudState:
        top_bar = calibration.top_bar
        if top_bar is None:
            # No calibration for the bar → nothing observable (NF4).
            return HudState(confidence=Confidence.unknown())

        stockpiles = self._read_resources(frame, calibration)
        population = self._read_population(frame, calibration)
        phase = self._read_phase(frame, calibration)
        self_color = sample_color_rgb(frame.image, self._top_region(calibration, "swatch"))
        self_civ = normalize_civ(self._ocr_region(frame, self._top_region(calibration, "civ")))

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

    def _read_resources(self, frame: Frame, calibration: Calibration) -> dict[ResourceType, int]:
        regions: dict[ResourceType, str] = {
            ResourceType.FOOD: "food",
            ResourceType.WOOD: "wood",
            ResourceType.STONE: "stone",
            ResourceType.METAL: "metal",
        }
        stockpiles: dict[ResourceType, int] = {}
        for resource, name in regions.items():
            value = self._read_count(frame, self._top_region(calibration, name))
            if value is not None:
                stockpiles[resource] = value
        return stockpiles

    def _top_region(self, calibration: Calibration, name: str) -> ScreenBBox:
        manual = calibration.hud_regions.get(name)
        if manual is not None:
            return manual
        assert calibration.top_bar is not None
        regions: dict[str, FractionalRegion] = {
            "food": self._top_bar.food,
            "wood": self._top_bar.wood,
            "stone": self._top_bar.stone,
            "metal": self._top_bar.metal,
            "population": self._top_bar.population,
            "phase": self._top_bar.phase,
            "swatch": self._top_bar.swatch,
            "civ": self._top_bar.civ,
        }
        return regions[name].project(calibration.top_bar)

    def _read_population(self, frame: Frame, calibration: Calibration) -> Population | None:
        parsed = self._read_population_value(frame, self._top_region(calibration, "population"))
        if parsed is None:
            return None
        current, cap = parsed
        return Population(current=current, cap=cap)

    def _read_phase(self, frame: Frame, calibration: Calibration) -> Phase:
        return normalize_phase(self._ocr_region(frame, self._top_region(calibration, "phase")))

    def _ocr_region(self, frame: Frame, bbox: ScreenBBox) -> str:
        return self._ocr.read_text(crop(frame.image, bbox))

    def _read_count(self, frame: Frame, bbox: ScreenBBox) -> int | None:
        region = crop(frame.image, bbox)
        raw = self._ocr.read_text(region)
        parsed = parse_count(raw)
        if parsed is not None:
            return parsed

        candidates: list[tuple[int, int]] = []
        for text in self._numeric_ocr_texts(region, include_right_trims=True):
            value = parse_count(text)
            if value is not None:
                candidates.append((self._numeric_score(text), value))
        if not candidates:
            return None
        counts = Counter(value for _, value in candidates)
        return max(candidates, key=lambda candidate: (counts[candidate[1]], candidate[0]))[1]

    def _read_population_value(self, frame: Frame, bbox: ScreenBBox) -> tuple[int, int] | None:
        region = crop(frame.image, bbox)
        raw = self._ocr.read_text(region)
        parsed = parse_population(raw)
        if parsed is not None:
            return parsed

        candidates: list[tuple[int, tuple[int, int]]] = []
        for text in self._numeric_ocr_texts(region):
            parsed = parse_population(text)
            if parsed is not None:
                candidates.append((len(str(parsed[0])) + len(str(parsed[1])), parsed))
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate[0])[1]

    def _numeric_ocr_texts(self, image: Any, *, include_right_trims: bool = False) -> Iterable[str]:
        """Retry tiny HUD counters on OCR-friendly views of the same crop.

        Manual calibration boxes often include the resource icon beside the digits.
        Tesseract is much more reliable when the crop is enlarged, and the HSV mask
        gives it a second chance on bright HUD text without changing text fields.
        """

        if getattr(image, "size", 0) == 0:
            return ()

        def scaled(view: Any, *, nearest: bool = False) -> Any:
            interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
            return cv2.resize(view, None, fx=4, fy=4, interpolation=interpolation)

        variants = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        variants.append(scaled(gray))

        if image.ndim == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            text_mask = cv2.inRange(
                hsv,
                np.array((0, 0, 120), dtype=np.uint8),
                np.array((180, 95, 255), dtype=np.uint8),
            )
            variants.append(scaled(text_mask, nearest=True))

        if include_right_trims and image.shape[1] >= 40:
            for ratio in (0.28, 0.42, 0.50):
                start = int(round(image.shape[1] * ratio))
                trimmed = image[:, start:]
                trimmed_gray = (
                    cv2.cvtColor(trimmed, cv2.COLOR_BGR2GRAY) if trimmed.ndim == 3 else trimmed
                )
                variants.append(scaled(trimmed_gray))

        texts = []
        for variant in variants:
            try:
                texts.append(self._ocr.read_text(variant))
            except Exception:
                continue
        return texts

    @staticmethod
    def _numeric_score(text: str) -> int:
        runs = _DIGITS.findall(text)
        return max((len(run) for run in runs), default=0)
