"""EntityEnricher — classical ownership/health/selection fill-if-absent (EPIC E → G)."""

from __future__ import annotations

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import EntityEnricher
from zero_ad_eyes.application.settings import (
    ConstructionCueSettings,
    GarrisonCueSettings,
    HealthReadSettings,
    SelectionCueSettings,
    StateCueSettings,
)
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership
from zero_ad_eyes.domain.world_model import FrameMeta
from zero_ad_eyes.infrastructure.perception.enrichment import ClassicalEntityEnricher
from zero_ad_eyes.infrastructure.perception.palette import HsvBand, PlayerColor, PlayerPalette

BLUE = (200, 40, 40)  # lands in the SELF hue band (see perception/test_ownership)


def _enricher() -> ClassicalEntityEnricher:
    """The enricher wired with the historical defaults, spelled out explicitly."""

    return ClassicalEntityEnricher(
        ownership_palette=PlayerPalette(
            colors=(
                PlayerColor(
                    name="blue",
                    ownership=Ownership.SELF,
                    bands=(HsvBand(h_lo=100, h_hi=130, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
                PlayerColor(
                    name="green",
                    ownership=Ownership.ALLY,
                    bands=(HsvBand(h_lo=45, h_hi=85, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
                PlayerColor(
                    name="red",
                    ownership=Ownership.ENEMY,
                    bands=(
                        HsvBand(h_lo=0, h_hi=10, s_lo=70, s_hi=255, v_lo=50, v_hi=255),
                        HsvBand(h_lo=170, h_hi=179, s_lo=70, s_hi=255, v_lo=50, v_hi=255),
                    ),
                ),
                PlayerColor(
                    name="yellow",
                    ownership=Ownership.GAIA,
                    bands=(HsvBand(h_lo=22, h_hi=34, s_lo=70, s_hi=255, v_lo=50, v_hi=255),),
                ),
            )
        ),
        ownership_min_fraction=0.02,
        health=HealthReadSettings(max_offset=20, s_min=60, v_min=60, min_run=0.15),
        state=StateCueSettings(
            selection=SelectionCueSettings(thickness=3, brightness=200, min_fraction=0.4),
            construction=ConstructionCueSettings(
                edge_density_min=0.12, canny_lo=60.0, canny_hi=180.0
            ),
            garrison=GarrisonCueSettings(
                top_fraction=0.35,
                brightness=200,
                max_saturation=70,
                min_badge_area=6,
                max_badge_width_fraction=0.5,
            ),
        ),
    )


def _frame_with_box() -> Frame:
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (50, 50), BLUE, -1)
    return Frame(
        image=image,
        meta=FrameMeta(frame_id=1, timestamp=1.0, source="test", width=120, height=90),
    )


def _entity(**kwargs: object) -> Entity:
    base: dict[str, object] = {
        "entity_id": 0,
        "kind": EntityKind.UNIT,
        "screen_bbox": ScreenBBox(x=10, y=10, width=40, height=40),
    }
    base.update(kwargs)
    return Entity(**base)


def test_enricher_conforms_to_port() -> None:
    assert isinstance(_enricher(), EntityEnricher)


def test_fills_ownership_when_unknown() -> None:
    enriched = _enricher().enrich((_entity(ownership=Ownership.UNKNOWN),), _frame_with_box())
    assert enriched[0].ownership is Ownership.SELF


def test_does_not_clobber_existing_ownership() -> None:
    # A confident prior owner (e.g. from a learned model) must survive enrichment.
    enriched = _enricher().enrich((_entity(ownership=Ownership.ENEMY),), _frame_with_box())
    assert enriched[0].ownership is Ownership.ENEMY


def test_entity_without_bbox_is_returned_untouched() -> None:
    entity = Entity(entity_id=7, kind=EntityKind.UNIT)
    enriched = _enricher().enrich((entity,), _frame_with_box())
    assert enriched[0] is entity


def test_unknown_stays_unknown_on_black_frame() -> None:
    black = Frame(
        image=np.zeros((90, 120, 3), dtype=np.uint8),
        meta=FrameMeta(frame_id=1, timestamp=1.0, source="test", width=120, height=90),
    )
    enriched = _enricher().enrich((_entity(ownership=Ownership.UNKNOWN),), black)
    assert enriched[0].ownership is Ownership.UNKNOWN  # honest "cannot tell", no fabrication
    assert enriched[0].health is None
