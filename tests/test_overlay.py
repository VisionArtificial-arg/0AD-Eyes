"""Tests for the rich debug overlay (REQUIREMENTS.md X1).

Every test renders onto a synthetic black frame and asserts on shape/dtype and on
*pixels changing* for each drawn element — no display is ever opened (headless).
"""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.domain.confidence import Confidence
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.hud import HudState, Population
from zero_ad_eyes.domain.minimap import Blip, FogState, MinimapModel, ViewportRect
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership, Phase, ResourceType
from zero_ad_eyes.domain.world_model import WorldModel
from zero_ad_eyes.interface.overlay import render

from .conftest import make_frame


def _changed(canvas: np.ndarray, frame_image: np.ndarray) -> bool:
    return bool(np.any(canvas != frame_image))


def test_basic_signature_still_works_and_preserves_shape_dtype() -> None:
    frame = make_frame(width=200, height=150)
    wm = WorldModel(meta=frame.meta)

    canvas = render(frame, wm)  # original two-argument call

    assert canvas.shape == frame.image.shape
    assert canvas.dtype == frame.image.dtype


def test_entities_are_drawn() -> None:
    frame = make_frame(width=200, height=150)
    wm = WorldModel(
        meta=frame.meta,
        entities=(
            Entity(
                entity_id=7,
                kind=EntityKind.UNIT,
                ownership=Ownership.ENEMY,
                entity_type="cavalry",
                screen_bbox=ScreenBBox(x=40, y=40, width=30, height=20),
            ),
        ),
    )

    canvas = render(frame, wm)

    assert _changed(canvas, frame.image)


def test_owner_colour_tints_the_box() -> None:
    frame = make_frame(width=200, height=150)
    wm = WorldModel(
        meta=frame.meta,
        entities=(
            Entity(
                entity_id=1,
                kind=EntityKind.UNIT,
                ownership=Ownership.ENEMY,  # default palette RGB (230, 25, 75)
                screen_bbox=ScreenBBox(x=50, y=50, width=40, height=30),
            ),
        ),
    )

    canvas = render(frame, wm)

    # RGB (230, 25, 75) -> BGR (75, 25, 230): the red channel must appear somewhere.
    assert bool(np.any(canvas[:, :, 2] == 230))


def test_health_bar_is_drawn() -> None:
    frame = make_frame(width=200, height=150)
    box = ScreenBBox(x=60, y=60, width=40, height=25)
    healthy = WorldModel(
        meta=frame.meta,
        entities=(Entity(entity_id=1, kind=EntityKind.UNIT, screen_bbox=box, health=1.0),),
    )
    hurt = WorldModel(
        meta=frame.meta,
        entities=(Entity(entity_id=1, kind=EntityKind.UNIT, screen_bbox=box, health=0.1),),
    )

    healthy_canvas = render(frame, healthy)
    hurt_canvas = render(frame, hurt)

    # Different health fractions must paint the bar differently.
    assert _changed(healthy_canvas, hurt_canvas)


def test_hud_summary_is_drawn() -> None:
    frame = make_frame(width=240, height=180)
    hud = HudState(
        stockpiles={ResourceType.FOOD: 120, ResourceType.WOOD: 80},
        population=Population(current=12, cap=20),
        phase=Phase.TOWN,
        self_civ="athen",
        confidence=Confidence.unknown(),
    )
    with_hud = WorldModel(meta=frame.meta, hud=hud)
    without_hud = WorldModel(meta=frame.meta)

    with_canvas = render(frame, with_hud)
    without_canvas = render(frame, without_hud)

    assert _changed(with_canvas, without_canvas)


def test_minimap_blips_and_viewport_are_drawn() -> None:
    frame = make_frame(width=240, height=180)
    minimap = MinimapModel(
        blips=(
            Blip(
                world_pos=WorldPoint(x=0.0, y=0.0),
                ownership=Ownership.SELF,
                confidence=Confidence.unknown(),
            ),
            Blip(
                world_pos=WorldPoint(x=10.0, y=10.0),
                ownership=Ownership.ENEMY,
                confidence=Confidence.unknown(),
            ),
        ),
        viewport=ViewportRect(
            top_left=WorldPoint(x=2.0, y=2.0),
            bottom_right=WorldPoint(x=8.0, y=8.0),
        ),
        confidence=Confidence.unknown(),
    )
    wm = WorldModel(meta=frame.meta, minimap=minimap)

    canvas = render(frame, wm)

    assert _changed(canvas, frame.image)


def test_fog_grid_is_drawn() -> None:
    frame = make_frame(width=240, height=180)
    wm = WorldModel(meta=frame.meta)
    fog = [
        [FogState.UNEXPLORED, FogState.EXPLORED],
        [FogState.VISIBLE, FogState.VISIBLE],
    ]

    without_fog = render(frame, wm)
    with_fog = render(frame, wm, fog=fog)

    assert _changed(with_fog, without_fog)
