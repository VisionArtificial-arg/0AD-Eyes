"""EVT-4 — minimap events: blips, enemy in territory, territory, fog, camera."""

from __future__ import annotations

from zero_ad_eyes.application.frames import PerceptionContext
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.events import Event, EventKind, EventOrigin
from zero_ad_eyes.domain.geometry import WorldPoint
from zero_ad_eyes.domain.minimap import (
    Blip,
    FogGrid,
    FogState,
    MinimapModel,
    TerritoryMap,
    TerritoryRegion,
    ViewportQuad,
)
from zero_ad_eyes.domain.taxonomy import Ownership
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel
from zero_ad_eyes.infrastructure.minimap.events import MinimapEventSource
from zero_ad_eyes.interface.default_config import default_config

U, E, V = FogState.UNEXPLORED, FogState.EXPLORED, FogState.VISIBLE
SELF, ENEMY = Ownership.SELF, Ownership.ENEMY


def _blip(x: float, y: float, owner: Ownership = SELF) -> Blip:
    return Blip(
        world_pos=WorldPoint(x=x, y=y),
        ownership=owner,
        confidence=Confidence(value=0.7, provenance=Provenance.CLASSICAL),
    )


def _view(cx: float, cy: float) -> ViewportQuad:
    return ViewportQuad(
        top_left=WorldPoint(x=cx - 10, y=cy - 10),
        top_right=WorldPoint(x=cx + 10, y=cy - 10),
        bottom_right=WorldPoint(x=cx + 10, y=cy + 10),
        bottom_left=WorldPoint(x=cx - 10, y=cy + 10),
    )


def _home(coverage: float = 0.05) -> TerritoryMap:
    # 5% of a 1024² map as a disc: radius ≈ 129 world units around (200, 200).
    return TerritoryMap(
        regions=(
            TerritoryRegion(ownership=SELF, centroid=WorldPoint(x=200, y=200), coverage=coverage),
        )
    )


def _run(*minimaps: MinimapModel | None) -> list[list[Event]]:
    cfg = default_config()
    source = MinimapEventSource.from_settings(cfg.events, cfg.minimap.world_extent)
    out = []
    for i, minimap in enumerate(minimaps):
        meta = FrameMeta(frame_id=i, timestamp=float(i), source="t", width=8, height=8)
        out.append(list(source.events(WorldModel(meta=meta, minimap=minimap), PerceptionContext())))
    return out


def _kinds(events: list[Event]) -> list[EventKind]:
    return [e.kind for e in events]


def test_first_frame_is_a_baseline() -> None:
    assert _run(MinimapModel(blips=(_blip(1, 1),), viewport=_view(50, 50))) == [[]]


def test_moving_blip_is_the_same_blip_new_and_gone_ones_are_events() -> None:
    frames = _run(
        MinimapModel(blips=(_blip(100, 100), _blip(500, 500))),
        MinimapModel(blips=(_blip(105, 102), _blip(800, 800, ENEMY))),
    )
    events = frames[1]
    assert _kinds(events) == [EventKind.BLIP_APPEARED, EventKind.BLIP_VANISHED]
    appeared, vanished = events
    assert appeared.details == {"ownership": "enemy"} and appeared.world_pos == WorldPoint(
        x=800, y=800
    )
    assert vanished.world_pos == WorldPoint(x=500, y=500)
    assert appeared.origin is EventOrigin.MINIMAP and appeared.confidence.value == 0.7


def test_owner_change_is_not_the_same_blip() -> None:
    frames = _run(
        MinimapModel(blips=(_blip(100, 100),)), MinimapModel(blips=(_blip(100, 100, ENEMY),))
    )
    assert _kinds(frames[1]) == [EventKind.BLIP_APPEARED, EventKind.BLIP_VANISHED]


def test_enemy_entering_territory_fires_once() -> None:
    frames = _run(
        MinimapModel(blips=(_blip(400, 400, ENEMY),), territory=_home()),
        MinimapModel(blips=(_blip(395, 395, ENEMY),), territory=_home()),
        MinimapModel(blips=(_blip(280, 280, ENEMY),), territory=_home()),  # far jump: a new blip
        MinimapModel(blips=(_blip(278, 278, ENEMY),), territory=_home()),
    )
    assert frames[1] == []
    assert EventKind.ENEMY_IN_TERRITORY in _kinds(frames[2])
    assert EventKind.ENEMY_IN_TERRITORY not in _kinds(frames[3])  # still inside: no repeat


def test_territory_gain_and_loss_past_the_threshold() -> None:
    frames = _run(
        MinimapModel(territory=_home(0.05)),
        MinimapModel(territory=_home(0.051)),  # below 0.002
        MinimapModel(territory=_home(0.08)),
        MinimapModel(territory=_home(0.06)),
    )
    assert frames[1] == []
    assert frames[2][0].kind is EventKind.TERRITORY_GAINED
    assert frames[2][0].details == {"ownership": "self", "before": 0.05, "after": 0.08}
    assert frames[3][0].kind is EventKind.TERRITORY_LOST


def test_fog_changes_are_aggregated_per_kind() -> None:
    before = FogGrid(rows=2, cols=2, cells=((U, U), (E, V)))
    after = FogGrid(rows=2, cols=2, cells=((E, V), (V, E)))
    events = _run(MinimapModel(fog=before), MinimapModel(fog=after))[1]
    by_kind = {e.kind: e.details for e in events}
    assert by_kind[EventKind.AREA_EXPLORED]["cells"] == 2
    assert by_kind[EventKind.AREA_VISIBLE] == {
        "cells": 1,
        "row": 1.0,
        "col": 0.0,
        "explored_fraction": 1.0,
    }
    assert by_kind[EventKind.AREA_FOGGED]["cells"] == 1


def test_unread_fog_is_not_a_change() -> None:
    grid = FogGrid(rows=1, cols=1, cells=((U,),))
    seen = FogGrid(rows=1, cols=1, cells=((V,),))
    frames = _run(MinimapModel(fog=grid), MinimapModel(fog=None), MinimapModel(fog=seen))
    assert frames[1] == [] and _kinds(frames[2]) == [EventKind.AREA_EXPLORED]


def test_camera_moves_beyond_a_tile() -> None:
    frames = _run(
        MinimapModel(viewport=_view(100, 100)),
        MinimapModel(viewport=_view(102, 100)),  # 2 units: under the 4-unit minimum
        MinimapModel(viewport=_view(160, 180)),
    )
    assert frames[1] == []
    (moved,) = frames[2]
    assert moved.kind is EventKind.CAMERA_MOVED and moved.world_pos == WorldPoint(x=160, y=180)
    assert moved.details["dx"] == 60.0 and moved.details["dy"] == 80.0


def test_a_slow_pan_adds_up_to_a_camera_move() -> None:
    frames = _run(*(MinimapModel(viewport=_view(100 + 1.5 * i, 100)) for i in range(4)))
    assert [len(f) for f in frames] == [0, 0, 0, 1]  # 1.5 + 1.5 + 1.5 = 4.5 ≥ 4


def test_slow_territory_growth_adds_up() -> None:
    frames = _run(*(MinimapModel(territory=_home(0.05 + 0.001 * i)) for i in range(4)))
    assert [len(f) for f in frames] == [0, 0, 1, 0]  # +0.002 reached on the third frame
