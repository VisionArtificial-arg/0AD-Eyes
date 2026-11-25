"""EVT-5 — entity state events: sight, health recovery, motion, selection, spawns."""

from __future__ import annotations

from typing import Any

from zero_ad_eyes.application.frames import PerceptionContext
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.events import Event, EventKind, EventOrigin
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.minimap import MinimapModel, ViewportQuad
from zero_ad_eyes.domain.motion import Motion
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel
from zero_ad_eyes.infrastructure.tracking.entity_events import EntityStateEventSource
from zero_ad_eyes.interface.default_config import default_config

VIEW = ViewportQuad(
    top_left=WorldPoint(x=0, y=0),
    top_right=WorldPoint(x=100, y=0),
    bottom_right=WorldPoint(x=100, y=100),
    bottom_left=WorldPoint(x=0, y=100),
)


def _e(entity_id: int = 1, **fields: Any) -> Entity:
    base: dict[str, Any] = {
        "entity_id": entity_id,
        "kind": EntityKind.UNIT,
        "ownership": Ownership.SELF,
        "screen_bbox": ScreenBBox(x=10, y=10, width=4, height=8),
        "world_pos": WorldPoint(x=50, y=50),
        "confidence": Confidence(value=0.8, provenance=Provenance.LEARNED),
    }
    return Entity.model_validate({**base, **fields})


def _run(*frames: tuple[Entity, ...], viewport: ViewportQuad | None = VIEW) -> list[list[Event]]:
    source = EntityStateEventSource.from_settings(default_config().events)
    out = []
    for i, entities in enumerate(frames):
        meta = FrameMeta(frame_id=i, timestamp=float(i), source="t", width=200, height=200)
        model = WorldModel(meta=meta, entities=entities, minimap=MinimapModel(viewport=viewport))
        out.append(list(source.events(model, PerceptionContext())))
    return out


def _kinds(events: list[Event]) -> list[EventKind]:
    return [e.kind for e in events]


def test_leaving_the_camera_footprint_vs_vanishing_inside_it() -> None:
    gone_out = _run(
        (_e(),), (_e(staleness=1, world_pos=WorldPoint(x=300, y=50)),), (_e(staleness=0),)
    )
    assert _kinds(gone_out[1]) == [EventKind.ENTITY_LEFT_VIEW]
    assert _kinds(gone_out[2]) == [EventKind.ENTITY_ENTERED_VIEW]
    assert gone_out[2][0].details["frames_unseen"] == 1

    hidden = _run((_e(),), (_e(staleness=1),), (_e(staleness=2),), (_e(),))
    assert _kinds(hidden[1]) == [EventKind.ENTITY_HIDDEN]
    assert hidden[2] == []  # still unseen: no repeat
    assert _kinds(hidden[3]) == [EventKind.ENTITY_REVEALED]
    assert hidden[3][0].details["frames_unseen"] == 2


def test_without_a_viewport_a_lost_entity_is_hidden() -> None:
    frames = _run((_e(),), (_e(staleness=1, world_pos=WorldPoint(x=999, y=999)),), viewport=None)
    assert _kinds(frames[1]) == [EventKind.ENTITY_HIDDEN]
    assert frames[1][0].details["viewport_known"] is False


def test_health_recovery_but_not_jitter() -> None:
    frames = _run((_e(health=0.5),), (_e(health=0.51),), (_e(health=0.6),))
    assert frames[1] == []
    assert _kinds(frames[2]) == [EventKind.HEALTH_RECOVERED]
    assert frames[2][0].details["health_before"] == 0.51


def test_motion_start_and_stop_with_hysteresis() -> None:
    def moving(speed: float) -> Entity:
        return _e(motion=Motion(dx=speed, dy=0.0))

    frames = _run(
        *((e,) for e in (moving(0.0), moving(2.0), moving(0.6), moving(0.2), moving(0.6)))
    )
    assert [_kinds(f) for f in frames] == [
        [],
        [EventKind.STARTED_MOVING],
        [],  # 0.6 is between stop (0.3) and move (1.0): no flicker
        [EventKind.STOPPED_MOVING],
        [],
    ]


def test_selection_changes() -> None:
    frames = _run((_e(),), (_e(selected=True),), (_e(selected=False),))
    assert _kinds(frames[1]) == [EventKind.ENTITY_SELECTED]
    assert _kinds(frames[2]) == [EventKind.ENTITY_DESELECTED]


def test_new_buildings_are_seen_and_units_next_to_own_buildings_are_possible_trainees() -> None:
    house = _e(
        9,
        kind=EntityKind.BUILDING,
        entity_type="house",
        screen_bbox=ScreenBBox(x=100, y=100, width=40, height=40),
    )
    near = _e(2, screen_bbox=ScreenBBox(x=150, y=110, width=4, height=8))  # 12 px right of it
    far = _e(3, screen_bbox=ScreenBBox(x=10, y=10, width=4, height=8))
    enemy = _e(
        4, ownership=Ownership.ENEMY, screen_bbox=ScreenBBox(x=150, y=120, width=4, height=8)
    )
    frames = _run((house,), (house, near, far, enemy))
    assert _kinds(frames[0]) == [EventKind.BUILDING_SEEN]
    (trained,) = frames[1]
    assert trained.kind is EventKind.POSSIBLE_UNIT_TRAINED and trained.entity_id == 2
    assert (
        trained.details["building_entity_id"] == 9 and trained.details["building_type"] == "house"
    )
    assert trained.details["distance_px"] == 12.0


def test_no_trainee_guesses_on_the_first_frame() -> None:
    house = _e(
        9, kind=EntityKind.BUILDING, screen_bbox=ScreenBBox(x=100, y=100, width=40, height=40)
    )
    near = _e(2, screen_bbox=ScreenBBox(x=110, y=110, width=4, height=8))
    assert _kinds(_run((house, near))[0]) == [EventKind.BUILDING_SEEN]


def test_events_carry_entity_context() -> None:
    (event,) = _run((_e(),), (_e(selected=True),))[1]
    assert event.origin is EventOrigin.ENTITY and event.entity_id == 1
    assert event.details["ownership"] == "self" and event.details["entity_kind"] == "unit"
    assert event.world_pos == WorldPoint(x=50, y=50) and event.confidence.value == 0.8
