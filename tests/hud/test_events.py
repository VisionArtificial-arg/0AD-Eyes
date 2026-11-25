"""EVT-3 — HUD events: every observable top-bar / selection-panel transition."""

from __future__ import annotations

from zero_ad_eyes.application.frames import PerceptionContext
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.events import DetailValue, EventKind, EventOrigin
from zero_ad_eyes.domain.hud import HudState, Population, SelectionState
from zero_ad_eyes.domain.taxonomy import Phase, ResourceType
from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel
from zero_ad_eyes.infrastructure.hud.events import HudEventSource
from zero_ad_eyes.interface.default_config import default_config

FOOD, WOOD = ResourceType.FOOD, ResourceType.WOOD


def _model(frame_id: int, hud: HudState | None) -> WorldModel:
    meta = FrameMeta(frame_id=frame_id, timestamp=float(frame_id), source="t", width=8, height=8)
    return WorldModel(meta=meta, hud=hud)


def _hud(**fields: object) -> HudState:
    base: dict[str, object] = {
        "stockpiles": {FOOD: 300, WOOD: 200},
        "population": Population(current=5, cap=20),
        "phase": Phase.VILLAGE,
        "confidence": Confidence(value=0.9, provenance=Provenance.CLASSICAL),
    }
    return HudState.model_validate({**base, **fields})


def _run(*huds: HudState | None) -> list[list[tuple[EventKind, dict[str, DetailValue]]]]:
    source = HudEventSource.from_settings(default_config().events)
    return [
        [(e.kind, e.details) for e in source.events(_model(i, hud), PerceptionContext())]
        for i, hud in enumerate(huds)
    ]


def test_first_frame_is_a_baseline_not_a_change() -> None:
    assert _run(_hud()) == [[]]


def test_resource_gain_and_spend_carry_the_amount() -> None:
    _, second = _run(_hud(), _hud(stockpiles={FOOD: 350, WOOD: 150}))
    assert second == [
        (
            EventKind.RESOURCE_GAINED,
            {"resource": "food", "before": 300, "after": 350, "amount": 50},
        ),
        (EventKind.RESOURCE_SPENT, {"resource": "wood", "before": 200, "after": 150, "amount": 50}),
    ]


def test_an_unread_resource_is_not_a_change_and_last_value_is_kept() -> None:
    frames = _run(_hud(), _hud(stockpiles={FOOD: 300}), _hud(stockpiles={FOOD: 300, WOOD: 210}))
    assert frames[1] == []
    assert frames[2] == [
        (EventKind.RESOURCE_GAINED, {"resource": "wood", "before": 200, "after": 210, "amount": 10})
    ]


def test_a_missing_hud_frame_does_not_reset_the_baseline() -> None:
    frames = _run(_hud(), None, _hud(stockpiles={FOOD: 301, WOOD: 200}))
    assert frames[1] == []
    assert [kind for kind, _ in frames[2]] == [EventKind.RESOURCE_GAINED]


def test_population_change_cap_raise_and_cap_reached() -> None:
    frames = _run(
        _hud(population=Population(current=19, cap=20)),
        _hud(population=Population(current=20, cap=20)),
        _hud(population=Population(current=20, cap=30)),
        _hud(population=Population(current=20, cap=25)),
    )
    assert frames[1] == [
        (EventKind.POPULATION_CHANGED, {"before": 19, "after": 20, "delta": 1, "cap": 20}),
        (EventKind.POPULATION_CAP_REACHED, {"current": 20, "cap": 20}),
    ]
    assert frames[2] == [(EventKind.POPULATION_CAP_RAISED, {"before": 20, "after": 30})]
    assert frames[3] == [(EventKind.POPULATION_CAP_LOWERED, {"before": 30, "after": 25})]


def test_phase_change_but_not_to_or_from_unknown() -> None:
    frames = _run(_hud(), _hud(phase=Phase.UNKNOWN), _hud(phase=Phase.TOWN))
    assert frames[1] == []
    assert frames[2] == [(EventKind.PHASE_CHANGED, {"before": "village", "after": "town"})]


def test_selection_change_then_panel_details() -> None:
    house = SelectionState(entity_type="house", health=1.0, production_queue=(), garrison=0)
    frames = _run(
        _hud(),
        _hud(selection=house),
        _hud(selection=house.model_copy(update={"health": 0.8, "production_queue": ("female",)})),
        _hud(
            selection=house.model_copy(
                update={"health": 0.8, "production_queue": ("female",), "garrison": 2}
            )
        ),
        _hud(),
    )
    assert frames[1] == [(EventKind.SELECTION_CHANGED, {"before": None, "after": "house"})]
    assert frames[2] == [
        (EventKind.SELECTION_HEALTH_CHANGED, {"entity_type": "house", "before": 1.0, "after": 0.8}),
        (
            EventKind.PRODUCTION_QUEUE_CHANGED,
            {
                "entity_type": "house",
                "before": "",
                "after": "female",
                "length_before": 0,
                "length_after": 1,
            },
        ),
    ]
    assert frames[3] == [
        (EventKind.GARRISON_CHANGED, {"entity_type": "house", "before": 0, "after": 2})
    ]
    assert frames[4] == [(EventKind.SELECTION_CHANGED, {"before": "house", "after": None})]


def test_events_carry_origin_and_the_hud_read_confidence() -> None:
    source = HudEventSource.from_settings(default_config().events)
    source.events(_model(0, _hud()), PerceptionContext())
    (event,) = source.events(_model(1, _hud(phase=Phase.CITY)), PerceptionContext())
    assert event.origin is EventOrigin.HUD and event.frame_id == 1
    assert event.confidence == Confidence(value=0.9, provenance=Provenance.CLASSICAL)


def test_min_resource_delta_filters_small_changes() -> None:
    source = HudEventSource(min_resource_delta=10, min_selection_health_delta=0.01)
    source.events(_model(0, _hud()), PerceptionContext())
    small = _hud(stockpiles={FOOD: 305, WOOD: 200})
    assert source.events(_model(1, small), PerceptionContext()) == ()
