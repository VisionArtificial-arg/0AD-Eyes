"""Tests for the rich debug overlay (REQUIREMENTS.md X1).

Every test renders onto a synthetic black frame and asserts on shape/dtype and on
*pixels changing* for each drawn element — no display is ever opened (headless).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zero_ad_eyes.domain.confidence import Confidence
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.geometry import ScreenBBox, WorldPoint
from zero_ad_eyes.domain.hud import HudState, Population
from zero_ad_eyes.domain.minimap import Blip, FogState, MinimapModel, ViewportQuad
from zero_ad_eyes.domain.taxonomy import EntityKind, Ownership, Phase, ResourceType
from zero_ad_eyes.domain.world_model import WorldModel
from zero_ad_eyes.interface.default_config import default_config
from zero_ad_eyes.interface.overlay import OverlayVideoSink, render

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
        viewport=ViewportQuad(
            top_left=WorldPoint(x=2.0, y=2.0),
            top_right=WorldPoint(x=8.0, y=2.0),
            bottom_right=WorldPoint(x=8.0, y=8.0),
            bottom_left=WorldPoint(x=2.0, y=8.0),
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


class _FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2
        return True

    def write(self, image: np.ndarray) -> None:
        self.frames.append(image.copy())

    def release(self) -> None:
        self.released = True


def test_overlay_video_sink_writes_annotated_frames(tmp_path: Path) -> None:
    writer = _FakeWriter()
    calls: list[tuple[str, int, float, tuple[int, int]]] = []

    def factory(path: str, fourcc: int, fps: float, size: tuple[int, int]) -> _FakeWriter:
        calls.append((path, fourcc, fps, size))
        return writer

    frame = make_frame(width=200, height=150)
    wm = WorldModel(
        meta=frame.meta,
        entities=(
            Entity(
                entity_id=1,
                kind=EntityKind.UNIT,
                ownership=Ownership.SELF,
                screen_bbox=ScreenBBox(x=20, y=20, width=30, height=30),
            ),
        ),
    )

    sink = OverlayVideoSink(
        settings=default_config().overlay,
        video_path=tmp_path / "overlay.avi",
        fps=12.0,
        fourcc="MJPG",
        writer_factory=factory,
    )
    sink.publish(frame, wm)
    sink.close()

    assert calls[0][0].endswith("overlay.avi")
    assert calls[0][2:] == (12.0, (200, 150))
    assert len(writer.frames) == 1
    assert _changed(writer.frames[0], frame.image)
    assert writer.released


# --------------------------------------------------------------------------- #
# EVT-7 — event log + markers                                                 #
# --------------------------------------------------------------------------- #


def _events_model(frame_id: int, *events: Any) -> Any:
    from zero_ad_eyes.domain.world_model import FrameMeta, WorldModel

    meta = FrameMeta(
        frame_id=frame_id, timestamp=float(frame_id), source="t", width=320, height=240
    )
    return WorldModel(meta=meta, events=tuple(events))


def _event(kind: str, origin: str, frame_id: int = 0, **extra: Any) -> Any:
    from zero_ad_eyes.domain.events import Event

    return Event.model_validate({"kind": kind, "origin": origin, "frame_id": frame_id, **extra})


def test_format_event_is_one_compact_line() -> None:
    from zero_ad_eyes.interface.overlay import format_event

    event = _event(
        "combat",
        "entity",
        entity_id=7,
        details={
            "entity_kind": "unit",
            "ownership": "enemy",
            "health_before": 1.0,
            "health_after": 0.6,
        },
    )
    assert format_event(event) == "combat #7 ownership=enemy health_before=1.00 health_after=0.60"
    long = _event("blip_appeared", "minimap", details={"ownership": "enemy" * 20})
    assert len(format_event(long)) == 72 and format_event(long).endswith("...")


def test_event_log_keeps_the_newest_and_expires_old_entries() -> None:
    from zero_ad_eyes.interface.overlay import EventLog

    log = EventLog(lines=3, hold_frames=5)
    first = log.add(
        _events_model(0, _event("phase_changed", "hud"), _event("recalibrated", "system"))
    )
    assert [line.text for line in first] == ["recalibrated", "phase_changed"]  # newest first
    crowded = log.add(_events_model(1, *(_event("camera_moved", "minimap") for _ in range(3))))
    assert len(crowded) == 3 and all(line.frame_id == 1 for line in crowded)
    assert log.add(_events_model(6)) == ()  # 5 frames later: all expired


def test_event_log_panel_and_markers_are_drawn() -> None:
    from zero_ad_eyes.domain.geometry import ScreenPoint
    from zero_ad_eyes.interface.overlay import EventLog

    frame = make_frame(width=320, height=240)
    event = _event("unit_appeared", "entity", screen_pos=ScreenPoint(x=160, y=200))
    model = _events_model(0, event)
    plain = render(frame, model.model_copy(update={"events": ()}))
    marked = render(frame, model)
    assert (marked != plain).any()  # the ring
    logged = render(frame, model, event_log=EventLog(lines=5, hold_frames=10).add(model))
    top_right = (slice(0, 20), slice(240, 320))
    assert (logged[top_right] != marked[top_right]).any()  # the panel
