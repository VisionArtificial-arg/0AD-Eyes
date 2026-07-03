"""Rich debug overlay (REQUIREMENTS.md X1).

Renders the world model back onto the frame — the primary way every deliverable is
visually verified (§9 X1, DoD-B). Draws, from the ``WorldModel``:

- entity boxes tinted by *owner colour* (E3), with entity id + fine type;
- a *health bar* per entity coloured by fraction (E4);
- a *HUD summary* banner (resources, population, phase, self-id) (§4.2);
- *minimap blips* and the camera viewport projected into a corner panel (§4.4);
- an optional *fog-of-war* grid tinted per cell state (§4.4/§4.5).

Design notes:
- ``render`` keeps its original ``(frame, world_model)`` signature working; the
  extra inputs are keyword-only and optional, so existing callers are unaffected.
- OpenCV is imported lazily and the result is an ``ndarray`` — no window is ever
  opened — so the overlay stays HEADLESS-SAFE for tests and offline batch runs.
- Colours live in :class:`OverlaySettings` (X3), never as magic numbers here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import OverlaySettings
from zero_ad_eyes.domain.entities import Entity
from zero_ad_eyes.domain.minimap import Blip, FogState, MinimapModel
from zero_ad_eyes.domain.world_model import WorldModel
from zero_ad_eyes.interface.default_config import default_config

# A fog grid is a row-major 2-D grid of visibility states (§4.5). Kept as a plain
# nested sequence so the overlay does not force a numpy shape on its callers.
FogGrid = Sequence[Sequence[FogState]]


class OverlayVideoSink:
    """Writes and/or displays frames annotated with :func:`render`.

    This is the runtime side of the overlay annotator: ``render`` remains pure and
    headless-safe, while this class owns the OpenCV side effects used by the CLI.
    The video writer opens lazily on the first frame because the frame supplies the
    output dimensions.
    """

    def __init__(
        self,
        *,
        settings: OverlaySettings,
        video_path: str | Path | None = None,
        fps: float = 30.0,
        fourcc: str = "mp4v",
        show: bool = False,
        window_name: str = "0AD-Eyes overlay",
        writer_factory: Callable[[str, int, float, tuple[int, int]], Any] | None = None,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self._settings = settings
        self._video_path = Path(video_path) if video_path is not None else None
        self._fps = fps
        self._fourcc = fourcc
        self._show = show
        self._window_name = window_name
        self._writer_factory = writer_factory
        self._writer: Any | None = None

    @property
    def video_path(self) -> Path | None:
        return self._video_path

    def publish(self, frame: Frame, world_model: WorldModel) -> None:
        minimap = world_model.minimap
        fog = minimap.fog.cells if minimap is not None and minimap.fog is not None else None
        annotated = render(frame, world_model, settings=self._settings, fog=fog)
        self._write(annotated)
        self._display(annotated)

    def _write(self, annotated: Any) -> None:
        if self._video_path is None:
            return
        if self._writer is None:
            import cv2

            height, width = annotated.shape[:2]
            self._video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter.fourcc(*self._fourcc)
            factory = self._writer_factory
            if factory is None:
                factory = cv2.VideoWriter
            self._writer = factory(str(self._video_path), fourcc, self._fps, (width, height))
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise OSError(
                    f"cannot open overlay video writer for {self._video_path} "
                    f"(codec {self._fourcc!r} unavailable?)"
                )
        self._writer.write(annotated)

    def _display(self, annotated: Any) -> None:
        if not self._show:
            return
        import cv2

        cv2.imshow(self._window_name, annotated)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._show:
            import cv2

            cv2.destroyWindow(self._window_name)

    def __enter__(self) -> OverlayVideoSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def render(
    frame: Frame,
    world_model: WorldModel,
    *,
    settings: OverlaySettings | None = None,
    fog: FogGrid | None = None,
) -> Any:
    """Return a copy of ``frame.image`` with the world model drawn on top.

    ``settings`` defaults to the generated overlay defaults
    (:func:`interface.default_config.default_config` — the single source of default
    values; no disk/env I/O, so rendering stays pure and deterministic, NF5). ``fog``,
    when given, tints a grid over the minimap panel; it is a separate argument because
    per-cell fog is not (yet) carried inside :class:`WorldModel`.
    """

    import cv2  # local import: keep module import headless-safe

    cfg = settings if settings is not None else default_config().overlay
    canvas = frame.image.copy()

    for entity in world_model.entities:
        _draw_entity(canvas, entity, cfg, cv2)

    _draw_minimap_panel(canvas, world_model.minimap, fog, cfg, cv2)
    _draw_hud_summary(canvas, world_model, cfg, cv2)

    return canvas


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an RGB triple (config/domain convention) to OpenCV's BGR order."""

    r, g, b = rgb
    return (b, g, r)


def _draw_entity(canvas: Any, entity: Entity, cfg: OverlaySettings, cv2: Any) -> None:
    box = entity.screen_bbox
    if box is None:
        return

    owner_bgr = _bgr(cfg.owner_palette.for_ownership(entity.ownership))
    x1, y1 = int(box.x), int(box.y)
    x2, y2 = int(box.x + box.width), int(box.y + box.height)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), owner_bgr, cfg.box_thickness)

    label = f"#{entity.entity_id}"
    if entity.entity_type:
        label += f" {entity.entity_type}"
    if entity.health is not None:
        label += f" {int(round(entity.health * 100))}%"
    if entity.staleness > 0:
        label += f" ~{entity.staleness}"
    cv2.putText(
        canvas,
        label,
        (x1, max(0, y1 - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg.font_scale,
        owner_bgr,
        1,
        cv2.LINE_AA,
    )

    if entity.selected:
        cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 1)

    if entity.health is not None:
        _draw_health_bar(canvas, box, entity.health, cfg, cv2)


def _draw_health_bar(canvas: Any, box: Any, health: float, cfg: OverlaySettings, cv2: Any) -> None:
    fraction = max(0.0, min(1.0, health))
    x1, y1 = int(box.x), int(box.y)
    width = max(1, int(box.width))
    bar_y = max(0, y1 - 4)

    cv2.rectangle(canvas, (x1, bar_y), (x1 + width, bar_y + 2), (40, 40, 40), -1)
    filled = int(round(width * fraction))
    if filled > 0:
        colour = _bgr(cfg.health_color(fraction))
        cv2.rectangle(canvas, (x1, bar_y), (x1 + filled, bar_y + 2), colour, -1)


def _draw_hud_summary(canvas: Any, world_model: WorldModel, cfg: OverlaySettings, cv2: Any) -> None:
    lines = _hud_lines(world_model)
    if not lines:
        return

    scale = cfg.font_scale
    line_h = max(10, int(18 * scale / 0.4))
    pad = 4
    panel_h = pad * 2 + line_h * len(lines)
    panel_w = pad * 2 + int(max(len(line) for line in lines) * 6.5 * (scale / 0.4))
    panel_w = min(panel_w, canvas.shape[1])

    cv2.rectangle(canvas, (0, 0), (panel_w, panel_h), _bgr(cfg.hud_panel_color), -1)
    text_bgr = _bgr(cfg.hud_text_color)
    for i, line in enumerate(lines):
        y = pad + line_h * (i + 1) - 4
        cv2.putText(
            canvas,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_bgr,
            1,
            cv2.LINE_AA,
        )


def _hud_lines(world_model: WorldModel) -> list[str]:
    meta = world_model.meta
    lines = [f"frame {meta.frame_id} | {meta.source} | entities {len(world_model.entities)}"]

    hud = world_model.hud
    if hud is not None:
        if hud.stockpiles:
            stock = "  ".join(
                f"{res.value[:1].upper()}:{amt}" for res, amt in hud.stockpiles.items()
            )
            lines.append(stock)
        if hud.population is not None:
            lines.append(f"pop {hud.population.current}/{hud.population.cap}")
        detail = f"phase {hud.phase.value}"
        if hud.self_civ:
            detail += f" | civ {hud.self_civ}"
        lines.append(detail)
    return lines


def _draw_minimap_panel(
    canvas: Any,
    minimap: MinimapModel | None,
    fog: FogGrid | None,
    cfg: OverlaySettings,
    cv2: Any,
) -> None:
    if minimap is None and fog is None:
        return

    height, width = int(canvas.shape[0]), int(canvas.shape[1])
    side = max(24, int(min(height, width) * cfg.minimap_fraction))
    side = min(side, height, width)
    x0, y0 = 0, height - side
    x1, y1 = side, height

    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (30, 30, 30), -1)

    if fog is not None:
        _draw_fog(canvas, fog, (x0, y0, side, side), cfg, cv2)

    if minimap is not None:
        bounds = _world_bounds(minimap)
        if bounds is not None:
            _draw_blips(canvas, minimap.blips, bounds, (x0, y0, side, side), cfg, cv2)
            _draw_viewport(canvas, minimap, bounds, (x0, y0, side, side), cv2)

    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (200, 200, 200), 1)


def _draw_fog(
    canvas: Any,
    fog: FogGrid,
    panel: tuple[int, int, int, int],
    cfg: OverlaySettings,
    cv2: Any,
) -> None:
    rows = [row for row in fog if len(row) > 0]
    if not rows:
        return
    px, py, pw, ph = panel
    n_rows = len(rows)
    for r, row in enumerate(rows):
        n_cols = len(row)
        cy0 = py + int(ph * r / n_rows)
        cy1 = py + int(ph * (r + 1) / n_rows)
        for c, state in enumerate(row):
            cx0 = px + int(pw * c / n_cols)
            cx1 = px + int(pw * (c + 1) / n_cols)
            cv2.rectangle(
                canvas, (cx0, cy0), (cx1, cy1), _bgr(cfg.fog_palette.for_state(state)), -1
            )


def _world_bounds(minimap: MinimapModel) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for blip in minimap.blips:
        xs.append(blip.world_pos.x)
        ys.append(blip.world_pos.y)
    if minimap.viewport is not None:
        for corner in minimap.viewport.corners():
            xs.append(corner.x)
            ys.append(corner.y)
    if not xs or not ys:
        return None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Pad degenerate (single-point / axis-aligned) extents so projection is stable.
    if max_x - min_x < 1e-6:
        min_x, max_x = min_x - 1.0, max_x + 1.0
    if max_y - min_y < 1e-6:
        min_y, max_y = min_y - 1.0, max_y + 1.0
    return (min_x, min_y, max_x, max_y)


def _project(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    px, py, pw, ph = panel
    u = (x - min_x) / (max_x - min_x)
    v = (y - min_y) / (max_y - min_y)
    sx = px + int(round(u * (pw - 1)))
    sy = py + int(round(v * (ph - 1)))
    return (sx, sy)


def _draw_blips(
    canvas: Any,
    blips: tuple[Blip, ...],
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
    cfg: OverlaySettings,
    cv2: Any,
) -> None:
    for blip in blips:
        centre = _project(blip.world_pos.x, blip.world_pos.y, bounds, panel)
        cv2.circle(canvas, centre, 2, _bgr(cfg.owner_palette.for_ownership(blip.ownership)), -1)


def _draw_viewport(
    canvas: Any,
    minimap: MinimapModel,
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
    cv2: Any,
) -> None:
    viewport = minimap.viewport
    if viewport is None:
        return
    points = [_project(corner.x, corner.y, bounds, panel) for corner in viewport.corners()]
    for index in range(len(points)):
        cv2.line(canvas, points[index], points[(index + 1) % len(points)], (255, 255, 255), 1)
