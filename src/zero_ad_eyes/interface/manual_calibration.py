"""Manual HUD calibration helper.

Automatic HUD anchoring is intentionally best-effort; this module provides the
operator-controlled fallback for real capture setups. It asks the user to mark the
three broad regions the pipeline consumes and saves them through the existing
``CalibrationProfileStore`` so normal runs can reuse the profile.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import FrameSource
from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.geometry import ScreenBBox
from zero_ad_eyes.infrastructure.calibration import CalibrationProfileStore

BoxSelector = Callable[[str, Any], tuple[int, int, int, int]]
FrameFreezer = Callable[[str, Frame, Callable[[], Frame]], Frame]

_REQUIRED_TOP_FIELDS = ("food", "wood", "stone", "metal", "population")
_OPTIONAL_TOP_FIELDS = ("phase", "swatch", "civ")
_BAR_HEIGHT = 56


def _with_instruction_bar(image: Any, text: str) -> Any:
    """Return a preview image with instructions below, outside the calibrated frame."""

    import cv2

    image_height = image.shape[0]
    clone = cv2.copyMakeBorder(
        image,
        0,
        _BAR_HEIGHT,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(24, 24, 24),
    )
    cv2.putText(
        clone,
        text,
        (12, image_height + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return clone


def _default_selector(label: str, image: Any) -> tuple[int, int, int, int]:
    """Ask the user to draw one ROI using OpenCV's built-in selector."""

    import cv2

    window = f"0AD-Eyes manual calibration: {label}"
    image_height, image_width = image.shape[:2]
    clone = _with_instruction_bar(
        image,
        f"Draw {label}, Enter/Space accepts, C skips optional boxes",
    )
    try:
        raw = cv2.selectROI(window, clone, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(window)
    x, y, width, height = raw
    x0 = max(0, min(int(x), image_width))
    y0 = max(0, min(int(y), image_height))
    x1 = max(0, min(int(x + width), image_width))
    y1 = max(0, min(int(y + height), image_height))
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _default_freezer(label: str, first_frame: Frame, next_frame: Callable[[], Frame]) -> Frame:
    """Preview a frame stream until the operator freezes the frame for this label."""

    import cv2

    window = f"0AD-Eyes manual calibration preview: {label}"
    frame = first_frame
    exhausted = False
    try:
        while True:
            preview = _with_instruction_bar(
                frame.image,
                f"{label}: Space/Enter freezes this frame, Esc/Q uses current frame",
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 32, 27, ord("q"), ord("Q")):
                return frame
            if not exhausted:
                try:
                    frame = next_frame()
                except StopIteration:
                    exhausted = True
    finally:
        cv2.destroyWindow(window)


def _bbox(label: str, raw: tuple[int, int, int, int]) -> ScreenBBox:
    x, y, width, height = raw
    if width <= 0 or height <= 0:
        raise ValueError(f"manual calibration cancelled or empty box for {label}")
    return ScreenBBox(x=float(x), y=float(y), width=float(width), height=float(height))


def _optional_bbox(label: str, raw: tuple[int, int, int, int]) -> ScreenBBox | None:
    x, y, width, height = raw
    if width <= 0 or height <= 0:
        return None
    return ScreenBBox(x=float(x), y=float(y), width=float(width), height=float(height))


def collect_manual_calibration(
    frame: Frame,
    *,
    selector: BoxSelector | None = None,
) -> Calibration:
    """Collect broad HUD boxes from ``frame`` and return a calibration profile."""

    choose = selector if selector is not None else _default_selector
    hud_regions = {
        field: _bbox(field, choose(field, frame.image)) for field in _REQUIRED_TOP_FIELDS
    }
    for field in _OPTIONAL_TOP_FIELDS:
        box = _optional_bbox(field, choose(f"{field} (optional)", frame.image))
        if box is not None:
            hud_regions[field] = box
    top_bar = _union(tuple(hud_regions.values()))
    selection_panel = _optional_bbox(
        "selection panel",
        choose("selection panel (optional)", frame.image),
    )
    return Calibration(
        width=frame.meta.width,
        height=frame.meta.height,
        ui_scale=1.0,
        top_bar=top_bar,
        minimap=_bbox("minimap", choose("minimap", frame.image)),
        selection_panel=selection_panel,
        hud_regions=hud_regions,
    )


def collect_manual_calibration_from_source(
    source: FrameSource,
    *,
    selector: BoxSelector | None = None,
    freezer: FrameFreezer | None = None,
) -> Calibration:
    """Collect manual boxes while previewing a live/recorded frame stream.

    Each requested label first shows continuous frames. The operator freezes the
    relevant moment, then the normal ROI selector runs on that frozen image.
    """

    frames = source.frames()
    first_frame = next(frames)
    current = first_frame

    def next_frame() -> Frame:
        nonlocal current
        current = next(frames)
        return current

    freeze = freezer if freezer is not None else _default_freezer
    choose_box = selector if selector is not None else _default_selector

    def choose(label: str, _image: Any) -> tuple[int, int, int, int]:
        frozen = freeze(label, current, next_frame)
        return choose_box(label, frozen.image)

    return collect_manual_calibration(first_frame, selector=choose)


def _union(boxes: tuple[ScreenBBox, ...]) -> ScreenBBox:
    """Smallest box containing all selected boxes."""

    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x + box.width for box in boxes)
    y1 = max(box.y + box.height for box in boxes)
    return ScreenBBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def save_manual_calibration(
    frame: Frame,
    *,
    directory: str | Path,
    theme: str,
    selector: BoxSelector | None = None,
) -> Path:
    """Collect and persist a manual profile, returning the written JSON path."""

    calibration = collect_manual_calibration(frame, selector=selector)
    store = CalibrationProfileStore(directory)
    store.save(calibration, theme)
    return store.directory / f"{store.key(calibration.width, calibration.height, theme)}.json"


def save_manual_calibration_from_source(
    source: FrameSource,
    *,
    directory: str | Path,
    theme: str,
    selector: BoxSelector | None = None,
    freezer: FrameFreezer | None = None,
) -> Path:
    """Collect and persist a streamed manual profile, returning the written path."""

    calibration = collect_manual_calibration_from_source(
        source,
        selector=selector,
        freezer=freezer,
    )
    store = CalibrationProfileStore(directory)
    store.save(calibration, theme)
    return store.directory / f"{store.key(calibration.width, calibration.height, theme)}.json"
