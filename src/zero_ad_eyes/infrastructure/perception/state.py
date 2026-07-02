"""E5 — State cues: selection ring, construction scaffold, garrison badge `[S]`.

These are *best-effort* classical readings (hence the `[S]` speculative marker in
REQUIREMENTS.md): each is a single, honest heuristic over the entity's crop, not
a learned classifier. They return booleans a caller folds into an ``Entity``:

- selection — a bright outline hugging the entity's border (the selection ring);
- construction — a busy, wireframe-like interior (a build scaffold has far more
  internal edges than a finished, solid building);
- garrison — a small white/grey badge (the garrison count) near the top, distinct
  from the coloured entity body and from the full-width health bar.

Everything here is ``Provenance.CLASSICAL`` and deliberately coarse; treat a
``True`` as "likely", not "certain".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import StateCueSettings
from zero_ad_eyes.domain.geometry import ScreenBBox

from .masks import connected_components


@dataclass(frozen=True)
class StateCues:
    """The three best-effort state readings for one entity."""

    selected: bool = False
    under_construction: bool = False
    garrisoned: bool = False


def _clamp_box(image: np.ndarray, bbox: ScreenBBox) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x0 = max(0, int(bbox.x))
    y0 = max(0, int(bbox.y))
    x1 = min(w, int(bbox.x + bbox.width))
    y1 = min(h, int(bbox.y + bbox.height))
    return x0, y0, x1, y1


def detect_selection(
    frame: Frame,
    bbox: ScreenBBox,
    thickness: int = 3,
    brightness: int = 200,
    min_fraction: float = 0.4,
) -> bool:
    """True if a bright ring straddles the entity's border (a selection ring)."""

    image = frame.image
    h, w = image.shape[:2]
    t = max(1, thickness)
    ox0 = max(0, int(bbox.x) - t)
    oy0 = max(0, int(bbox.y) - t)
    ox1 = min(w, int(bbox.x + bbox.width) + t)
    oy1 = min(h, int(bbox.y + bbox.height) + t)
    outer = image[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return False

    band = 2 * t
    ring = np.zeros(outer.shape[:2], dtype=bool)
    ring[:band, :] = True
    ring[-band:, :] = True
    ring[:, :band] = True
    ring[:, -band:] = True

    value = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)[:, :, 2]
    ring_pixels = value[ring]
    if ring_pixels.size == 0:
        return False
    bright_fraction = float(np.count_nonzero(ring_pixels >= brightness)) / ring_pixels.size
    return bright_fraction >= min_fraction


def detect_construction(
    frame: Frame,
    bbox: ScreenBBox,
    edge_density_min: float = 0.12,
    canny_lo: float = 60.0,
    canny_hi: float = 180.0,
) -> bool:
    """True if the interior is wireframe-busy — a construction scaffold cue."""

    x0, y0, x1, y1 = _clamp_box(frame.image, bbox)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return False
    interior = frame.image[y0:y1, x0:x1]
    gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    density = float(np.count_nonzero(edges)) / float(edges.size)
    return density >= edge_density_min


def detect_garrison(
    frame: Frame,
    bbox: ScreenBBox,
    top_fraction: float = 0.35,
    brightness: int = 200,
    max_saturation: int = 70,
    min_badge_area: int = 6,
    max_badge_width_fraction: float = 0.5,
) -> bool:
    """True if a small white/grey badge sits near the top (a garrison count)."""

    x0, y0, x1, y1 = _clamp_box(frame.image, bbox)
    strip_h = int((y1 - y0) * top_fraction)
    if x1 - x0 < 2 or strip_h < 2:
        return False
    strip = frame.image[y0 : y0 + strip_h, x0:x1]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    lo = np.array([0, 0, brightness], dtype=np.uint8)
    hi = np.array([179, max_saturation, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    width_limit = max_badge_width_fraction * (x1 - x0)
    for comp in connected_components(mask, min_area=min_badge_area):
        if comp.bbox.width <= width_limit:
            return True
    return False


def read_state_cues(
    frame: Frame, bbox: ScreenBBox, settings: StateCueSettings | None = None
) -> StateCues:
    """Run all three best-effort cues for one entity and bundle the result.

    Knobs come from ``settings`` (config-driven, NF7); default reproduces the former
    hard-coded thresholds.
    """

    cfg = settings or StateCueSettings()
    sel, con, gar = cfg.selection, cfg.construction, cfg.garrison
    return StateCues(
        selected=detect_selection(frame, bbox, sel.thickness, sel.brightness, sel.min_fraction),
        under_construction=detect_construction(
            frame, bbox, con.edge_density_min, con.canny_lo, con.canny_hi
        ),
        garrisoned=detect_garrison(
            frame,
            bbox,
            gar.top_fraction,
            gar.brightness,
            gar.max_saturation,
            gar.min_badge_area,
            gar.max_badge_width_fraction,
        ),
    )
