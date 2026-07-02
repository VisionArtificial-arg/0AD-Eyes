"""D6 — Minimap-pixel ⇄ world-coordinate calibration (REQUIREMENTS.md §4.4/§4.6, CV-22/CV-26).

The minimap is a top-down, axis-aligned, linear projection of the playable world
onto a rectangle of pixels. This module owns the *only* place that mapping lives, so
downstream consumers reason in world space, never in minimap pixels.

Assumed world extent is **configurable** (:class:`WorldExtent`), not baked in: a
0 A.D. map is square with an engine-defined size, and different maps differ, so the
extent must be supplied per session (analogous to calibration, EPIC B) — there is no
in-code default; the composition root provides it from the ``minimap`` config.

Coordinate conventions (documented, because they are load-bearing):

- Minimap pixels use OpenCV's convention: origin top-left, ``px`` right, ``py`` down,
  relative to the *segmented region* (not the full frame).
- World ``x`` grows east (with ``px``). World ``y`` grows **north by default**
  (``flip_y=True``), i.e. up on the minimap is +y — matching how a player reads a
  map. Flip it off for engines whose ground plane grows southward.
"""

from __future__ import annotations

from dataclasses import dataclass

from zero_ad_eyes.domain.geometry import WorldPoint


@dataclass(frozen=True)
class WorldExtent:
    """The world rectangle the minimap covers, in engine units (configurable).

    ``origin`` is the world coordinate that maps to the minimap's top-left corner
    when ``flip_y`` is off; with ``flip_y`` on it is the world coordinate of the
    bottom-left corner (because +y points up). Width/height are the world span.
    """

    origin_x: float
    origin_y: float
    width: float
    height: float
    flip_y: bool

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("world extent width/height must be positive")

    @classmethod
    def square(cls, size: float, *, flip_y: bool = True) -> WorldExtent:
        """A square world of ``size`` engine units (the common 0 A.D. case)."""

        return cls(origin_x=0.0, origin_y=0.0, width=size, height=size, flip_y=flip_y)


@dataclass(frozen=True)
class MinimapProjector:
    """Bidirectional linear map between minimap pixels and world coordinates.

    Constructed for a concrete region pixel size; the reader builds one per frame
    from the segmented region's dimensions and the session :class:`WorldExtent`.
    """

    region_width: int
    region_height: int
    extent: WorldExtent

    def __post_init__(self) -> None:
        if self.region_width <= 0 or self.region_height <= 0:
            raise ValueError("region size must be positive")

    def to_world(self, px: float, py: float) -> WorldPoint:
        """Map a region-local minimap pixel to a world point."""

        u = px / self.region_width
        v = py / self.region_height
        wx = self.extent.origin_x + u * self.extent.width
        if self.extent.flip_y:
            wy = self.extent.origin_y + (1.0 - v) * self.extent.height
        else:
            wy = self.extent.origin_y + v * self.extent.height
        return WorldPoint(x=wx, y=wy)

    def to_pixel(self, point: WorldPoint) -> tuple[float, float]:
        """Inverse of :meth:`to_world`: world point back to a region-local pixel."""

        u = (point.x - self.extent.origin_x) / self.extent.width
        if self.extent.flip_y:
            v = 1.0 - (point.y - self.extent.origin_y) / self.extent.height
        else:
            v = (point.y - self.extent.origin_y) / self.extent.height
        return (u * self.region_width, v * self.region_height)
