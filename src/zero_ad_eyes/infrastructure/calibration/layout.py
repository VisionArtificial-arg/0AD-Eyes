"""HUD calibration adapter (EPIC B — B1 resolution/UI-scale, +B3 reuse).

``HudCalibrator`` implements the ``Calibrator`` port (application.ports). It turns a
``Frame`` into a ``Calibration`` by:

- **B1** reading the frame resolution from the pixel buffer and estimating the UI
  scale — from a detected top-bar anchor when the pixels give one, else a supplied
  default;
- delegating region geometry to :class:`HudLayoutRatios` (B2), passing anchor
  refinements where detection succeeded;
- **B3** optionally reusing/persisting a stored profile keyed by resolution+theme,
  so a resolution/theme already calibrated skips re-detection.

No absolute per-resolution pixel constants are used: every region is derived from
width/height/ui_scale (A4), anchor-refined where feasible, ratio-fallback otherwise.
"""

from __future__ import annotations

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import CalibrationSettings
from zero_ad_eyes.domain.calibration import Calibration

from .anchors import bottom_band_fraction, top_band_fraction
from .profiles import CalibrationProfileStore
from .ratios import HudLayoutRatios, clamp

# UI scale is a bounded correction, not an open-ended multiplier: 0 A.D.'s HUD scale
# lives roughly within these bounds, so an anchor-derived estimate is clamped here to
# reject spurious detections rather than propagate them into every region.
_UI_SCALE_MIN = 0.5
_UI_SCALE_MAX = 3.0

DEFAULT_THEME = "default"


class HudCalibrator:
    """A ``Calibrator`` (structural port impl) that discovers the HUD per session.

    Collaborators are injected: the layout ``ratios`` (what the HUD looks like), an
    optional profile ``store`` (B3 reuse/persistence), and a ``theme`` label. Anchor
    detection can be disabled for deterministic ratio-only behaviour.
    """

    def __init__(
        self,
        ratios: HudLayoutRatios | None = None,
        *,
        theme: str = DEFAULT_THEME,
        store: CalibrationProfileStore | None = None,
        use_anchors: bool = True,
        default_ui_scale: float = 1.0,
        ui_scale_min: float = _UI_SCALE_MIN,
        ui_scale_max: float = _UI_SCALE_MAX,
    ) -> None:
        self._ratios = ratios or HudLayoutRatios()
        self._theme = theme
        self._store = store
        self._use_anchors = use_anchors
        self._default_ui_scale = default_ui_scale
        self._ui_scale_min = ui_scale_min
        self._ui_scale_max = ui_scale_max

    @classmethod
    def from_settings(
        cls, settings: CalibrationSettings, *, store: CalibrationProfileStore | None = None
    ) -> HudCalibrator:
        """Build from pure config (Approach B). ``store`` (B3) is wired separately."""

        return cls(
            HudLayoutRatios.model_validate(settings.ratios.model_dump()),
            theme=settings.theme,
            store=store,
            use_anchors=settings.use_anchors,
            default_ui_scale=settings.default_ui_scale,
            ui_scale_min=settings.ui_scale_min,
            ui_scale_max=settings.ui_scale_max,
        )

    @property
    def theme(self) -> str:
        return self._theme

    def calibrate(self, frame: Frame) -> Calibration:
        width, height = self._resolution(frame)

        if self._store is not None:
            cached = self._store.load(width, height, self._theme)
            if cached is not None:
                return cached

        top_anchor = self._top_anchor(frame)
        bottom_anchor = self._bottom_anchor(frame)
        ui_scale = self._estimate_ui_scale(top_anchor)

        calibration = Calibration(
            width=width,
            height=height,
            ui_scale=ui_scale,
            top_bar=self._ratios.top_bar(width, height, ui_scale, top_anchor),
            minimap=self._ratios.minimap(width, height, ui_scale, bottom_anchor),
            selection_panel=self._ratios.selection_panel(width, height, ui_scale, bottom_anchor),
        )

        if self._store is not None:
            self._store.save(calibration, self._theme)

        return calibration

    def _resolution(self, frame: Frame) -> tuple[int, int]:
        """Prefer the actual pixel buffer shape; fall back to declared metadata."""

        image = frame.image
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
        return frame.meta.width, frame.meta.height

    def _top_anchor(self, frame: Frame) -> float | None:
        if not self._use_anchors or frame.image is None:
            return None
        return top_band_fraction(frame.image)

    def _bottom_anchor(self, frame: Frame) -> float | None:
        if not self._use_anchors or frame.image is None:
            return None
        return bottom_band_fraction(frame.image)

    def _estimate_ui_scale(self, top_anchor: float | None) -> float:
        if top_anchor is None:
            return self._default_ui_scale
        estimated = top_anchor / self._ratios.top_bar_height
        return clamp(estimated, self._ui_scale_min, self._ui_scale_max)
