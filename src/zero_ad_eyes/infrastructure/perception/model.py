"""Classical ``PerceptionModel`` adapter — the E11/E6a baseline behind the seam.

A real, non-stub alternative to the learned model (MP3 "classical mode",
REQUIREMENTS.md §5.10.2): it satisfies the ``PerceptionModel`` port structurally
(``infer(frame, roi) -> Detections``) using only deterministic CV — template
matching on fixed art (E11) and colour/contour resource cues (E6a). It plugs
into the same pipeline as the learned adapter with zero downstream change.

Scope note (the 🔌 boundary): this adapter does **not** attempt general learned
entity detection/classification (E1/E2) or segmentation (E8/E9). It sees only
what fixed-art templates and colour cues can find, and every detection it emits
carries ``Provenance.CLASSICAL`` — never ``LEARNED``.
"""

from __future__ import annotations

from collections.abc import Sequence

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.detections import Detection, Detections
from zero_ad_eyes.domain.geometry import ScreenBBox

from .resources import DEFAULT_RESOURCE_CUES, ResourceCue, detect_resource_nodes
from .templates import Template, TemplateBank


class ClassicalPerceptionModel:
    """Deterministic ``PerceptionModel``: template matches + classical resource cues."""

    def __init__(
        self,
        template_bank: TemplateBank | None = None,
        resource_cues: Sequence[ResourceCue] = DEFAULT_RESOURCE_CUES,
        resource_templates: Sequence[Template] = (),
        detect_resources: bool = True,
    ) -> None:
        self._template_bank = template_bank if template_bank is not None else TemplateBank()
        self._resource_cues = tuple(resource_cues)
        self._resource_templates = tuple(resource_templates)
        self._detect_resources = detect_resources

    def infer(self, frame: Frame, roi: ScreenBBox | None = None) -> Detections:
        """Run the classical detectors over ``frame`` and package their detections."""

        items: list[Detection] = list(self._template_bank.detect(frame.image, roi=roi))
        if self._detect_resources:
            items.extend(
                detect_resource_nodes(
                    frame,
                    cues=self._resource_cues,
                    templates=self._resource_templates,
                    roi=roi,
                )
            )
        return Detections(frame_id=frame.meta.frame_id, items=tuple(items))
