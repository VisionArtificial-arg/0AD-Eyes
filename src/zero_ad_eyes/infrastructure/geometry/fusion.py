"""Reconcile two independent world-position estimates (F3; CV-26).

The main viewport and the minimap are *independent* position sources for the same
world (REQUIREMENTS.md §4.4, risk R2). ``reconcile`` fuses one estimate from each —
each a ``WorldPoint`` with its ``Confidence`` — into a single confidence-weighted
estimate tagged ``Provenance.FUSED`` (the pattern EPIC G/G5 generalises).

This helper deliberately operates on plain domain types only: it never imports the
minimap feature. Either estimate may come from anywhere that produces a
``WorldPoint`` + ``Confidence``.

The confidence model is an explicit heuristic (not a calibrated probability):
- position is the confidence-weighted mean of the two estimates;
- the fused sureness combines the two confidences by *noisy-OR*
  (``va + vb - va·vb`` — independent evidence reinforces), then is discounted by an
  *agreement factor* ``exp(-‖a − b‖ / agreement_scale)`` so that estimates which
  disagree in world space lower, rather than inflate, the fused confidence.
"""

from __future__ import annotations

import math

from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.geometry import WorldPoint


def reconcile(
    main_view: WorldPoint,
    main_confidence: Confidence,
    minimap: WorldPoint,
    minimap_confidence: Confidence,
    *,
    agreement_scale: float,
) -> tuple[WorldPoint, Confidence]:
    """Fuse a main-view and a minimap world estimate into one (F3).

    ``agreement_scale`` is the world-unit distance at which a disagreement between
    the two estimates costs a 1/e confidence discount. Returns the fused
    ``WorldPoint`` and its ``Confidence`` (provenance ``FUSED``).
    """

    if agreement_scale <= 0.0:
        raise ValueError("agreement_scale must be positive")

    wa = main_confidence.value
    wb = minimap_confidence.value
    total = wa + wb

    if total == 0.0:
        # Neither source is trusted at all: fall back to the midpoint, no sureness.
        fused_point = WorldPoint(
            x=(main_view.x + minimap.x) / 2.0,
            y=(main_view.y + minimap.y) / 2.0,
        )
        return fused_point, Confidence(value=0.0, provenance=Provenance.FUSED)

    fused_point = WorldPoint(
        x=(wa * main_view.x + wb * minimap.x) / total,
        y=(wa * main_view.y + wb * minimap.y) / total,
    )

    distance = math.hypot(main_view.x - minimap.x, main_view.y - minimap.y)
    agreement = math.exp(-distance / agreement_scale)
    combined = wa + wb - wa * wb  # noisy-OR of the two sureties, in [0, 1]
    fused_value = max(0.0, min(1.0, combined * agreement))

    return fused_point, Confidence(value=fused_value, provenance=Provenance.FUSED)
