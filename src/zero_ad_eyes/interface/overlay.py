"""Debug overlay skeleton (REQUIREMENTS.md X1).

Renders the world model back onto the frame — the primary way every deliverable is
visually verified. The trunk provides a minimal, dependency-light implementation
(entity boxes + ids); the tooling agent extends it (owner colours, health, fog,
minimap). Kept import-light so it never blocks headless test runs.
"""

from __future__ import annotations

from typing import Any

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.world_model import WorldModel


def render(frame: Frame, world_model: WorldModel) -> Any:
    """Return a copy of ``frame.image`` with the world model drawn on top.

    Uses OpenCV lazily so importing this module has no hard runtime cost.
    """

    import cv2  # local import: keep module import headless-safe

    canvas = frame.image.copy()
    for entity in world_model.entities:
        box = entity.screen_bbox
        if box is None:
            continue
        p1 = (int(box.x), int(box.y))
        p2 = (int(box.x + box.width), int(box.y + box.height))
        cv2.rectangle(canvas, p1, p2, (0, 255, 0), 1)
        cv2.putText(
            canvas,
            f"#{entity.entity_id}",
            (p1[0], max(0, p1[1] - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
    return canvas
