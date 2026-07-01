"""Shared synthetic frames for the preprocessing tests (EPIC P).

No display, recording, or model — a deterministic in-memory BGR frame with real
structure (a bright rectangle) so edge/contrast/normalization steps have something
to act on.
"""

from __future__ import annotations

import numpy as np

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.world_model import FrameMeta


def make_pattern_frame(frame_id: int = 7, width: int = 32, height: int = 24) -> Frame:
    """A deterministic, non-flat BGR frame with structure for edges/contrast."""

    rng = np.random.default_rng(1234)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    image[4:12, 6:20] = 240
    return Frame(
        image=image,
        meta=FrameMeta(
            frame_id=frame_id,
            timestamp=float(frame_id),
            source="test",
            width=width,
            height=height,
        ),
    )
