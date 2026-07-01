"""Shared test fixtures and the golden-fixture harness scaffold (T1/T2).

Provides tiny synthetic frames so tests never need a display, a recording, or the
model. Feature agents drop real golden inputs under ``tests/fixtures/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.domain.world_model import FrameMeta


def make_frame(frame_id: int = 0, width: int = 64, height: int = 48) -> Frame:
    """A black synthetic frame with valid metadata."""

    return Frame(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        meta=FrameMeta(
            frame_id=frame_id,
            timestamp=float(frame_id),
            source="test",
            width=width,
            height=height,
        ),
    )


@pytest.fixture
def frame() -> Frame:
    return make_frame()
