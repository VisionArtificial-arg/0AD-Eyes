"""The ``Frame`` carrier — pixels plus metadata.

Frames live at the application/infrastructure boundary, not in the domain: they
hold a raw image buffer (a numpy ``ndarray``), which we deliberately keep out of
the framework-free domain core. Everything downstream of preprocessing consumes
``Frame``; the emitted ``WorldModel`` never contains pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zero_ad_eyes.domain.world_model import FrameMeta


@dataclass(frozen=True)
class Frame:
    """A captured (or preprocessed) image with its provenance.

    ``image`` is an HxWxC ``numpy.ndarray`` (BGR, OpenCV convention). It is typed
    as ``Any`` to avoid leaking a hard numpy dependency into type-only positions
    and to keep the domain import-light.
    """

    image: Any
    meta: FrameMeta
