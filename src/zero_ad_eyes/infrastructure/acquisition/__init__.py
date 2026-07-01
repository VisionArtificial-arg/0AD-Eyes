"""Frame acquisition adapters (EPIC A).

``InMemoryFrameSource`` is a trunk-provided minimal ``FrameSource`` used by tests
and the CLI smoke path. The live (screen-capture) and offline (video/image-folder)
sources are built by the acquisition feature agent.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from zero_ad_eyes.application.frames import Frame


class InMemoryFrameSource:
    """Yields a fixed sequence of frames (satisfies the ``FrameSource`` port)."""

    def __init__(self, frames: Sequence[Frame]) -> None:
        self._frames = tuple(frames)

    def frames(self) -> Iterator[Frame]:
        yield from self._frames
