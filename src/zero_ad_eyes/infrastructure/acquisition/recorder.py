"""Recording / dump mode (EPIC A / A5).

``FrameRecorder`` is a passthrough decorator around *any* ``FrameSource``: as each
frame flows through it is persisted to disk as an image sequence (for dataset
building, per ML1) while the frame itself is forwarded unchanged. Because it is
both a ``FrameSource`` and a wrapper, it can sit transparently anywhere in an
acquisition chain — e.g. record the live source, or record while replaying.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import FrameSource

FrameWriter = Callable[[str, Any], Any]


class FrameRecorder:
    """Persists every passing frame to an image sequence, forwarding it untouched."""

    def __init__(
        self,
        source: FrameSource,
        out_dir: str | Path,
        *,
        filename_pattern: str = "frame_{frame_id:06d}.png",
        writer: FrameWriter = cv2.imwrite,
        create_dir: bool = True,
    ) -> None:
        self._source = source
        self._dir = Path(out_dir)
        self._pattern = filename_pattern
        self._writer = writer
        if create_dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def out_dir(self) -> Path:
        """Directory the frames are written to."""

        return self._dir

    def frames(self) -> Iterator[Frame]:
        for frame in self._source.frames():
            path = self._dir / self._pattern.format(frame_id=frame.meta.frame_id)
            self._writer(str(path), frame.image)
            yield frame
