"""Command-line entry point (``zero-ad-eyes``).

Trunk scope: a ``run`` command that drives the pipeline with the stub model over a
synthetic in-memory source, emitting world models as JSON. This proves the seam
end-to-end headlessly. Feature agents extend it with real sources (``--recording``),
live capture, and the overlay window.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from zero_ad_eyes.application.pipeline import PerceptionPipeline
from zero_ad_eyes.infrastructure.model.stub_adapter import StubPerceptionModel


def _synthetic_source(n_frames: int, width: int, height: int) -> object:
    import numpy as np

    from zero_ad_eyes.application.frames import Frame
    from zero_ad_eyes.domain.world_model import FrameMeta
    from zero_ad_eyes.infrastructure.acquisition import InMemoryFrameSource

    frames = [
        Frame(
            image=np.zeros((height, width, 3), dtype=np.uint8),
            meta=FrameMeta(
                frame_id=i,
                timestamp=float(i),
                source="synthetic",
                width=width,
                height=height,
            ),
        )
        for i in range(n_frames)
    ]
    return InMemoryFrameSource(frames)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zero-ad-eyes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline (synthetic source, stub model)")
    run.add_argument("--frames", type=int, default=3)
    run.add_argument("--width", type=int, default=1280)
    run.add_argument("--height", type=int, default=720)

    args = parser.parse_args(argv)

    if args.command == "run":
        source = _synthetic_source(args.frames, args.width, args.height)
        pipeline = PerceptionPipeline(source, StubPerceptionModel())  # type: ignore[arg-type]
        for world_model in pipeline.run():
            print(world_model.model_dump_json())
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
