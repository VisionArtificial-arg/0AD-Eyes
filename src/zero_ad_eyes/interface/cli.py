"""Command-line entry point (``zero-ad-eyes``).

Trunk scope: a ``run`` command that drives the pipeline with the stub model over a
synthetic in-memory source, emitting world models as JSON. This proves the seam
end-to-end headlessly. Feature agents extend it with real sources (``--recording``),
live capture, and the overlay window.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

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


def _format_metric(metric: object) -> str:
    from zero_ad_eyes.infrastructure.data import MetricResult

    assert isinstance(metric, MetricResult)
    if metric.is_pending:
        return f"  {metric.name}: pending-model (needs the trained model, MP4)"
    verdict = {True: "PASS", False: "FAIL", None: "n/a"}[metric.passed]
    bound = "<=" if not metric.higher_is_better else ">="
    threshold = "" if metric.threshold is None else f" (target {bound} {metric.threshold})"
    return f"  {metric.name}: {metric.value:.4f} [{verdict}]{threshold}"


def _run_eval(dataset: str | None) -> int:
    """ML8 accuracy gate. Honest by construction: with no ground-truth dataset it
    reports the classical metrics as unmeasured and detection mAP as pending-model,
    rather than scoring empty inputs. With a dataset it runs the real harness and
    exits non-zero only on a measured failure."""

    from zero_ad_eyes.infrastructure.data import evaluate

    if not dataset or not Path(dataset).exists():
        print("eval: no ground-truth dataset supplied (pass --dataset PATH once recordings exist)")
        print("  detection_map: pending-model (needs the trained model, MP4)")
        print("  hud_read_error / ownership_accuracy / tracking_mota: unmeasured (no ground truth)")
        return 0

    raw = json.loads(Path(dataset).read_text(encoding="utf-8"))
    from zero_ad_eyes.domain.world_model import WorldModel

    predicted = [WorldModel.model_validate(item) for item in raw.get("predicted", [])]
    truth = [WorldModel.model_validate(item) for item in raw.get("truth", [])]
    report = evaluate(predicted, truth)
    for metric in report.metrics:
        print(_format_metric(metric))
    verdict = report.passed
    print(f"eval: {'PENDING' if verdict is None else ('PASS' if verdict else 'FAIL')}")
    return 1 if verdict is False else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zero-ad-eyes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline (synthetic source, stub model)")
    run.add_argument("--frames", type=int, default=3)
    run.add_argument("--width", type=int, default=1280)
    run.add_argument("--height", type=int, default=720)

    ev = sub.add_parser("eval", help="run the ML8 accuracy harness (NF3 metrics)")
    ev.add_argument("--dataset", default=None, help="JSON with {predicted, truth} world models")

    args = parser.parse_args(argv)

    if args.command == "run":
        source = _synthetic_source(args.frames, args.width, args.height)
        pipeline = PerceptionPipeline(source, StubPerceptionModel())  # type: ignore[arg-type]
        for world_model in pipeline.run():
            print(world_model.model_dump_json())
        return 0

    if args.command == "eval":
        return _run_eval(args.dataset)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
