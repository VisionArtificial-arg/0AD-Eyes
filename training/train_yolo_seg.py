"""Train YOLO11-seg on a `zero-ad-eyes dataset build` output (MOD-2).

Standalone on purpose: on Kaggle or Colab you need only this file and the dataset,
not the 0AD-Eyes package.

Kaggle / Colab
--------------
1. Zip the dataset build output (``images/``, ``labels/``, ``data.yaml``,
   ``report.json``) and upload it as a Kaggle dataset (or to Drive for Colab).
2. In a GPU notebook::

       !pip install -q "ultralytics>=8.3,<9"
       !python train_yolo_seg.py --data /kaggle/input/<dataset>/data.yaml \
           --project /kaggle/working/runs --name v1

3. Download ``<project>/<name>/`` (``weights/best.pt`` + ``sidecar.json``) into the repo
   and point ``--detector yolo`` at it.

What it does
------------
- Rewrites ``data.yaml``'s ``path`` to where the dataset actually is: the build writes
  an absolute path from the machine that built it.
- Trains with Ultralytics (imgsz 1280 by default: units are only a few pixels wide at
  640), with early stopping.
- Scores the best weights on the **test** split. That split holds whole held-out
  matches, so the numbers are honest.
- Writes ``sidecar.json`` next to the weights: input size, class names, the base model,
  the test metrics, and the dataset build report. The adapter reads it, so the model's
  class map travels with the weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SIDECAR_FORMAT = "zero-ad-eyes/yolo-seg/1"


def prepare_data_yaml(data_yaml: Path, fixed: Path) -> Path:
    """Copy ``data.yaml`` to ``fixed`` with ``path`` pointing at the dataset; return it."""

    from ultralytics.utils import YAML

    spec = dict(YAML.load(str(data_yaml)))
    spec["path"] = str(data_yaml.resolve().parent)
    fixed.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(str(fixed), spec)
    return fixed


def class_names(data_yaml: Path) -> dict[int, str]:
    from ultralytics.utils import YAML

    names = YAML.load(str(data_yaml))["names"]
    pairs = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(k): str(v) for k, v in pairs}


def metrics_summary(results: Any) -> dict[str, Any]:
    """The box and mask mAPs (overall + per class) from an Ultralytics val result."""

    names = getattr(results, "names", {}) or {}
    summary: dict[str, Any] = {}
    for kind in ("box", "seg"):
        metric = getattr(results, kind, None)
        if metric is None:
            continue
        per_class = getattr(metric, "maps", None)
        summary[kind] = {
            "map50": float(metric.map50),
            "map50_95": float(metric.map),
            "precision": float(metric.mp),
            "recall": float(metric.mr),
            "per_class_map50_95": (
                {names.get(i, str(i)): float(v) for i, v in enumerate(per_class)}
                if per_class is not None
                else {}
            ),
        }
    return summary


def write_sidecar(
    run_dir: Path,
    *,
    model: str,
    imgsz: int,
    names: dict[int, str],
    epochs: int,
    test_metrics: dict[str, Any],
    data_report: dict[str, Any] | None,
) -> Path:
    sidecar = {
        "format": SIDECAR_FORMAT,
        "weights": "weights/best.pt",
        "base_model": model,
        "imgsz": imgsz,
        "names": {str(i): n for i, n in sorted(names.items())},
        "epochs": epochs,
        "test_metrics": test_metrics,
        "data_report": data_report,
    }
    path = run_dir / "sidecar.json"
    path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, type=Path, help="data.yaml from `dataset build`")
    p.add_argument("--model", default="yolo11n-seg.pt", help="base weights or a model .yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=float, default=-1, help="-1 = auto-fit GPU memory")
    p.add_argument("--patience", type=int, default=15, help="early-stopping epochs")
    p.add_argument("--project", type=Path, default=Path("runs"))
    p.add_argument("--name", default="v1")
    p.add_argument("--device", default=None, help="e.g. 0, cpu (default: auto)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="none",
        help="cache decoded images; ram skips PNG decoding, the CPU bottleneck on Kaggle",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", choices=("none", "onnx", "ncnn"), default="none")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO

    run_dir = (args.project / args.name).resolve()
    # Beside the run dir, not in it: multi-GPU (DDP) training recreates the run dir when it
    # launches its workers, which deleted a data.yaml kept inside.
    data = prepare_data_yaml(args.data, args.project.resolve() / f"{args.name}.data.yaml")
    batch = int(args.batch) if args.batch >= 1 else args.batch

    model = YOLO(args.model)
    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        seed=args.seed,
        deterministic=True,
        plots=True,
    )
    best = run_dir / "weights" / "best.pt"
    tuned = YOLO(str(best))
    # Training may span GPUs ("0,1", DDP); validation runs on one.
    val_device = args.device.split(",")[0] if args.device else None
    test = tuned.val(data=str(data), split="test", imgsz=args.imgsz, device=val_device, plots=True)
    report_path = args.data.resolve().parent / "report.json"
    sidecar = write_sidecar(
        run_dir,
        model=args.model,
        imgsz=args.imgsz,
        names=class_names(data),
        epochs=args.epochs,
        test_metrics=metrics_summary(test),
        data_report=json.loads(report_path.read_text()) if report_path.is_file() else None,
    )
    if args.export != "none":
        tuned.export(format=args.export, imgsz=args.imgsz)
    print(json.dumps(json.loads(sidecar.read_text())["test_metrics"], indent=2))
    print(f"weights: {best}\nsidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
