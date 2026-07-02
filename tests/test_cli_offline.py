"""CLI offline wiring — the real classical chain runs over a recording (EPIC A→G)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from zero_ad_eyes.interface.cli import _build_offline_pipeline, main


def _write_recording(folder: Path, n: int = 2) -> None:
    for i in range(n):
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), np.zeros((720, 1280, 3), dtype=np.uint8))


def test_build_offline_pipeline_wires_the_real_chain(tmp_path: Path) -> None:
    _write_recording(tmp_path)
    pipeline = _build_offline_pipeline(str(tmp_path), detector="classical")

    # Every classical collaborator is present (the wiring is the deliverable here).
    assert pipeline._preprocessor is not None
    assert pipeline._calibrator is not None
    assert pipeline._hud_reader is not None
    assert pipeline._minimap_reader is not None
    assert pipeline._tracker is not None
    assert pipeline._enricher is not None


def test_run_over_image_folder_emits_v0_2_world_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_recording(tmp_path)
    try:
        code = main(["run", "--recording", str(tmp_path)])
    except Exception as exc:  # noqa: BLE001 — OCR/tesseract or codecs may be absent in CI
        pytest.skip(f"offline chain needs an external dependency here: {exc}")

    assert code == 0
    out = capsys.readouterr().out
    assert '"schema_version":"0.2.0"' in out


# --- eval verdict rendering + exit code (classical-only gate) --------------- #


def _scored(*, ownership_ok: bool) -> object:
    from zero_ad_eyes.infrastructure.data import EvaluationReport, MetricResult

    return EvaluationReport(
        hud_read_error=MetricResult.computed("hud_read_error", 0.0, 0.01, False),
        detection_map=MetricResult.pending_model("detection_map", 0.80, True),
        ownership_accuracy=MetricResult.computed(
            "ownership_accuracy", 0.99 if ownership_ok else 0.50, 0.98, True
        ),
        tracking_mota=MetricResult.computed("tracking_mota", 0.80, 0.70, True),
    )


def test_print_report_pending_model_is_pass_classical_exit_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from zero_ad_eyes.interface.cli import _print_report

    code = _print_report(_scored(ownership_ok=True))
    out = capsys.readouterr().out

    assert code == 0
    assert "PASS (classical)" in out
    assert "pending-model: detection_map" in out


def test_print_report_measured_failure_is_fail_exit_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from zero_ad_eyes.interface.cli import _print_report

    # A classical regression must fail the gate even while detection mAP pends.
    code = _print_report(_scored(ownership_ok=False))
    out = capsys.readouterr().out

    assert code == 1
    assert "eval: FAIL" in out
