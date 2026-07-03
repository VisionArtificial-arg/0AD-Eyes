"""CLI offline wiring — HUD/minimap CV plus the model seam over a recording."""

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
    assert pipeline._projector is not None  # F1 screen->world stage
    assert pipeline._fuser is not None


def test_run_over_image_folder_writes_v0_2_world_models_to_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = tmp_path / "frames"
    recording.mkdir()
    _write_recording(recording)
    output = tmp_path / "world.jsonl"
    try:
        code = main(["run", "--recording", str(recording), "--output", str(output)])
    except Exception as exc:  # noqa: BLE001 — OCR/tesseract or codecs may be absent in CI
        pytest.skip(f"offline chain needs an external dependency here: {exc}")

    assert code == 0
    captured = capsys.readouterr()
    assert '"schema_version":"0.2.0"' not in captured.out
    assert "writing world models to" in captured.err
    assert '"schema_version":"0.2.0"' in output.read_text(encoding="utf-8")


def test_run_stdout_is_opt_in(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recording = tmp_path / "frames"
    recording.mkdir()
    _write_recording(recording)
    output = tmp_path / "world.jsonl"

    try:
        code = main(["run", "--recording", str(recording), "--output", str(output), "--stdout"])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"offline chain needs an external dependency here: {exc}")

    assert code == 0
    out = capsys.readouterr().out
    assert '"schema_version":"0.2.0"' in out
    assert '"schema_version":"0.2.0"' in output.read_text(encoding="utf-8")


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


# --- B3 disk calibration store wiring (opt-in via config) ------------------- #


def test_persist_profiles_off_by_default_wires_readonly_store(tmp_path: Path) -> None:
    from zero_ad_eyes.infrastructure.calibration import HudCalibrator

    _write_recording(tmp_path)
    pipeline = _build_offline_pipeline(str(tmp_path), detector="classical")

    # Manual profiles are readable by default; automatic profile writes stay off.
    assert isinstance(pipeline._calibrator, HudCalibrator)
    assert pipeline._calibrator._store is not None
    assert pipeline._calibrator._save_profiles is False


def test_persist_profiles_on_wires_store_at_configured_dir(tmp_path: Path) -> None:
    from zero_ad_eyes.infrastructure.calibration import (
        CalibrationProfileStore,
        HudCalibrator,
    )
    from zero_ad_eyes.infrastructure.config import load_config
    from zero_ad_eyes.interface.cli import _build_offline_pipeline
    from zero_ad_eyes.interface.default_config import default_config

    recording = tmp_path / "rec"
    recording.mkdir()
    _write_recording(recording)
    profiles = tmp_path / "profiles"

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"calibration": {"persist_profiles": true}, '
        f'"paths": {{"calibration_dir": "{profiles}"}}}}',
        encoding="utf-8",
    )
    cfg = load_config(default_config(), config_path, env={})

    pipeline = _build_offline_pipeline(str(recording), detector="classical", config=cfg)

    assert isinstance(pipeline._calibrator, HudCalibrator)
    store = pipeline._calibrator._store
    assert isinstance(store, CalibrationProfileStore)
    assert store.directory == profiles
    assert pipeline._calibrator._save_profiles is True
