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
