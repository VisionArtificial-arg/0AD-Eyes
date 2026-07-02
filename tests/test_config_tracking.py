"""Config-externalization guards for the tracking subsystem (Approach B, P2).

Golden: building the tracker / event detector from default TrackingSettings
reproduces the historical knobs. Threading: config-file values reach the adapters.
"""

from __future__ import annotations

from pathlib import Path

from zero_ad_eyes.application.settings import TrackingSettings
from zero_ad_eyes.infrastructure.config import load_config
from zero_ad_eyes.infrastructure.tracking.events import ClassicalEventDetector, EventDetector
from zero_ad_eyes.infrastructure.tracking.tracker import IouTracker


def test_tracker_from_default_settings_reproduces_knobs() -> None:
    tracker = IouTracker.from_settings(TrackingSettings())
    assert tracker._iou_threshold == 0.3
    assert tracker._min_hits == 1
    assert tracker._max_staleness == 15
    assert tracker._decay == 0.85


def test_event_detector_from_default_settings_reproduces_knobs() -> None:
    detector = EventDetector.from_settings(TrackingSettings())
    assert detector._combat_drop == 0.05
    assert detector._depletion_health == 0.02


def test_classical_event_detector_from_settings_wraps_configured_detector() -> None:
    adapter = ClassicalEventDetector.from_settings(TrackingSettings(combat_drop=0.2))
    assert adapter._detector._combat_drop == 0.2


def test_config_file_threads_tracking_knobs(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"tracking": {"iou_threshold": 0.5, "combat_drop": 0.1}}', encoding="utf-8")

    config = load_config(path, env={})
    tracker = IouTracker.from_settings(config.tracking)
    detector = EventDetector.from_settings(config.tracking)

    assert tracker._iou_threshold == 0.5
    assert tracker._max_staleness == 15  # untouched default
    assert detector._combat_drop == 0.1
