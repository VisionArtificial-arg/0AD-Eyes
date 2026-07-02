"""Tests for the Wayland capture backend (A1) — grim-style screenshot CLI grabber."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
import pytest

from zero_ad_eyes.infrastructure.acquisition import (
    MssGrabber,
    ScreenCaptureSource,
    WaylandGrabber,
)
from zero_ad_eyes.infrastructure.acquisition.screen import CaptureRegion
from zero_ad_eyes.interface.default_config import default_config


def _png_runner(
    image: np.ndarray,
) -> tuple[Callable[[Sequence[str]], bytes], list[Sequence[str]]]:
    """A fake command runner that returns ``image`` PNG-encoded, logging the command."""

    encoded = cv2.imencode(".png", image)[1].tobytes()
    calls: list[Sequence[str]] = []

    def runner(command: Sequence[str]) -> bytes:
        calls.append(command)
        return encoded

    return runner, calls


def test_grab_decodes_the_command_output_to_bgr() -> None:
    # A distinct BGR image round-trips through PNG encode → the grabber's decode.
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :, 2] = 200  # red channel (BGR) so we can tell channels didn't swap
    runner, calls = _png_runner(image)

    grabber = WaylandGrabber(("grim", "-"), runner=runner)
    out = grabber.grab()

    assert out.shape == (8, 10, 3)
    assert np.array_equal(out, image)
    assert calls == [("grim", "-")]  # ran exactly the configured command


def test_grab_crops_to_region() -> None:
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    runner, _calls = _png_runner(image)
    region = CaptureRegion(top=5, left=4, width=10, height=6)

    grabber = WaylandGrabber(("grim", "-"), region=region, runner=runner)
    out = grabber.grab()

    assert out.shape == (6, 10, 3)
    assert np.array_equal(out, image[5:11, 4:14])


def test_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        WaylandGrabber(())


def test_grab_raises_on_undecodable_output() -> None:
    def runner(command: Sequence[str]) -> bytes:
        return b"not an image"

    with pytest.raises(OSError, match="cannot decode image"):
        WaylandGrabber(("grim", "-"), runner=runner).grab()


def test_from_settings_selects_wayland_backend() -> None:
    settings = default_config().acquisition.model_copy(update={"capture_backend": "wayland"})

    source = ScreenCaptureSource.from_settings(settings)

    assert isinstance(source._grabber, WaylandGrabber)


def test_from_settings_defaults_to_mss_backend() -> None:
    source = ScreenCaptureSource.from_settings(default_config().acquisition)

    assert isinstance(source._grabber, MssGrabber)


def test_injected_grabber_overrides_backend_selection() -> None:
    settings = default_config().acquisition.model_copy(update={"capture_backend": "wayland"})
    runner, _calls = _png_runner(np.zeros((2, 2, 3), dtype=np.uint8))
    sentinel = WaylandGrabber(("custom",), runner=runner)

    source = ScreenCaptureSource.from_settings(settings, grabber=sentinel)

    assert source._grabber is sentinel
