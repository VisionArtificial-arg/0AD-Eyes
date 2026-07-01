"""MP5 — the reusable adapter-parity harness.

One function, ``assert_satisfies_contract``, asserts that *any* object satisfying
the ``PerceptionModel`` port also honours the frozen MP2 I/O contract. It is the
executable form of "plug-and-play": the stub runs it today, and the real adapter
(MP4) is expected to pass the identical harness on the same golden fixtures, so a
contract-breaking model is caught at hand-off, not in production (risk R6).

It checks, for a given frame, that the model:
  * returns a domain ``Detections`` value object;
  * tags it with the frame's own ``frame_id``;
  * gives every detection a bounded confidence and an allowed provenance
    (delegated to ``ModelIOContract.validate_detections``);
  * *accepts and handles* the optional ``roi`` argument — the contract requires
    the parameter be honoured (structurally), not that detections be spatially
    filtered, which is an adapter-specific policy.

Assertions are plain ``assert`` / raises so the harness reads naturally inside a
pytest test yet stays a library helper with no pytest dependency.
"""

from __future__ import annotations

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.ports import PerceptionModel
from zero_ad_eyes.domain.detections import Detections
from zero_ad_eyes.domain.geometry import ScreenBBox

from .contract import MODEL_IO_CONTRACT_V1, ModelIOContract


def assert_satisfies_contract(
    model: PerceptionModel,
    frame: Frame,
    *,
    contract: ModelIOContract = MODEL_IO_CONTRACT_V1,
    roi: ScreenBBox | None = None,
) -> Detections:
    """Assert ``model`` honours ``contract`` for ``frame``; return its detections.

    Runs the model with ``roi=None`` and again with a concrete ``roi`` (a small
    box, or the caller-supplied one), asserting both calls yield contract-valid
    detections for the frame. Returns the ``roi=None`` result for further
    frame-specific assertions by the caller.
    """

    frame_id = frame.meta.frame_id

    baseline = model.infer(frame)
    _assert_valid(baseline, frame_id, contract)

    probe_roi = roi if roi is not None else _default_roi(frame)
    gated = model.infer(frame, probe_roi)
    _assert_valid(gated, frame_id, contract)

    return baseline


def _assert_valid(result: object, frame_id: int, contract: ModelIOContract) -> None:
    assert isinstance(result, Detections), (
        f"infer must return Detections, got {type(result).__name__}"
    )
    # Confidence boundedness is enforced by the domain model; the contract covers
    # the frame-id and provenance invariants and raises on any breach.
    contract.validate_detections(result, frame_id)


def _default_roi(frame: Frame) -> ScreenBBox:
    return ScreenBBox(
        x=0.0,
        y=0.0,
        width=float(frame.meta.width) / 2.0,
        height=float(frame.meta.height) / 2.0,
    )
