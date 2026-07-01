"""MP2 — the frozen, versioned ``PerceptionModel`` I/O contract.

This is the single cross-team artifact of §5.10: the *only* thing the model team
builds *to*, and the thing MP5 parity tests enforce so "plug-and-play" never
silently becomes a rewrite (risk R6). It is a machine-readable value object — a
frozen pydantic model that serialises to/from JSON — so it can be shipped in the
artifact hand-off (§5.10.3) and diffed in CI.

It pins four things (REQUIREMENTS.md §5.10.2 / MP2):
  1. the **input tensor** spec (size, channel order, layout, dtype, normalization
     expressed in EPIC-P terms);
  2. the **output schema** (the domain ``Detections`` value object);
  3. the **class-id ↔ taxonomy** mapping (§4.3);
  4. the **coordinate + scale convention**, plus the rule that *every* detection
     carries a first-class confidence and a provenance tag.

Nothing here imports a model framework or weights: the contract describes the plug
socket, not the plug.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from zero_ad_eyes.domain.confidence import Provenance
from zero_ad_eyes.domain.detections import Detections
from zero_ad_eyes.domain.taxonomy import EntityKind

CONTRACT_VERSION = "1.0.0"
"""SemVer of the I/O contract. Bump on any breaking change to the fields below."""

OUTPUT_SCHEMA = f"{Detections.__module__}.{Detections.__qualname__}"
"""Fully-qualified name of the output value object the port must return."""


class ChannelOrder(StrEnum):
    """Byte order of the three colour channels in the input tensor."""

    RGB = "rgb"
    BGR = "bgr"  # OpenCV/``Frame.image`` native order


class TensorLayout(StrEnum):
    """Axis order of the batched input tensor."""

    NCHW = "nchw"  # channels-first (PyTorch/ONNX default)
    NHWC = "nhwc"  # channels-last (TensorFlow default)


class BBoxFormat(StrEnum):
    """How a bounding box is encoded."""

    XYWH = "xywh"  # top-left corner + size — matches ``ScreenBBox``
    XYXY = "xyxy"  # two corners


class CoordinateSpace(StrEnum):
    """Which pixel grid detection coordinates are expressed in."""

    SOURCE_FRAME_PIXELS = "source_frame_pixels"  # original captured frame
    INPUT_TENSOR_PIXELS = "input_tensor_pixels"  # the resized model input


class Normalization(BaseModel):
    """EPIC-P normalization the adapter must apply before inference (P3).

    Applied per channel as ``(pixel * scale - mean) / std`` on float pixels. The
    v1 default is a neutral ``[0, 1]`` scaling (``scale = 1/255``, identity
    mean/std) so the model team can swap in learned statistics without a contract
    break, provided EPIC P can express them.
    """

    model_config = ConfigDict(frozen=True)

    scale: float = Field(gt=0.0)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class InputTensorSpec(BaseModel):
    """The exact tensor the adapter feeds the model (§5.10.3 input resolution)."""

    model_config = ConfigDict(frozen=True)

    height: int = Field(gt=0)
    width: int = Field(gt=0)
    channel_order: ChannelOrder
    layout: TensorLayout
    dtype: str
    normalization: Normalization


class ClassMapping(BaseModel):
    """Class-id → taxonomy mapping (§4.3, OQ-4).

    ``kind_by_class_id`` is the closed coarse ``EntityKind`` every id must resolve
    to. ``entity_type_by_class_id`` is the optional fine free-string type (NF7),
    absent when the model only predicts coarse kinds.
    """

    model_config = ConfigDict(frozen=True)

    kind_by_class_id: Mapping[int, EntityKind]
    entity_type_by_class_id: Mapping[int, str] = Field(default_factory=dict)

    def kind_of(self, class_id: int) -> EntityKind:
        """Resolve a raw model class id to its coarse taxonomy kind."""

        return self.kind_by_class_id[class_id]


class CoordinateConvention(BaseModel):
    """Coordinate + scale convention detections are reported in (MP2).

    The port returns ``ScreenBBox`` values, so the adapter is responsible for
    rescaling model outputs from the (possibly resized) input tensor back to
    ``SOURCE_FRAME_PIXELS`` with a top-left origin. Downstream never rescales.
    """

    model_config = ConfigDict(frozen=True)

    origin: str = "top_left"
    bbox_format: BBoxFormat = BBoxFormat.XYWH
    space: CoordinateSpace = CoordinateSpace.SOURCE_FRAME_PIXELS


class DetectionRequirements(BaseModel):
    """Per-detection invariants every adapter (stub or real) must satisfy.

    Confidence and provenance are first-class and mandatory so G/H weight a coarse
    classical guess and a confident learned detection identically regardless of
    which adapter produced them (§5.10.2). ``LEARNED`` is reserved for the real
    adapter; the stub may only claim the deterministic/eval provenances.
    """

    model_config = ConfigDict(frozen=True)

    confidence_required: bool = True
    provenance_required: bool = True
    allowed_provenance: tuple[Provenance, ...] = (
        Provenance.CLASSICAL,
        Provenance.LEARNED,
        Provenance.ENGINE_GT,
        Provenance.FUSED,
    )


class ContractViolation(ValueError):
    """Raised when a ``Detections`` payload breaches the frozen I/O contract."""


class ModelIOContract(BaseModel):
    """The frozen, versioned model I/O contract (MP2).

    Serialise with ``model_dump(mode="json")`` to ship it beside the artifact;
    rehydrate with ``ModelIOContract.model_validate`` to check an incoming plug
    against the socket this team froze.
    """

    model_config = ConfigDict(frozen=True)

    version: str = CONTRACT_VERSION
    output_schema: str = OUTPUT_SCHEMA
    input: InputTensorSpec
    classes: ClassMapping
    coordinates: CoordinateConvention = CoordinateConvention()
    detection: DetectionRequirements = DetectionRequirements()

    def validate_detections(self, detections: Detections, frame_id: int) -> None:
        """Assert a ``Detections`` payload honours this contract.

        Checks the frame id matches and that every item carries an allowed
        provenance (confidence is a mandatory, already-bounded field on the domain
        model). Raises ``ContractViolation`` on the first breach; returns ``None``
        when the payload is conformant. Shared by the fixture loader (MP3) and the
        parity harness (MP5) so both enforce identically.
        """

        if detections.frame_id != frame_id:
            raise ContractViolation(
                f"frame_id mismatch: detections={detections.frame_id} expected={frame_id}"
            )
        allowed = set(self.detection.allowed_provenance)
        for index, item in enumerate(detections.items):
            provenance = item.confidence.provenance
            if self.detection.provenance_required and provenance not in allowed:
                raise ContractViolation(
                    f"detection[{index}] provenance {provenance!r} "
                    f"not in allowed set {sorted(allowed)}"
                )


def default_contract() -> ModelIOContract:
    """The v1 contract this team freezes; the model team builds to this.

    The class map covers only the coarse ``EntityKind`` closed enum — the fine
    per-civ types (OQ-4) are delivered by the model team as an extension of
    ``entity_type_by_class_id`` without a contract break.
    """

    return ModelIOContract(
        input=InputTensorSpec(
            height=640,
            width=640,
            channel_order=ChannelOrder.RGB,
            layout=TensorLayout.NCHW,
            dtype="float32",
            normalization=Normalization(
                scale=1.0 / 255.0,
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            ),
        ),
        classes=ClassMapping(
            kind_by_class_id={
                0: EntityKind.UNIT,
                1: EntityKind.BUILDING,
                2: EntityKind.RESOURCE_NODE,
                3: EntityKind.PROJECTILE,
                4: EntityKind.OTHER,
            },
        ),
    )


MODEL_IO_CONTRACT_V1 = default_contract()
"""The frozen v1 contract instance. Import this as the canonical seam spec."""
