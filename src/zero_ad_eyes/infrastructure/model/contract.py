"""MP2 — the versioned ``PerceptionModel`` I/O contract (provisional on model internals).

The architectural seam of §5.10: the interface every adapter (stub, classical,
future learned) satisfies, enforced by MP5 parity so swapping one for another never
silently becomes a rewrite. A machine-readable value object — a frozen pydantic model
that serialises to/from JSON — diffable in CI.

There is **no model team building a learned model**, so the contract is split by who
each field actually serves:

  **Frozen now** (model-agnostic; this team's downstream depends on it):
  2. the **output schema** (the domain ``Detections`` value object);
  3. the **class-id ↔ taxonomy** mapping — coarse ``EntityKind`` only in v1;
  4. the **coordinate + scale convention**, plus the rule that *every* detection
     carries a first-class confidence and a provenance tag.

  **Provisional / deferred** (model-internal; committed only when a model exists):
  1. the **input tensor** spec (size, channel order, layout, dtype, normalization).
     ``default_contract().input is None`` (``is_provisional``); a real model fills it
     via :func:`committed_contract` at hand-off. Committing specific values now would
     be a guess — the classical and stub adapters never read it anyway.

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
    # Model-internal: the exact input tensor a learned model consumes. Left absent
    # until a real model is delivered — there is no team building one, so committing
    # a specific size/normalization now would be a guess with no empirical basis. The
    # stub and classical adapters never read it; the downstream-invariant half of the
    # contract (output schema, coordinates, per-detection invariants, coarse classes)
    # is what this team freezes and what MP5 parity enforces.
    input: InputTensorSpec | None = None
    classes: ClassMapping
    coordinates: CoordinateConvention = CoordinateConvention()
    detection: DetectionRequirements = DetectionRequirements()

    @property
    def is_provisional(self) -> bool:
        """True while the model-internal input spec is uncommitted (no model yet)."""

        return self.input is None

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


def _coarse_class_map() -> ClassMapping:
    """v1 class map: the closed coarse ``EntityKind`` enum only.

    Exact per-civ types (A3/OQ-4) are deferred to the learned model as an additive
    extension of ``entity_type_by_class_id`` — nothing this team can build today
    emits them, and they are the project's hardest target (R1b), so v1 stays coarse.
    """

    return ClassMapping(
        kind_by_class_id={
            0: EntityKind.UNIT,
            1: EntityKind.BUILDING,
            2: EntityKind.RESOURCE_NODE,
            3: EntityKind.PROJECTILE,
            4: EntityKind.OTHER,
        },
    )


def default_contract() -> ModelIOContract:
    """The **provisional** v1 contract this team freezes for its own downstream.

    Frozen now (model-agnostic, MP5-enforced): the ``Detections`` output schema, the
    coordinate/scale convention, the mandatory confidence+provenance invariants, and
    the coarse class map. **Not** committed: the model-internal input tensor spec —
    ``input`` is ``None`` until an actual model is delivered (``is_provisional``).
    """

    return ModelIOContract(classes=_coarse_class_map())


def committed_contract(
    input_spec: InputTensorSpec, *, classes: ClassMapping | None = None
) -> ModelIOContract:
    """The contract once a real model exists: the frozen v1 seam plus the delivered
    model-internal input tensor spec (and optionally an extended class map). Produced
    at model hand-off (MP4/§5.10.3); ``is_provisional`` is then ``False``."""

    return ModelIOContract(input=input_spec, classes=classes or _coarse_class_map())


MODEL_IO_CONTRACT_V1 = default_contract()
"""The frozen (downstream-invariant) v1 seam. Provisional on model-internal fields
until a model is delivered — see :func:`committed_contract`."""
