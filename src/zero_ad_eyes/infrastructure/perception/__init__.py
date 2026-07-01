"""Classical (non-model) perception for EPIC E — the 🔌 boundary's left side.

Everything in this package is deterministic computer vision (template matching,
colour/contour cues, morphology). It is the *classical* half of the model seam
(REQUIREMENTS.md §5.10.1): E3/E4/E5/E6a/E7/E10/E11. The learned tasks
(E1/E2/E6b/E8/E9) live behind the ``PerceptionModel`` port in the model team's
adapter and are **not** implemented here.

By construction, every fact this package emits carries ``Provenance.CLASSICAL``;
nothing here ever produces ``Provenance.LEARNED``.
"""

from __future__ import annotations

from .enrichment import ClassicalEntityEnricher
from .health import locate_health_bar, measure_fill, read_health
from .masks import (
    Component,
    clean_mask,
    connected_components,
    find_contours,
    morphological_close,
    morphological_open,
    to_binary,
)
from .model import ClassicalPerceptionModel
from .occlusion import VisibilityInfo, resolve_occlusions, visible_fraction
from .ownership import assign_ownership, ownership_mask
from .palette import DEFAULT_PALETTE, HsvBand, PlayerColor, PlayerPalette
from .resources import DEFAULT_RESOURCE_CUES, ResourceCue, detect_resource_nodes
from .state import (
    StateCues,
    detect_construction,
    detect_garrison,
    detect_selection,
    read_state_cues,
)
from .templates import (
    Match,
    Template,
    TemplateBank,
    count_feature_matches,
    describe_features,
    match_template,
)

__all__ = [
    "DEFAULT_PALETTE",
    "DEFAULT_RESOURCE_CUES",
    "ClassicalEntityEnricher",
    "ClassicalPerceptionModel",
    "Component",
    "HsvBand",
    "Match",
    "PlayerColor",
    "PlayerPalette",
    "ResourceCue",
    "StateCues",
    "Template",
    "TemplateBank",
    "VisibilityInfo",
    "assign_ownership",
    "clean_mask",
    "connected_components",
    "count_feature_matches",
    "describe_features",
    "detect_construction",
    "detect_garrison",
    "detect_resource_nodes",
    "detect_selection",
    "find_contours",
    "locate_health_bar",
    "match_template",
    "measure_fill",
    "morphological_close",
    "morphological_open",
    "ownership_mask",
    "read_health",
    "read_state_cues",
    "resolve_occlusions",
    "to_binary",
    "visible_fraction",
]
