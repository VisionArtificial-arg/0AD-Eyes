"""Entity taxonomy and enumerations (REQUIREMENTS.md §4.3, OQ-4).

Scope decision OQ-4 is *multi-civilization, exact entity types*. The coarse
``EntityKind`` is a closed enum; the fine ``entity_type`` (e.g. "female_citizen",
"house", "oak_tree") is a free string resolved against a config-driven registry so
the taxonomy can grow without code changes (NF7). This module fixes only the
closed, cross-civ enums.
"""

from __future__ import annotations

from enum import StrEnum


class EntityKind(StrEnum):
    """Coarse, civ-independent class of a perceived entity."""

    UNIT = "unit"
    BUILDING = "building"
    RESOURCE_NODE = "resource_node"
    PROJECTILE = "projectile"
    OTHER = "other"


class Ownership(StrEnum):
    """Who a screen entity belongs to, inferred from player colour."""

    SELF = "self"
    ALLY = "ally"
    ENEMY = "enemy"
    GAIA = "gaia"  # neutral / nature
    UNKNOWN = "unknown"


class ResourceType(StrEnum):
    """The four stockpiled resources shown in the top HUD bar."""

    FOOD = "food"
    WOOD = "wood"
    STONE = "stone"
    METAL = "metal"


class Phase(StrEnum):
    """Game phase (REQUIREMENTS.md §4.2)."""

    VILLAGE = "village"
    TOWN = "town"
    CITY = "city"
    UNKNOWN = "unknown"
