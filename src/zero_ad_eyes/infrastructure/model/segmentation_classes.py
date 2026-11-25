"""The 17 segmentation classes of the engine fork's render pass (``0A.DDataset``).

Class ids are the engine's ``ESegmentationClass`` enum values
(``source/graphics/SegmentationClasses.h``). They are shared by the learned
segmentation adapter (its output channels) and by the offline dataset tooling (the
class id stored in each mask pixel), so they live in one place. Framework-free: this
module must stay importable on the inference path.
"""

from __future__ import annotations

from zero_ad_eyes.domain.taxonomy import EntityKind

#: class-id → label, in engine enum order.
CLASS_NAMES: dict[int, str] = {
    0: "unlabeled",
    1: "terrain_grass",
    2: "terrain_sand",
    3: "terrain_dirt",
    4: "terrain_rock",
    5: "terrain_snow",
    6: "terrain_road",
    7: "terrain_special",
    8: "water",
    9: "resource_wood",
    10: "resource_stone",
    11: "resource_metal",
    12: "resource_food",
    13: "vegetation",
    14: "fauna",
    15: "building",
    16: "unit",
}

NUM_CLASSES = len(CLASS_NAMES)

#: class-id → coarse ``EntityKind``. Terrain, water, vegetation and the backdrop map
#: to OTHER; they are scene, not entities (see ``ENTITY_CLASS_IDS``).
KIND_BY_CLASS_ID: dict[int, EntityKind] = {
    **{class_id: EntityKind.OTHER for class_id in CLASS_NAMES},
    9: EntityKind.RESOURCE_NODE,
    10: EntityKind.RESOURCE_NODE,
    11: EntityKind.RESOURCE_NODE,
    12: EntityKind.RESOURCE_NODE,
    14: EntityKind.UNIT,
    15: EntityKind.BUILDING,
    16: EntityKind.UNIT,
}

#: Classes that are entities worth detecting (resources, fauna, buildings, units).
ENTITY_CLASS_IDS: frozenset[int] = frozenset({9, 10, 11, 12, 14, 15, 16})
