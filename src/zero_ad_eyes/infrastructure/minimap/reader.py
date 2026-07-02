"""EPIC D — ``ClassicalMinimapReader``: the ``MinimapReader`` port adapter.

Composes the D1–D6 collaborators into a single ``read(frame, calibration) ->
MinimapModel``. It is deliberately a *thin orchestrator*: each perception concern
lives in its own object and this class only wires them and projects pixel findings
into world space (D6).

The domain :class:`MinimapModel` carries blips + viewport + confidence. Territory
(D3) and fog (D4) have no field on that frozen contract, so they are exposed as
separate helpers (:meth:`read_territory`, :meth:`read_fog`) for the fusion layer
(EPIC G) to consume — this adapter does not mutate the shared domain schema.

Provenance of everything produced here is ``Provenance.CLASSICAL``.
"""

from __future__ import annotations

from zero_ad_eyes.application.frames import Frame
from zero_ad_eyes.application.settings import MinimapSettings
from zero_ad_eyes.domain.calibration import Calibration
from zero_ad_eyes.domain.confidence import Confidence, Provenance
from zero_ad_eyes.domain.minimap import FogGrid as DomainFogGrid
from zero_ad_eyes.domain.minimap import MinimapModel
from zero_ad_eyes.domain.minimap import TerritoryMap as DomainTerritoryMap
from zero_ad_eyes.domain.minimap import TerritoryRegion as DomainTerritoryRegion

from .blips import BlipDetector
from .fog import FogClassifier, FogGrid
from .palette import MinimapPalette
from .projector import MinimapProjector, WorldExtent
from .segmentation import MinimapSegmenter, MinimapShape, Segmentation
from .territory import TerritoryExtractor, TerritoryMap
from .viewport import ViewportDetector


class ClassicalMinimapReader:
    """Classical, pixel-only ``MinimapReader`` (satisfies the port structurally)."""

    def __init__(
        self,
        *,
        segmenter: MinimapSegmenter,
        blip_detector: BlipDetector,
        viewport_detector: ViewportDetector,
        territory_extractor: TerritoryExtractor,
        fog_classifier: FogClassifier,
        world_extent: WorldExtent,
        region_confidence: float,
    ) -> None:
        self._segmenter = segmenter
        self._blip_detector = blip_detector
        self._viewport_detector = viewport_detector
        self._territory_extractor = territory_extractor
        self._fog_classifier = fog_classifier
        self._world_extent = world_extent
        self._region_confidence = region_confidence

    @classmethod
    def from_settings(cls, settings: MinimapSettings) -> ClassicalMinimapReader:
        """Build the whole D1–D6 chain from pure config (Approach B boundary mapping)."""

        palette = MinimapPalette.from_settings(settings.palette)
        return cls(
            segmenter=MinimapSegmenter(
                MinimapShape.DISC if settings.disc_shape else MinimapShape.SQUARE
            ),
            blip_detector=BlipDetector(
                palette=palette,
                tolerance=settings.blips.tolerance,
                min_area=settings.blips.min_area,
                max_area=settings.blips.max_area,
                confidence=settings.blips.confidence,
            ),
            viewport_detector=ViewportDetector(
                white_min=settings.viewport.white_min,
                min_area=settings.viewport.min_area,
                min_side=settings.viewport.min_side,
                approx_epsilon_fraction=settings.viewport.approx_epsilon_fraction,
            ),
            territory_extractor=TerritoryExtractor(
                palette=palette,
                tolerance=settings.territory.tolerance,
                min_area=settings.territory.min_area,
            ),
            fog_classifier=FogClassifier(
                rows=settings.fog.rows,
                cols=settings.fog.cols,
                unexplored_max=settings.fog.unexplored_max,
                visible_min=settings.fog.visible_min,
            ),
            world_extent=WorldExtent(
                origin_x=settings.world_extent.origin_x,
                origin_y=settings.world_extent.origin_y,
                width=settings.world_extent.width,
                height=settings.world_extent.height,
                flip_y=settings.world_extent.flip_y,
            ),
            region_confidence=settings.region_confidence,
        )

    def read(self, frame: Frame, calibration: Calibration) -> MinimapModel:
        segmentation = self._segment(frame, calibration)
        if segmentation is None:
            return MinimapModel(confidence=Confidence.unknown())

        projector = self._projector(segmentation)
        blips = self._blip_detector.detect(segmentation, projector)
        viewport = self._viewport_detector.detect(segmentation, projector)
        fog = self._domain_fog(self._fog_classifier.classify(segmentation))
        territory = self._domain_territory(
            self._territory_extractor.extract(segmentation, projector), segmentation
        )

        return MinimapModel(
            blips=blips,
            viewport=viewport,
            fog=fog,
            territory=territory,
            confidence=Confidence(value=self._region_confidence, provenance=Provenance.CLASSICAL),
        )

    @staticmethod
    def _domain_fog(grid: FogGrid) -> DomainFogGrid:
        """v0.2: the classifier's grid mapped onto the domain contract (same cells)."""

        return DomainFogGrid(rows=grid.rows, cols=grid.cols, cells=grid.cells)

    @staticmethod
    def _domain_territory(
        territory: TerritoryMap, segmentation: Segmentation
    ) -> DomainTerritoryMap:
        """v0.2: infra regions → domain contract.

        The domain region carries a world-space centroid and a *coverage* fraction
        (region pixels over the whole minimap area); the infra border mask is a
        rendering detail and is dropped. An empty map is a valid "no territory yet".
        """

        total = float(segmentation.width * segmentation.height) or 1.0
        regions = tuple(
            DomainTerritoryRegion(
                ownership=region.ownership,
                centroid=region.world_center,
                coverage=min(1.0, region.area / total),
            )
            for region in territory.regions
        )
        return DomainTerritoryMap(regions=regions)

    def read_territory(self, frame: Frame, calibration: Calibration) -> TerritoryMap | None:
        """D3 side-channel: territory/border regions (not part of ``MinimapModel``)."""

        segmentation = self._segment(frame, calibration)
        if segmentation is None:
            return None
        return self._territory_extractor.extract(segmentation, self._projector(segmentation))

    def read_fog(self, frame: Frame, calibration: Calibration) -> FogGrid | None:
        """D4 side-channel: per-cell fog-of-war grid (not part of ``MinimapModel``)."""

        segmentation = self._segment(frame, calibration)
        if segmentation is None:
            return None
        return self._fog_classifier.classify(segmentation)

    def _segment(self, frame: Frame, calibration: Calibration) -> Segmentation | None:
        if calibration.minimap is None:
            return None
        return self._segmenter.segment(frame.image, calibration.minimap)

    def _projector(self, segmentation: Segmentation) -> MinimapProjector:
        return MinimapProjector(
            region_width=segmentation.width,
            region_height=segmentation.height,
            extent=self._world_extent,
        )
