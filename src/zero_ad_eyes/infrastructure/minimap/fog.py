"""D4 — Fog-of-war cell classification on the minimap (REQUIREMENTS.md §4.4/§4.5).

0 A.D. renders the three visibility states as brightness tiers: unexplored is black,
explored-but-not-visible is dimmed (shroud), and currently-visible is full
brightness. This stage tiles the minimap into a coarse grid and classifies each cell
by its mean brightness into a :class:`~zero_ad_eyes.domain.minimap.FogState`.

Grid resolution and the two brightness thresholds are configurable (NF7). Results
are returned as a :class:`FogGrid` helper (the domain :class:`MinimapModel` carries
no fog field); the fusion layer (EPIC G) consumes it alongside the main-view fog.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zero_ad_eyes.domain.minimap import FogState

from .segmentation import Segmentation


@dataclass(frozen=True)
class FogGrid:
    """A grid of per-cell :class:`FogState` with row/column accessors."""

    cells: tuple[tuple[FogState, ...], ...]

    @property
    def rows(self) -> int:
        return len(self.cells)

    @property
    def cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def at(self, row: int, col: int) -> FogState:
        return self.cells[row][col]

    def count(self, state: FogState) -> int:
        return sum(cell is state for row in self.cells for cell in row)


@dataclass(frozen=True)
class FogClassifier:
    """Classifies minimap cells into unexplored / explored / visible (D4)."""

    rows: int = 16
    cols: int = 16
    unexplored_max: float = 25.0  # mean brightness below this ⇒ never seen (black)
    visible_min: float = 140.0  # mean brightness at/above this ⇒ currently visible

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("fog grid must have positive dimensions")

    def classify(self, segmentation: Segmentation) -> FogGrid:
        brightness = segmentation.region.max(axis=2).astype(np.float32)
        active = segmentation.mask > 0
        row_edges = np.linspace(0, brightness.shape[0], self.rows + 1, dtype=int)
        col_edges = np.linspace(0, brightness.shape[1], self.cols + 1, dtype=int)

        grid: list[tuple[FogState, ...]] = []
        for r in range(self.rows):
            row_cells: list[FogState] = []
            for c in range(self.cols):
                cell = brightness[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]]
                cell_active = active[
                    row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]
                ]
                row_cells.append(self._classify_cell(cell, cell_active))
            grid.append(tuple(row_cells))
        return FogGrid(cells=tuple(grid))

    def _classify_cell(self, cell: np.ndarray, cell_active: np.ndarray) -> FogState:
        if cell.size == 0 or not cell_active.any():
            return FogState.UNEXPLORED
        mean = float(cell[cell_active].mean())
        if mean < self.unexplored_max:
            return FogState.UNEXPLORED
        if mean >= self.visible_min:
            return FogState.VISIBLE
        return FogState.EXPLORED
