"""The composable preprocessing pipeline (P1).

``PreprocessingPipeline`` is a Composite: it *is* a :class:`PreprocessStep` and it
*holds* a sequence of steps, so pipelines nest inside pipelines. It satisfies the
application ``Preprocessor`` port structurally (it exposes ``process``), which lets
it slot straight into ``PerceptionPipeline`` — while each contained step remains
usable standalone.

Per-consumer variants (a HUD-tuned chain, a scene-tuned chain) are just different
step lists; see :mod:`.variants`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from zero_ad_eyes.application.frames import Frame

from .base import PreprocessStep


class PreprocessingPipeline:
    """Runs an ordered sequence of steps, threading the frame through each.

    The step list is captured as an immutable tuple: a pipeline is a value, not a
    mutable buffer, so sharing one across consumers is safe.
    """

    def __init__(self, steps: Iterable[PreprocessStep] = ()) -> None:
        self._steps: tuple[PreprocessStep, ...] = tuple(steps)

    @property
    def steps(self) -> Sequence[PreprocessStep]:
        """The ordered steps this pipeline applies (read-only view)."""

        return self._steps

    def process(self, frame: Frame) -> Frame:
        """Apply every step in order; satisfies the ``Preprocessor`` port."""

        for step in self._steps:
            frame = step(frame)
        return frame

    # A pipeline is itself a step, so it composes into larger pipelines.
    def __call__(self, frame: Frame) -> Frame:
        return self.process(frame)

    def then(self, step: PreprocessStep) -> PreprocessingPipeline:
        """Return a new pipeline with ``step`` appended (non-mutating)."""

        return PreprocessingPipeline((*self._steps, step))
