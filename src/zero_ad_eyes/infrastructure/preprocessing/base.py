"""Shared vocabulary for preprocessing steps (EPIC P).

A *step* is an autonomous, composable collaborator: it receives a ``Frame`` and
returns a ``Frame``. The step protocol is intentionally tiny — the same shape as
the ``Preprocessor`` port's ``process`` — so a single step, a hand-built chain, or
a whole :class:`PreprocessingPipeline` are all interchangeable.

Steps preserve provenance: they replace ``frame.image`` and carry ``frame.meta``
through untouched (a step conditions pixels, it does not re-capture them). Most
steps only care about the pixels, so :class:`ImageStep` factors out the
``meta``-preserving rewrap and asks subclasses for a pure array→array transform.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

from zero_ad_eyes.application.frames import Frame

# An OpenCV image buffer. Kept as ``Any`` for the same reason ``Frame.image`` is:
# we do not want a hard numpy dependency leaking into type-only positions.
Image = Any


@runtime_checkable
class PreprocessStep(Protocol):
    """A composable frame→frame conditioning operation."""

    def __call__(self, frame: Frame) -> Frame: ...


class ImageStep:
    """Base for steps whose behaviour is a pure ``image -> image`` transform.

    Subclasses implement :meth:`transform`; this base guarantees the ``Frame``
    contract — ``meta`` is preserved, only ``image`` changes — so no subclass can
    accidentally drop provenance.
    """

    def transform(self, image: Image) -> Image:
        raise NotImplementedError

    def __call__(self, frame: Frame) -> Frame:
        return replace(frame, image=self.transform(frame.image))
