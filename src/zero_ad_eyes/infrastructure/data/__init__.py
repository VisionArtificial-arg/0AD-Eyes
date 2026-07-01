"""Offline data-support infrastructure (REQUIREMENTS.md §7 — ML1/ML2/ML3/ML8).

This package holds the *[this team] / [shared]* pieces of the Data & ML scope:

- **ML1** dataset-collection support: turn a ``FrameSource`` into frames + a
  labelling manifest on disk (``dataset_collection``).
- **ML2** ground-truth alignment scaffolding (D6): a documented engine-state JSON
  schema mapped to domain labels, aligned to captured frames (``ground_truth``).
- **ML3** annotation format: a typed schema for detection / classification /
  ownership / health / fog / minimap labels, with load/save (``annotation``).
- **ML8** evaluation harness: NF3 metrics (HUD read error, detection mAP,
  ownership accuracy, tracking MOTA) comparing predicted vs ground truth, with
  a ``pending-model`` marker for the model-dependent (🔌) metrics (``evaluation``).

Boundary rules (onion): everything here is *offline* infrastructure. It depends
only on ``zero_ad_eyes.domain`` and ``zero_ad_eyes.application`` (the ``FrameSource``
port and the frame carrier). It is **never linked into the inference path** — the
production pipeline reads pixels only (D1/D6). The ground-truth pieces in particular
consume engine-derived state, which is authoritative *for training/eval only*.
"""
