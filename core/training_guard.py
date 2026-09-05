"""training_guard.py — backward-compatible re-export of :mod:`core.adaptation`.

All training-adaptation logic now lives in ``core.adaptation`` (the single
source of truth).  This module only re-exports the canonical names so legacy
imports keep working; new code should import directly from ``core.adaptation``.

The previously-duplicated implementations here (``Watchdog`` with its arbitrary
``ce > 15`` threshold and ``×0.5`` LR halving, ``ReadinessActivator`` with its
degenerate ``meta_maturity`` proxy and fixed ``stage_steps`` schedule,
``CosineWarmup`` conflicting with the notebook's ``MirrorLRScheduler``) have
been removed to eliminate method duplication.  Their principled successors are
``FailureDetector``, ``DepthController`` and ``LRController`` in
``core.adaptation``.
"""

from .adaptation import (  # noqa: F401
    LossBalancer,
    DepthController,
    LRController,
    FailureDetector,
    GradientClipper,
    set_active_depth,
    build_optimizer,
)

__all__ = [
    "LossBalancer",
    "DepthController",
    "LRController",
    "FailureDetector",
    "GradientClipper",
    "set_active_depth",
    "build_optimizer",
]
