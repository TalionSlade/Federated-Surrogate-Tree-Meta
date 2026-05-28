"""Knowledge-distillation utilities.

Distillation turns a *teacher* (typically a deep random forest) into a *student*
(a small decision tree) by training the student to match the teacher's
``predict_proba`` output rather than the hard labels. The student is

* small enough to ship over a network in kilobytes,
* differentiable through soft-label noise (essential for the DP path),
* framework-agnostic at the wire level (the coordinator only sees bytes plus
  a class path used for safe re-instantiation).

Two main entry points:

* :class:`SoftLabelGenerator` — runs ``teacher.predict_proba`` with an optional
  temperature and class-prior calibration.
* :class:`SurrogateDistiller` — fits a student adapter on the resulting soft
  labels using either the ``argmax + sample_weight`` route (default) or
  ``multi-output regression`` route (preserves more information).
"""

from fedrf_distill.distillation.soft_labels import (
    SoftLabelGenerator,
    apply_temperature,
)
from fedrf_distill.distillation.surrogate import (
    DistillationMode,
    SurrogateDistiller,
)

__all__ = [
    "SoftLabelGenerator",
    "apply_temperature",
    "SurrogateDistiller",
    "DistillationMode",
]
