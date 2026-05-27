"""Meta-learning: refining each client's local model with the global surrogate.

The canonical strategy is *stacking*: augment the client's feature matrix with
the global surrogate's predicted probabilities, then retrain the local model
on the wider matrix. This lets the local model exploit cross-client knowledge
without ever seeing other clients' data — only their distilled summary.
"""

from fedrf_distill.meta.stacking import StackingMetaLearner

__all__ = ["StackingMetaLearner"]
