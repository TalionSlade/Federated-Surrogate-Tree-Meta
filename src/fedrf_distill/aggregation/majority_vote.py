"""Hard-label majority-vote aggregator — included as the simplest baseline.

Each client surrogate predicts a discrete label on each proxy row; the global
prediction is the mode. We materialise this as a surrogate by training a new
classifier on (proxy_X, vote_labels). No probability information survives —
this is intentionally weaker than re-distillation and serves as the lower-bound
reference in our ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from fedrf_distill.core.exceptions import AggregationError
from fedrf_distill.core.types import ClientUpdate, FeatureMatrix
from fedrf_distill.models.base import BaseModelAdapter
from fedrf_distill.models.factory import make_model
from fedrf_distill.utils.serialization import deserialize_adapter


@dataclass
class MajorityVoteAggregator:
    """Predict hard label on proxy via majority vote, then fit a new student."""

    name: ClassVar[str] = "majority_vote"

    student_framework: str = "sklearn_dt"
    student_kwargs: dict = field(default_factory=dict)
    random_state: int | None = None

    def aggregate(
        self,
        client_updates: list[ClientUpdate],
        proxy_X: FeatureMatrix,
        classes: list[int],
    ) -> BaseModelAdapter:
        if not client_updates:
            raise AggregationError("Cannot aggregate an empty client set.")

        votes = np.zeros((proxy_X.shape[0], len(classes)), dtype=np.int64)
        class_to_col = {c: i for i, c in enumerate(classes)}
        for upd in client_updates:
            student = deserialize_adapter(
                upd.surrogate_blob, upd.surrogate_class_path
            )
            preds = student.predict(proxy_X)
            for i, p in enumerate(preds):
                votes[i, class_to_col[int(p)]] += 1
        vote_labels = np.asarray(classes, dtype=np.int64)[np.argmax(votes, axis=1)]

        kw = dict(self.student_kwargs)
        if self.random_state is not None and "random_state" not in kw:
            kw["random_state"] = self.random_state
        student = make_model(self.student_framework, **kw)
        student.fit(proxy_X, vote_labels)
        return student
