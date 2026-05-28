"""Loaders for the 10 benchmark datasets used in the FedRF-Distill paper.

Each loader returns a :class:`Dataset` dataclass with:

* ``X``   — ``np.float64`` feature matrix.
* ``y``   — ``np.int64`` label vector.
* ``name`` — short canonical name.
* ``task_type`` — ``"binary"`` or ``"multiclass"``.
* ``n_classes`` — number of classes.
* ``feature_names`` — column names (best-effort).

Datasets are downloaded on first access and cached under
``~/.cache/fedrf_distill/<dataset>/``. We prefer sklearn / OpenML loaders over
ad-hoc downloads to keep the dependency surface small.

Curated 10 benchmarks
---------------------
========================  ========================  =========  =========
Dataset                   Task                      Size       Source
========================  ========================  =========  =========
Adult                     Binary income             ~49K       UCI
Bank Marketing            Binary subscription       ~45K       UCI
Credit Card Fraud         Binary fraud              ~285K      Kaggle
HIGGS                     Binary HEP                10M (10%)  UCI/OpenML
Forest Cover Type         Multiclass terrain (7c)   ~580K      UCI
Breast Cancer Wisconsin   Binary cancer             569        sklearn
Letter Recognition        Multiclass (26c)          20K        UCI
NSL-KDD                   Multiclass intrusion      ~125K      Tavallaee'09
Diabetes (Pima)           Binary medical            768        UCI/OpenML
Mushroom                  Binary edible/poisonous   8K         UCI
========================  ========================  =========  =========

These cover small/medium/large, binary/multiclass, tabular/imbalanced regimes
— a comprehensive sweep typical of an A* benchmark.
"""

from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry, load_dataset

# Force-import each loader so it registers itself.
from fedrf_distill.data.loaders import (  # noqa: F401  (side-effect imports)
    adult,
    bank_marketing,
    breast_cancer,
    covertype,
    credit_card_fraud,
    diabetes,
    higgs,
    letter_recognition,
    mushroom,
    nsl_kdd,
)

__all__ = ["Dataset", "DatasetRegistry", "load_dataset"]
