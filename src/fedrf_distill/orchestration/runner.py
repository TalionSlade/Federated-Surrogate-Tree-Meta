"""End-to-end experiment runner driven by :class:`ExperimentConfig`.

Pipeline (one call to :meth:`ExperimentRunner.run`):

1. Load dataset, train/test split, partition train across K clients.
2. Build proxy dataset from server side.
3. Instantiate K clients with their per-client framework, DP mechanism, etc.
4. Instantiate the coordinator with aggregator + accountant.
5. For each round t in 1..n_rounds:
     a. Each client runs local_train() → ClientUpdate.
     b. Coordinator aggregates → GlobalArtifact.
     c. Each client.apply_global(artifact) (refines via meta-learner).
     d. Evaluate global surrogate and refined locals on the test split.
6. Return a list of :class:`RoundResult`.

The function exposes seam points (``build_clients``, ``build_coordinator``)
so unit tests can replace either side without re-running data loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fedrf_distill.aggregation import (
    ConfidenceWeightedAggregator,
    MajorityVoteAggregator,
    ProxyRedistillationAggregator,
)
from fedrf_distill.client import FederatedClient
from fedrf_distill.config import ExperimentConfig
from fedrf_distill.coordinator import FederatedCoordinator
from fedrf_distill.core.exceptions import ConfigurationError
from fedrf_distill.core.types import (
    ClientID,
    PrivacyBudget,
    RoundResult,
)
from fedrf_distill.data import (
    DirichletPartitioner,
    IIDPartitioner,
    PathologicalPartitioner,
    ProxyDataConfig,
    QuantitySkewPartitioner,
    build_proxy_dataset,
)
from fedrf_distill.data.loaders import load_dataset
from fedrf_distill.data.proxy import ProxySource
from fedrf_distill.distillation import SoftLabelGenerator, SurrogateDistiller
from fedrf_distill.meta import StackingMetaLearner
from fedrf_distill.metrics import classification_metrics
from fedrf_distill.models import make_model
from fedrf_distill.privacy.accountant import (
    BasicCompositionAccountant,
    RDPAccountant,
)
from fedrf_distill.privacy.mechanisms import (
    GaussianMechanism,
    LaplaceMechanism,
    NullDPMechanism,
)
from fedrf_distill.utils.serialization import deserialize_adapter
from fedrf_distill.utils.timer import Timer


# ── Builders ─────────────────────────────────────────────────────────────────
def _build_partitioner(cfg: ExperimentConfig):
    p = cfg.partition
    if p.strategy == "iid":
        return IIDPartitioner()
    if p.strategy == "dirichlet":
        return DirichletPartitioner(alpha=p.alpha)
    if p.strategy == "pathological":
        return PathologicalPartitioner(k_classes_per_client=p.k_classes_per_client)
    if p.strategy == "quantity_skew":
        return QuantitySkewPartitioner(skew_exponent=p.skew_exponent)
    raise ConfigurationError(f"Unknown partition strategy: {p.strategy}")


def _build_dp_mechanism(cfg: ExperimentConfig):
    rng = np.random.default_rng(cfg.dp.seed)
    if cfg.dp.mechanism == "null":
        return NullDPMechanism(rng=rng)
    if cfg.dp.mechanism == "laplace":
        return LaplaceMechanism(rng=rng)
    if cfg.dp.mechanism == "gaussian":
        return GaussianMechanism(rng=rng)
    raise ConfigurationError(f"Unknown DP mechanism: {cfg.dp.mechanism}")


def _build_aggregator(cfg: ExperimentConfig):
    a = cfg.aggregator
    common = dict(
        student_framework=a.student_framework,
        student_kwargs=dict(a.student_kwargs),
        random_state=cfg.random_state,
    )
    if a.strategy == "proxy_redistillation":
        return ProxyRedistillationAggregator(
            distillation_mode=cfg.distillation.mode, **common
        )
    if a.strategy == "confidence_weighted":
        return ConfidenceWeightedAggregator(
            distillation_mode=cfg.distillation.mode, **common
        )
    if a.strategy == "majority_vote":
        return MajorityVoteAggregator(**common)
    raise ConfigurationError(f"Unknown aggregation strategy: {a.strategy}")


def _build_accountant(cfg: ExperimentConfig):
    if cfg.dp.accountant == "basic":
        return BasicCompositionAccountant()
    if cfg.dp.accountant == "rdp":
        return RDPAccountant(target_delta=cfg.dp.target_delta)
    raise ConfigurationError(f"Unknown accountant: {cfg.dp.accountant}")


def build_clients(
    cfg: ExperimentConfig,
    shards: list[tuple[np.ndarray, np.ndarray]],
) -> list[FederatedClient]:
    """Instantiate K federated clients per the config."""
    n_clients = len(shards)
    # Per-client framework override (Hook B)
    per_client = cfg.model.frameworks_per_client
    if per_client and len(per_client) != n_clients:
        raise ConfigurationError(
            f"frameworks_per_client has {len(per_client)} entries but there are "
            f"{n_clients} clients."
        )

    distiller = SurrogateDistiller(
        student_framework=cfg.model.student_framework,
        mode=cfg.distillation.mode,
        student_kwargs=dict(cfg.model.student_kwargs),
        random_state=cfg.random_state,
    )
    soft_gen = SoftLabelGenerator(
        temperature=cfg.distillation.temperature,
        floor=cfg.distillation.floor,
    )
    meta = (
        StackingMetaLearner(standardise_proba_cols=cfg.meta.standardise_proba_cols)
        if cfg.meta.enabled and cfg.meta.strategy == "stacking"
        else None
    )

    clients: list[FederatedClient] = []
    for k, (X_k, y_k) in enumerate(shards):
        teacher_framework = per_client[k] if per_client else cfg.model.teacher_framework
        teacher_kwargs = dict(cfg.model.teacher_kwargs)
        teacher_kwargs.setdefault("random_state", cfg.random_state + k)

        def factory(
            fwk: str = teacher_framework, kw: dict[str, Any] = teacher_kwargs
        ):  # closure-safe defaults
            return make_model(fwk, **kw)

        dp = _build_dp_mechanism(cfg)
        clients.append(
            FederatedClient(
                client_id=ClientID(k),
                X_local=X_k,
                y_local=y_k,
                teacher_factory=factory,
                soft_label_generator=soft_gen,
                distiller=distiller,
                dp_mechanism=dp,
                epsilon_per_round=cfg.dp.epsilon_per_round,
                delta_per_round=cfg.dp.delta_per_round,
                meta_learner=meta,
            )
        )
    return clients


def build_coordinator(
    cfg: ExperimentConfig,
    classes: list[int],
    n_features: int,
    feature_ref: np.ndarray | None = None,
) -> FederatedCoordinator:
    """Instantiate the federated coordinator."""
    proxy_cfg = ProxyDataConfig(
        source=ProxySource(cfg.proxy.source),
        n_samples=cfg.proxy.n_samples,
        seed=cfg.proxy.seed,
        n_features=n_features,
    )
    proxy_X = build_proxy_dataset(proxy_cfg, external=feature_ref)
    return FederatedCoordinator(
        aggregator=_build_aggregator(cfg),
        proxy_X=proxy_X,
        classes=classes,
        accountant=_build_accountant(cfg),
        target_budget=PrivacyBudget(
            epsilon=cfg.dp.target_epsilon, delta=cfg.dp.target_delta
        ),
    )


# ── Runner ───────────────────────────────────────────────────────────────────
@dataclass
class ExperimentRunner:
    """Stateful driver of the federated round loop."""

    cfg: ExperimentConfig
    round_results: list[RoundResult] = field(default_factory=list, init=False)

    def run(self) -> list[RoundResult]:
        from fedrf_distill.utils.seeding import set_global_seed

        set_global_seed(self.cfg.random_state)

        # 1) Load + split
        ds = load_dataset(self.cfg.data.name)
        n = ds.X.shape[0]
        rng = np.random.default_rng(self.cfg.data.random_state)
        order = rng.permutation(n)
        n_test = int(round(n * self.cfg.data.test_fraction))
        test_idx = order[:n_test]
        train_idx = order[n_test:]
        X_train, y_train = ds.X[train_idx], ds.y[train_idx]
        X_test, y_test = ds.X[test_idx], ds.y[test_idx]
        classes = sorted(int(c) for c in np.unique(ds.y).tolist())

        # 2) Partition train
        partitioner = _build_partitioner(self.cfg)
        shards = partitioner.partition(
            X_train,
            y_train,
            n_clients=self.cfg.partition.n_clients,
            seed=self.cfg.partition.seed,
        )

        # 3) Build clients + coordinator
        clients = build_clients(self.cfg, shards)
        coordinator = build_coordinator(
            self.cfg,
            classes=classes,
            n_features=X_train.shape[1],
            feature_ref=X_train,
        )

        # 4) Round loop
        for t in range(self.cfg.n_rounds):
            with Timer() as timer:
                updates = [c.local_train() for c in clients]
                artifact = coordinator.round(
                    updates, round_idx=t, budget=PrivacyBudget(
                        epsilon=self.cfg.dp.target_epsilon,
                        delta=self.cfg.dp.target_delta,
                    )
                )
                for c in clients:
                    c.apply_global(artifact)
            # 5) Evaluate global on test set
            global_model = deserialize_adapter(
                artifact.global_surrogate_blob, artifact.global_surrogate_class_path
            )
            global_pred = global_model.predict(X_test)
            global_proba = global_model.predict_proba(X_test)
            global_metrics = classification_metrics(y_test, global_pred, global_proba)
            cumulative_eps, cumulative_delta = coordinator.accountant.cumulative()
            total_bytes = sum(len(u.surrogate_blob) for u in updates)

            self.round_results.append(
                RoundResult(
                    round_idx=t,
                    global_accuracy=global_metrics["accuracy"],
                    global_f1_macro=global_metrics["f1_macro"],
                    global_auc=global_metrics.get("auc"),
                    per_client_metrics=[],  # populated by per-client eval in tests
                    total_bytes_communicated=total_bytes,
                    cumulative_epsilon=cumulative_eps,
                    cumulative_delta=cumulative_delta,
                    wall_clock_seconds=timer.elapsed,
                )
            )
        return self.round_results
