"""End-to-end evaluation pipeline.

:class:`EvaluationPipeline` is a drop-in extension of
:class:`fedrf_distill.orchestration.ExperimentRunner` that captures the metrics
the paper needs but the bare runner skips:

* per-client local accuracy / F1 / AUC after every round,
* surrogate fidelity (KL teacher→student) per client,
* upload bytes per client (the bare runner only records the sum),
* optional Membership Inference Attack (MIA) on the final global surrogate.

Why a separate class?
---------------------
The bare runner is intentionally minimal so unit tests stay fast. The eval
pipeline is heavier and has optional dependencies (matplotlib for plots, etc.).
Keeping them split lets researchers run sweeps without paying the eval cost.

Reuses the existing builder functions in ``orchestration.runner`` so partition,
client, coordinator, and proxy construction stay in lock-step with the runner.
The round-loop body is re-implemented (rather than monkey-patched) so the
instrumentation points are explicit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fedrf_distill.config import ExperimentConfig
from fedrf_distill.core.types import (
    ClientID,
    ClientRoundMetrics,
    PrivacyBudget,
    RoundResult,
)
from fedrf_distill.data.loaders import load_dataset
from fedrf_distill.metrics.classification import (
    classification_metrics,
    surrogate_fidelity,
)
from fedrf_distill.orchestration.runner import (
    _build_partitioner,
    build_clients,
    build_coordinator,
)
from fedrf_distill.utils.serialization import deserialize_adapter
from fedrf_distill.utils.timer import Timer

from fedrf_distill.eval.mia import MIAResult, yeom_membership_inference


# ── Result types ─────────────────────────────────────────────────────────────
@dataclass
class EvaluationResult:
    """Comprehensive output of one :meth:`EvaluationPipeline.run` invocation.

    Holds the same per-round payload as the bare runner *plus* per-client
    metrics, fidelity, and optional MIA. Serialisable to JSON via
    :meth:`to_dict`.
    """

    config_name: str
    dataset: str
    dp_mechanism: str
    epsilon_per_round: float
    n_clients: int
    n_rounds: int
    round_results: list[RoundResult]
    mia: MIAResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Convenience accessors ───────────────────────────────────────────────
    @property
    def final_accuracy(self) -> float:
        return self.round_results[-1].global_accuracy if self.round_results else float("nan")

    @property
    def final_f1(self) -> float:
        return self.round_results[-1].global_f1_macro if self.round_results else float("nan")

    @property
    def total_bytes(self) -> int:
        return sum(r.total_bytes_communicated for r in self.round_results)

    @property
    def final_epsilon(self) -> float:
        return self.round_results[-1].cumulative_epsilon if self.round_results else 0.0

    # ── (De)serialisation ───────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "dataset": self.dataset,
            "dp_mechanism": self.dp_mechanism,
            "epsilon_per_round": self.epsilon_per_round,
            "n_clients": self.n_clients,
            "n_rounds": self.n_rounds,
            "round_results": [_asdict_round(r) for r in self.round_results],
            "mia": asdict(self.mia) if self.mia is not None else None,
            "extra": self.extra,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))


def _asdict_round(r: RoundResult) -> dict[str, Any]:
    return {
        "round_idx": r.round_idx,
        "global_accuracy": r.global_accuracy,
        "global_f1_macro": r.global_f1_macro,
        "global_auc": r.global_auc,
        "per_client_metrics": [asdict(m) for m in r.per_client_metrics],
        "total_bytes_communicated": r.total_bytes_communicated,
        "cumulative_epsilon": r.cumulative_epsilon,
        "cumulative_delta": r.cumulative_delta,
        "wall_clock_seconds": r.wall_clock_seconds,
        "extra": r.extra,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON-serialisable")


# ── Pipeline ─────────────────────────────────────────────────────────────────
@dataclass
class EvaluationPipeline:
    """Run one experiment end-to-end and capture the full eval payload.

    Parameters
    ----------
    cfg:
        Validated :class:`ExperimentConfig` — same object the bare runner takes.
    compute_per_client:
        Populate :attr:`RoundResult.per_client_metrics` each round. Adds
        roughly K × test_size predictions per round. Default ``True``.
    compute_fidelity:
        Compute KL(teacher‖surrogate) per client. Cheap but adds a forward
        pass through both models on the proxy. Default ``True``.
    compute_mia:
        Run a Yeom-style membership inference attack on the final-round
        global surrogate. Default ``True``.
    mia_n_members, mia_n_nonmembers:
        Sample budgets for the MIA. Larger → tighter empirical curve, slower.
    """

    cfg: ExperimentConfig
    compute_per_client: bool = True
    compute_fidelity: bool = True
    compute_mia: bool = True
    mia_n_members: int = 500
    mia_n_nonmembers: int = 500

    def run(self) -> EvaluationResult:
        from fedrf_distill.utils.seeding import set_global_seed

        set_global_seed(self.cfg.random_state)

        # 1) Load + split (mirrors orchestration.runner)
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

        # 2) Partition
        partitioner = _build_partitioner(self.cfg)
        shards = partitioner.partition(
            X_train,
            y_train,
            n_clients=self.cfg.partition.n_clients,
            seed=self.cfg.partition.seed,
        )

        # 3) Build clients + coordinator (Hook B-aware)
        clients = build_clients(self.cfg, shards)
        coordinator = build_coordinator(
            self.cfg,
            classes=classes,
            n_features=X_train.shape[1],
            feature_ref=X_train,
        )

        # 4) Round loop
        round_results: list[RoundResult] = []
        last_global_model = None
        for t in range(self.cfg.n_rounds):
            with Timer() as timer:
                updates = [c.local_train() for c in clients]
                artifact = coordinator.round(
                    updates,
                    round_idx=t,
                    budget=PrivacyBudget(
                        epsilon=self.cfg.dp.target_epsilon,
                        delta=self.cfg.dp.target_delta,
                    ),
                )
                for c in clients:
                    c.apply_global(artifact)

            # 5) Global metrics
            global_model = deserialize_adapter(
                artifact.global_surrogate_blob, artifact.global_surrogate_class_path
            )
            last_global_model = global_model
            global_pred = global_model.predict(X_test)
            global_proba = global_model.predict_proba(X_test)
            global_metrics = classification_metrics(y_test, global_pred, global_proba)
            cum_eps, cum_delta = coordinator.accountant.cumulative()

            # 6) Per-client metrics (the bit the bare runner skips)
            client_metrics: list[ClientRoundMetrics] = []
            if self.compute_per_client:
                for c, upd in zip(clients, updates, strict=True):
                    local = c.evaluate_local(X_test, y_test)
                    fid = 0.0
                    if self.compute_fidelity and c.teacher is not None:
                        teacher_proba = c.teacher.predict_proba(coordinator.proxy_X)
                        surrogate = deserialize_adapter(
                            upd.surrogate_blob, upd.surrogate_class_path
                        )
                        student_proba = surrogate.predict_proba(coordinator.proxy_X)
                        fid_dict = surrogate_fidelity(teacher_proba, student_proba)
                        fid = float(fid_dict["argmax_agreement"])
                    client_metrics.append(
                        ClientRoundMetrics(
                            client_id=ClientID(int(c.client_id)),
                            local_accuracy=float(local.get("accuracy", float("nan"))),
                            local_f1_macro=float(local.get("f1_macro", float("nan"))),
                            local_auc=local.get("auc"),
                            surrogate_fidelity=fid,
                            bytes_uploaded=len(upd.surrogate_blob),
                            eps_spent=float(upd.eps_spent_this_round),
                            framework=upd.framework,
                        )
                    )

            round_results.append(
                RoundResult(
                    round_idx=t,
                    global_accuracy=global_metrics["accuracy"],
                    global_f1_macro=global_metrics["f1_macro"],
                    global_auc=global_metrics.get("auc"),
                    per_client_metrics=client_metrics,
                    total_bytes_communicated=sum(len(u.surrogate_blob) for u in updates),
                    cumulative_epsilon=cum_eps,
                    cumulative_delta=cum_delta,
                    wall_clock_seconds=timer.elapsed,
                )
            )

        # 7) Optional MIA on the final global surrogate
        mia_result: MIAResult | None = None
        if self.compute_mia and last_global_model is not None:
            mia_result = yeom_membership_inference(
                model=last_global_model,
                members_X=X_train,
                members_y=y_train,
                nonmembers_X=X_test,
                nonmembers_y=y_test,
                n_members=self.mia_n_members,
                n_nonmembers=self.mia_n_nonmembers,
                seed=self.cfg.random_state,
            )

        return EvaluationResult(
            config_name=self.cfg.name,
            dataset=self.cfg.data.name,
            dp_mechanism=self.cfg.dp.mechanism,
            epsilon_per_round=float(self.cfg.dp.epsilon_per_round or 0.0),
            n_clients=self.cfg.partition.n_clients,
            n_rounds=self.cfg.n_rounds,
            round_results=round_results,
            mia=mia_result,
            extra={
                "target_epsilon": self.cfg.dp.target_epsilon,
                "target_delta": self.cfg.dp.target_delta,
                "accountant": self.cfg.dp.accountant,
                "aggregator": self.cfg.aggregator.strategy,
                "partition_strategy": self.cfg.partition.strategy,
            },
        )
