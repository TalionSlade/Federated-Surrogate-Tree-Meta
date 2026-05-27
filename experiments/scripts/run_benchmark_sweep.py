"""Run the full FedRF-Distill benchmark across all 10 global datasets.

Sweeps:
* dataset  ∈ {adult, bank_marketing, breast_cancer, covertype,
              credit_card_fraud, diabetes, higgs, letter_recognition,
              mushroom, nsl_kdd}
* DP       ∈ {null, laplace(ε=1), laplace(ε=4), gaussian(ε=1, δ=1e-5)}
* partition ∈ {iid, dirichlet(0.5), dirichlet(0.1), pathological(k=2)}

Outputs ``experiments/results/benchmark_sweep.csv`` with one row per
(dataset, DP, partition, round) tuple.

Usage:
    python experiments/scripts/run_benchmark_sweep.py
    python experiments/scripts/run_benchmark_sweep.py --datasets adult diabetes
    python experiments/scripts/run_benchmark_sweep.py --quick   # 2 rounds, 3 clients
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import traceback
from pathlib import Path

from fedrf_distill.config import ExperimentConfig
from fedrf_distill.orchestration import ExperimentRunner

ALL_DATASETS = [
    "adult",
    "bank_marketing",
    "breast_cancer",
    "covertype",
    "credit_card_fraud",
    "diabetes",
    "higgs",
    "letter_recognition",
    "mushroom",
    "nsl_kdd",
]

DP_VARIANTS = [
    ("null", "null", 100.0, 0.0),
    ("laplace_eps1", "laplace", 1.0, 0.0),
    ("laplace_eps4", "laplace", 4.0, 0.0),
    ("gaussian_eps1", "gaussian", 1.0, 1e-5),
]

PARTITION_VARIANTS = [
    ("iid", {"strategy": "iid"}),
    ("dirichlet_0.5", {"strategy": "dirichlet", "alpha": 0.5}),
    ("dirichlet_0.1", {"strategy": "dirichlet", "alpha": 0.1}),
    ("pathological_2", {"strategy": "pathological", "k_classes_per_client": 2}),
]


def build_cfg(
    dataset: str,
    dp_label: str,
    dp_mech: str,
    eps_round: float,
    delta_round: float,
    partition_label: str,
    partition_kwargs: dict,
    n_rounds: int,
    n_clients: int,
) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": f"{dataset}__{dp_label}__{partition_label}",
            "n_rounds": n_rounds,
            "random_state": 42,
            "data": {"name": dataset, "test_fraction": 0.2, "random_state": 0},
            "partition": {**partition_kwargs, "n_clients": n_clients, "seed": 0},
            "proxy": {"source": "synthetic_uniform", "n_samples": 1500, "seed": 0},
            "model": {
                "teacher_framework": "sklearn_rf",
                "teacher_kwargs": {"n_estimators": 80, "max_depth": 10},
                "student_framework": "sklearn_dt",
                "student_kwargs": {"max_depth": 6},
            },
            "distillation": {"mode": "argmax_weighted"},
            "dp": {
                "mechanism": dp_mech,
                "epsilon_per_round": eps_round,
                "delta_per_round": delta_round,
                "target_epsilon": 100.0,
                "target_delta": 1e-3,
                "accountant": "rdp" if dp_mech == "gaussian" else "basic",
            },
            "aggregator": {"strategy": "proxy_redistillation"},
            "meta": {"enabled": True, "strategy": "stacking"},
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/benchmark_sweep.csv"),
    )
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--n-clients", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="2 rounds, 3 clients")
    args = parser.parse_args(argv)

    if args.quick:
        args.n_rounds = 2
        args.n_clients = 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "dp_variant",
        "partition_variant",
        "round_idx",
        "accuracy",
        "f1_macro",
        "auc",
        "bytes_communicated",
        "cumulative_epsilon",
        "wall_clock_seconds",
    ]
    rows: list[dict] = []

    for dataset in args.datasets:
        for (dp_label, dp_mech, eps_round, delta_round), (
            part_label,
            part_kw,
        ) in itertools.product(DP_VARIANTS, PARTITION_VARIANTS):
            print(f"▶ {dataset} | {dp_label} | {part_label}")
            try:
                cfg = build_cfg(
                    dataset=dataset,
                    dp_label=dp_label,
                    dp_mech=dp_mech,
                    eps_round=eps_round,
                    delta_round=delta_round,
                    partition_label=part_label,
                    partition_kwargs=part_kw,
                    n_rounds=args.n_rounds,
                    n_clients=args.n_clients,
                )
                results = ExperimentRunner(cfg).run()
                for r in results:
                    rows.append(
                        {
                            "dataset": dataset,
                            "dp_variant": dp_label,
                            "partition_variant": part_label,
                            "round_idx": r.round_idx,
                            "accuracy": r.global_accuracy,
                            "f1_macro": r.global_f1_macro,
                            "auc": r.global_auc if r.global_auc is not None else "",
                            "bytes_communicated": r.total_bytes_communicated,
                            "cumulative_epsilon": r.cumulative_epsilon,
                            "wall_clock_seconds": r.wall_clock_seconds,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - log and continue sweep
                print(f"  ✗ FAILED: {exc}")
                traceback.print_exc()

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
