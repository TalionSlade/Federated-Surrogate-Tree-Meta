"""Run a single experiment from a YAML config and print per-round summaries.

Usage:
    python experiments/scripts/run_experiment.py path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from fedrf_distill.config import ExperimentConfig
from fedrf_distill.orchestration import ExperimentRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to YAML config file.")
    parser.add_argument(
        "--json", type=Path, default=None, help="If set, dump per-round JSON here."
    )
    args = parser.parse_args(argv)

    cfg_dict = yaml.safe_load(args.config.read_text())
    cfg = ExperimentConfig.model_validate(cfg_dict)
    print(f"=== {cfg.name} ===")
    print(
        f"dataset={cfg.data.name}  clients={cfg.partition.n_clients}  "
        f"DP={cfg.dp.mechanism}(ε={cfg.dp.epsilon_per_round}/round)  "
        f"agg={cfg.aggregator.strategy}"
    )

    runner = ExperimentRunner(cfg)
    results = runner.run()
    for r in results:
        print(r.summary())

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [_asdict_round(r) for r in results]
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.json}")
    return 0


def _asdict_round(r: object) -> dict:
    """Recursively convert a RoundResult dataclass to a JSON-safe dict."""
    d = asdict(r)
    # Convert non-JSONable fields if any
    return d


if __name__ == "__main__":
    sys.exit(main())
