"""Run the FedRF-Distill evaluation pipeline over one or many YAML configs.

Examples
--------
# One config:
python experiments/scripts/run_eval.py experiments/configs/breast_cancer_nodp.yaml

# Multiple explicit configs:
python experiments/scripts/run_eval.py \
    experiments/configs/breast_cancer_nodp.yaml \
    experiments/configs/breast_cancer_dp_laplace_eps5.yaml \
    experiments/configs/breast_cancer_dp_laplace.yaml

# Every YAML in a directory:
python experiments/scripts/run_eval.py --config-dir experiments/configs

# Skip the (slow) per-client + MIA passes for a quick smoke run:
python experiments/scripts/run_eval.py --no-mia --no-per-client \
    experiments/configs/breast_cancer_nodp.yaml

Outputs (default ``experiments/results/eval/``)
-----------------------------------------------
* ``<config>.eval.json`` — full :class:`EvaluationResult` per config.
* ``eval_report.md`` — combined privacy-utility frontier + per-round +
  MIA tables across every config that ran successfully.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

from fedrf_distill.config import ExperimentConfig
from fedrf_distill.eval import (
    EvaluationPipeline,
    render_markdown_report,
)


def _gather_configs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = list(args.configs or [])
    if args.config_dir:
        paths.extend(sorted(args.config_dir.glob("*.yaml")))
        paths.extend(sorted(args.config_dir.glob("*.yml")))
    if not paths:
        raise SystemExit(
            "No configs provided. Pass paths positionally or use --config-dir."
        )
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="One or more YAML config paths.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing YAML configs (loads every *.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/results/eval"),
        help="Where to write per-config JSON and the combined markdown report.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="eval_report.md",
        help="Filename for the combined markdown report inside --out-dir.",
    )
    parser.add_argument("--no-per-client", action="store_true", help="Skip per-client metrics.")
    parser.add_argument("--no-fidelity", action="store_true", help="Skip surrogate fidelity.")
    parser.add_argument("--no-mia", action="store_true", help="Skip membership inference attack.")
    parser.add_argument(
        "--mia-n-members",
        type=int,
        default=500,
        help="Sub-sample size of members for MIA (default 500).",
    )
    parser.add_argument(
        "--mia-n-nonmembers",
        type=int,
        default=500,
        help="Sub-sample size of non-members for MIA (default 500).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-round prints.")
    args = parser.parse_args(argv)

    configs = _gather_configs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict] = []
    failed: list[tuple[Path, str]] = []

    for cfg_path in configs:
        print(f"▶ {cfg_path}")
        try:
            cfg_dict = yaml.safe_load(cfg_path.read_text())
            cfg = ExperimentConfig.model_validate(cfg_dict)
            pipeline = EvaluationPipeline(
                cfg=cfg,
                compute_per_client=not args.no_per_client,
                compute_fidelity=not args.no_fidelity,
                compute_mia=not args.no_mia,
                mia_n_members=args.mia_n_members,
                mia_n_nonmembers=args.mia_n_nonmembers,
            )
            result = pipeline.run()
            if not args.quiet:
                for r in result.round_results:
                    print("  " + r.summary())
                if result.mia is not None:
                    print(
                        f"  MIA: AUC={result.mia.attack_auc:.4f} "
                        f"adv={result.mia.attack_advantage:.4f} "
                        f"TPR@1%FPR={result.mia.tpr_at_1_fpr:.4f}"
                    )
            out_json = args.out_dir / f"{cfg.name}.eval.json"
            result.to_json(out_json)
            payloads.append(result.to_dict())
            print(f"  ✓ Wrote {out_json}")
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"  ✗ FAILED: {exc}")
            traceback.print_exc()
            failed.append((cfg_path, str(exc)))

    if payloads:
        report = render_markdown_report(payloads)
        report_path = args.out_dir / args.report_name
        report_path.write_text(report)
        print(f"\n✓ Combined report → {report_path}")

    if failed:
        print(f"\n⚠ {len(failed)} config(s) failed:")
        for path, msg in failed:
            print(f"  - {path}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
