"""Multi-run aggregation utilities.

Load N :class:`EvaluationResult` JSON artifacts from a sweep, then collapse
them into:

* :class:`EvalSummary` — flat summary record per (config, final round).
* :func:`privacy_utility_table` — DataFrame-like list of dicts ordered by
  cumulative ε, useful for the Pareto frontier in the paper.

We deliberately avoid hard-depending on pandas so the eval module stays import-
light. If the caller wants a DataFrame, ``pd.DataFrame(privacy_utility_table(...))``
does it in one line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvalSummary:
    """One-row summary of an :class:`EvaluationResult`."""

    config_name: str
    dataset: str
    dp_mechanism: str
    epsilon_per_round: float
    n_clients: int
    n_rounds: int
    final_accuracy: float
    final_f1_macro: float
    final_auc: float | None
    final_cumulative_epsilon: float
    final_cumulative_delta: float
    total_bytes: int
    total_wall_clock_seconds: float
    mia_attack_auc: float | None
    mia_attack_advantage: float | None
    mia_tpr_at_1_fpr: float | None


def load_results(paths: list[Path] | Path) -> list[dict[str, Any]]:
    """Load one or more eval JSON artifacts.

    Accepts a single path, a list of paths, or a directory (in which case
    every ``*.json`` file directly under it is loaded). Returns the raw
    dicts; callers can construct :class:`EvalSummary` via :func:`summarise`.
    """
    if isinstance(paths, Path):
        if paths.is_dir():
            paths = sorted(paths.glob("*.json"))
        else:
            paths = [paths]

    results: list[dict[str, Any]] = []
    for p in paths:
        with p.open() as fh:
            payload = json.load(fh)
        # Defensive: tag with source path for debugging.
        payload.setdefault("_source_path", str(p))
        results.append(payload)
    return results


def summarise(payload: dict[str, Any]) -> EvalSummary:
    """Collapse one :class:`EvaluationResult` dict to a flat row."""
    rounds = payload.get("round_results") or []
    last = rounds[-1] if rounds else {}
    mia = payload.get("mia") or {}
    return EvalSummary(
        config_name=payload.get("config_name", ""),
        dataset=payload.get("dataset", ""),
        dp_mechanism=payload.get("dp_mechanism", ""),
        epsilon_per_round=float(payload.get("epsilon_per_round") or 0.0),
        n_clients=int(payload.get("n_clients") or 0),
        n_rounds=int(payload.get("n_rounds") or 0),
        final_accuracy=float(last.get("global_accuracy") or 0.0),
        final_f1_macro=float(last.get("global_f1_macro") or 0.0),
        final_auc=last.get("global_auc"),
        final_cumulative_epsilon=float(last.get("cumulative_epsilon") or 0.0),
        final_cumulative_delta=float(last.get("cumulative_delta") or 0.0),
        total_bytes=int(sum(r.get("total_bytes_communicated", 0) for r in rounds)),
        total_wall_clock_seconds=float(sum(r.get("wall_clock_seconds", 0.0) for r in rounds)),
        mia_attack_auc=mia.get("attack_auc"),
        mia_attack_advantage=mia.get("attack_advantage"),
        mia_tpr_at_1_fpr=mia.get("tpr_at_1_fpr"),
    )


def privacy_utility_table(
    results: list[dict[str, Any]] | None = None,
    *,
    paths: list[Path] | Path | None = None,
) -> list[dict[str, Any]]:
    """Build a privacy-utility frontier table (one row per config).

    Pass either pre-loaded ``results`` (list of payload dicts) or ``paths``
    (a directory / list of files) — never both.

    Returns rows sorted by ``final_cumulative_epsilon`` ascending so the
    no-DP baseline lands at the top and high-ε runs at the bottom. Suitable
    for ``pandas.DataFrame(...)`` or direct markdown rendering.
    """
    if results is None and paths is None:
        raise ValueError("Pass either `results` or `paths`.")
    if results is not None and paths is not None:
        raise ValueError("Pass `results` or `paths`, not both.")

    payloads = results if results is not None else load_results(paths)  # type: ignore[arg-type]
    rows = [asdict(summarise(p)) for p in payloads]
    rows.sort(key=lambda r: (r["dataset"], r["final_cumulative_epsilon"]))
    return rows


def per_round_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tall table: one row per (config, round) — useful for line plots."""
    out: list[dict[str, Any]] = []
    for p in results:
        for r in p.get("round_results") or []:
            out.append(
                {
                    "config_name": p.get("config_name", ""),
                    "dataset": p.get("dataset", ""),
                    "dp_mechanism": p.get("dp_mechanism", ""),
                    "epsilon_per_round": float(p.get("epsilon_per_round") or 0.0),
                    "round_idx": r.get("round_idx"),
                    "global_accuracy": r.get("global_accuracy"),
                    "global_f1_macro": r.get("global_f1_macro"),
                    "global_auc": r.get("global_auc"),
                    "cumulative_epsilon": r.get("cumulative_epsilon"),
                    "total_bytes_communicated": r.get("total_bytes_communicated"),
                    "wall_clock_seconds": r.get("wall_clock_seconds"),
                }
            )
    return out


def per_client_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tall table: one row per (config, round, client) — diagnostic only."""
    out: list[dict[str, Any]] = []
    for p in results:
        for r in p.get("round_results") or []:
            for cm in r.get("per_client_metrics") or []:
                out.append(
                    {
                        "config_name": p.get("config_name", ""),
                        "dataset": p.get("dataset", ""),
                        "round_idx": r.get("round_idx"),
                        **cm,
                    }
                )
    return out
