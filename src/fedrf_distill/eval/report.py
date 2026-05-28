"""Paper-ready markdown report generator for evaluation runs.

Emits a single ``.md`` file with three sections:

1. **Privacy-Utility Frontier** — flat table sorted by cumulative ε. Shows
   which DP regime each config falls in and the corresponding utility hit.
2. **Per-Round Trajectories** — one mini-table per config showing how
   accuracy and ε grow round-by-round. Useful for the appendix.
3. **Membership Inference Results** — empirical privacy curve via Yeom MIA
   if it was computed during the run.

The markdown is GitHub-flavoured and renders cleanly in DX pages, Notion,
or any LaTeX-via-pandoc pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fedrf_distill.eval.aggregate import privacy_utility_table, summarise


def render_markdown_report(
    results: list[dict[str, Any]],
    *,
    title: str = "FedRF-Distill — Evaluation Report",
) -> str:
    """Render a self-contained markdown report.

    Parameters
    ----------
    results:
        List of :class:`EvaluationResult` dicts (as produced by
        :meth:`EvaluationPipeline.run().to_dict()` or
        :func:`fedrf_distill.eval.aggregate.load_results`).
    title:
        Heading for the report.
    """
    if not results:
        return f"# {title}\n\n_No results._\n"

    sections = [
        _render_header(results, title),
        _render_frontier_section(results),
        _render_per_round_section(results),
        _render_mia_section(results),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


# ── Sections ────────────────────────────────────────────────────────────────
def _render_header(results: list[dict[str, Any]], title: str) -> str:
    datasets = sorted({r.get("dataset", "?") for r in results})
    dp_set = sorted({r.get("dp_mechanism", "?") for r in results})
    return (
        f"# {title}\n\n"
        f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_\n\n"
        f"- **Configs**: {len(results)}\n"
        f"- **Datasets**: {', '.join(datasets)}\n"
        f"- **DP mechanisms**: {', '.join(dp_set)}\n"
    )


def _render_frontier_section(results: list[dict[str, Any]]) -> str:
    rows = privacy_utility_table(results)
    if not rows:
        return ""
    headers = [
        "Dataset",
        "Config",
        "DP",
        "ε/round",
        "Clients",
        "Final Acc",
        "Final F1",
        "Cum. ε",
        "Total Bytes",
        "Wall (s)",
    ]
    table_rows: list[list[str]] = []
    for r in rows:
        # ε/round is meaningless for the null mechanism — show an em-dash so
        # readers don't mistake the schema default for an applied budget.
        eps_per_round = "—" if r["dp_mechanism"] == "null" else _fmt(r["epsilon_per_round"])
        table_rows.append(
            [
                str(r["dataset"]),
                str(r["config_name"]),
                str(r["dp_mechanism"]),
                eps_per_round,
                str(r["n_clients"]),
                _fmt(r["final_accuracy"], 4),
                _fmt(r["final_f1_macro"], 4),
                _fmt(r["final_cumulative_epsilon"]),
                f"{r['total_bytes']:,}",
                _fmt(r["total_wall_clock_seconds"], 1),
            ]
        )
    return (
        "## Privacy-Utility Frontier\n\n"
        "_Rows sorted by cumulative ε (no-DP baselines first)._\n\n"
        + _markdown_table(headers, table_rows)
    )


def _render_per_round_section(results: list[dict[str, Any]]) -> str:
    parts = ["## Per-Round Trajectories\n"]
    for payload in sorted(results, key=lambda p: p.get("config_name", "")):
        cfg_name = payload.get("config_name", "?")
        rounds = payload.get("round_results") or []
        if not rounds:
            continue
        parts.append(f"### `{cfg_name}`\n")
        headers = ["t", "Acc", "F1", "AUC", "Bytes", "Cum. ε", "Wall (s)"]
        body: list[list[str]] = []
        for r in rounds:
            body.append(
                [
                    str(r.get("round_idx")),
                    _fmt(r.get("global_accuracy"), 4),
                    _fmt(r.get("global_f1_macro"), 4),
                    _fmt(r.get("global_auc"), 4),
                    f"{int(r.get('total_bytes_communicated') or 0):,}",
                    _fmt(r.get("cumulative_epsilon")),
                    _fmt(r.get("wall_clock_seconds"), 2),
                ]
            )
        parts.append(_markdown_table(headers, body))
        parts.append("")  # blank line between configs
    return "\n".join(parts).rstrip()


def _render_mia_section(results: list[dict[str, Any]]) -> str:
    mia_rows = [r for r in results if r.get("mia")]
    if not mia_rows:
        return ""
    headers = [
        "Config",
        "DP",
        "ε/round",
        "Cum. ε",
        "Attack AUC",
        "Advantage",
        "TPR@1%FPR",
        "Mem loss",
        "Non-mem loss",
    ]
    body: list[list[str]] = []
    for payload in mia_rows:
        mia = payload["mia"]
        summary = summarise(payload)
        eps_per_round = (
            "—" if payload.get("dp_mechanism") == "null"
            else _fmt(payload.get("epsilon_per_round"))
        )
        body.append(
            [
                str(payload.get("config_name", "?")),
                str(payload.get("dp_mechanism", "?")),
                eps_per_round,
                _fmt(summary.final_cumulative_epsilon),
                _fmt(mia.get("attack_auc"), 4),
                _fmt(mia.get("attack_advantage"), 4),
                _fmt(mia.get("tpr_at_1_fpr"), 4),
                _fmt(mia.get("member_mean_loss"), 4),
                _fmt(mia.get("nonmember_mean_loss"), 4),
            ]
        )
    return (
        "## Membership Inference Attack (Yeom)\n\n"
        "_AUC = 0.5 → empirical privacy preserved; ≥ 0.6 → measurable leakage._\n\n"
        + _markdown_table(headers, body)
    )


# ── Helpers ─────────────────────────────────────────────────────────────────
def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    head = "| " + " | ".join(headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a numeric value with ``digits`` decimal places; handle None / NaN."""
    if value is None:
        return "—"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return str(value)
    if fv != fv:  # NaN
        return "—"
    return f"{fv:.{digits}f}"
