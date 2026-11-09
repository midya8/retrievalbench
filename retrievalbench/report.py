"""Terminal, JSON and Markdown reporting for evaluation runs."""

from __future__ import annotations

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "reciprocal_rank": "MRR",
    "ndcg": "nDCG",
}


def _metric_label(name: str) -> str:
    for prefix, label in METRIC_LABELS.items():
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix).strip("_")
            return f"{label}@{suffix.removeprefix('at_')}" if suffix else label
    return name


def _md_cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(result: dict, *, corpus_size: int) -> str:
    """Render an evaluation result as a Markdown report for CI summaries."""
    averages = result["averages"]
    k = result["k"]
    lines = [
        "# RetrievalBench evaluation",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Corpus documents | {corpus_size} |",
        f"| Judged queries | {result['query_count']} |",
        f"| Cutoff k | {k} |",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name in sorted(averages):
        lines.append(f"| {_metric_label(name)} | {averages[name]:.4f} |")
    lines.extend(["", "## Per-query results", ""])
    lines.append("| Query | " + " | ".join(_metric_label(name) for name in sorted(averages)) + " | Relevant | Retrieved |")
    lines.append("|" + "---|" * (len(averages)+3))
    for row in result["queries"]:
        cells = " | ".join(f"{row[name]:.4f}" for name in sorted(averages))
        lines.append(f"| {_md_cell(row['query_id'])} | {cells} | {row['relevant_count']} | {row['retrieved_count']} |")
    lines.append("")
    return "\n".join(lines)


def render_terminal(result: dict, *, corpus_size: int) -> str:
    """Render a compact terminal summary."""
    lines = [
        f"corpus {corpus_size} document(s), {result['query_count']} judged query(ies), k={result['k']}",
    ]
    for name in sorted(result["averages"]):
        lines.append(f"{_metric_label(name):12s} {result['averages'][name]:.4f}")
    weakest = min(result["queries"], key=lambda row: row["reciprocal_rank"])
    lines.append(f"weakest query {weakest['query_id']} (MRR {weakest['reciprocal_rank']:.4f})")
    return "\n".join(lines)
