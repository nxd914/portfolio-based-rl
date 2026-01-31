from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any


def collect_runs(out_dir: str) -> List[Dict[str, Any]]:
    """Collect run records from manifest and/or summary.json files."""

    out_path = Path(out_dir)
    runs: List[Dict[str, Any]] = []

    manifest_path = out_path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            runs.extend(data)

    # Fallback: scan for summary.json files
    for summary in out_path.rglob("summary.json"):
        try:
            with summary.open("r", encoding="utf-8") as f:
                data = json.load(f)
            summary_data = data.get("summary", {})
            run_dir = summary.parent
            record = {
                "agent_name": run_dir.parent.name,
                "run_id": run_dir.name,
                "path": str(run_dir),
                "metrics_mean": summary_data.get("metrics_mean", {}),
                "fields_mean": summary_data.get("fields_mean", {}),
            }
            runs.append(record)
        except Exception:
            continue

    return runs


def to_plot_data(runs: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """Convert run records to plotting arrays.

    Returns dict with x (agent_name list), y (metric values), and ci (tuples or None).
    """

    x = []
    y = []
    ci = []

    for run in runs:
        name = run.get("agent_name")
        metrics = run.get("metrics_mean", {})
        fields = run.get("fields_mean", {})

        value = None
        if metric in metrics:
            value = metrics.get(metric)
        elif metric in fields:
            value = fields.get(metric)

        if value is None:
            continue

        x.append(name)
        y.append(value)

        ci_key = None
        if metric == "cumulative_return":
            ci_key = "ci_cumulative_return"
        elif metric == "sharpe_ratio":
            ci_key = "ci_sharpe"

        if ci_key and ci_key in metrics:
            ci.append(metrics.get(ci_key))
        else:
            ci.append(None)

    return {"x": x, "y": y, "ci": ci}
