from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np

from portfolio_rl.data.sliced_source import SlicedDataSource
from portfolio_rl.experiments.run import run_experiment
from portfolio_rl.eval.bootstrap import bootstrap_ci
from portfolio_rl.experiments.reporting import save_json, save_csv


def _resolve_agent(agent_factory, seed: int, train_env) -> Any:
    result = agent_factory(seed)
    train_fn = None
    agent = result

    if isinstance(result, tuple) and len(result) == 2:
        agent, train_fn = result
    elif isinstance(result, dict) and "agent" in result:
        agent = result["agent"]
        train_fn = result.get("train_fn")

    if train_fn is not None:
        trained = train_fn(train_env, seed)
        if trained is not None:
            agent = trained
    elif hasattr(agent, "train") and callable(getattr(agent, "train")):
        trained = agent.train(train_env, seed)
        if trained is not None:
            agent = trained

    if agent is None:
        raise RuntimeError("agent_factory did not provide a usable agent")

    return agent


def run_walk_forward(
    data_source,
    make_env_fn: Callable[[Any], Any],
    agent_factory: Callable[[int], Any],
    seed: int,
    splits: List[Tuple[slice, slice]],
    max_steps: Optional[int] = None,
    out_dir: str | None = None,
    run_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run walk-forward evaluation over provided splits."""

    per_split: List[Dict[str, Any]] = []

    for i, (train_slice, test_slice) in enumerate(splits):
        train_ds = SlicedDataSource(base=data_source, slc=train_slice)
        test_ds = SlicedDataSource(base=data_source, slc=test_slice)

        train_env = make_env_fn(train_ds)
        test_env = make_env_fn(test_ds)

        agent = _resolve_agent(agent_factory, seed + i, train_env)
        result = run_experiment(test_env, agent, seed=seed + i, max_steps=max_steps)

        per_split.append(
            {
                "train_slice": train_slice,
                "test_slice": test_slice,
                "result": result,
            }
        )

    metric_keys = []
    if per_split:
        metric_keys = list(per_split[0]["result"]["metrics"].keys())

    metrics_mean = {}
    metrics_std = {}
    fields_mean = {}
    fields_std = {}

    if per_split:
        metric_values = {k: [] for k in metric_keys}
        field_keys = [
            "mean_transaction_cost",
            "mean_turnover",
            "mean_gross_exposure",
            "mean_net_exposure",
            "mean_weights_sum",
            "final_cash",
            "final_portfolio_value",
        ]
        field_values = {k: [] for k in field_keys}

        for split in per_split:
            res = split["result"]
            for k in metric_keys:
                metric_values[k].append(res["metrics"][k])
            for k in field_keys:
                field_values[k].append(res[k])

        metrics_mean = {k: float(np.mean(metric_values[k])) for k in metric_keys}
        metrics_std = {k: float(np.std(metric_values[k], ddof=0)) for k in metric_keys}
        fields_mean = {k: float(np.mean(field_values[k])) for k in field_keys}
        fields_std = {k: float(np.std(field_values[k], ddof=0)) for k in field_keys}

        sharpe_vals = np.asarray(metric_values.get("sharpe_ratio", []), dtype=float)
        cum_vals = np.asarray(metric_values.get("cumulative_return", []), dtype=float)
        if sharpe_vals.size > 0:
            metrics_mean["mean_sharpe"] = float(np.mean(sharpe_vals))
            metrics_mean["ci_sharpe"] = bootstrap_ci(sharpe_vals, np.mean)
        if cum_vals.size > 0:
            metrics_mean["mean_cumulative_return"] = float(np.mean(cum_vals))
            metrics_mean["ci_cumulative_return"] = bootstrap_ci(cum_vals, np.mean)

    result = {
        "splits": per_split,
        "summary": {
            "metrics_mean": metrics_mean,
            "metrics_std": metrics_std,
            "fields_mean": fields_mean,
            "fields_std": fields_std,
        },
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        split_params = {}
        if splits:
            train_size = splits[0][0].stop - splits[0][0].start
            test_size = splits[0][1].stop - splits[0][1].start
            embargo = splits[0][1].start - splits[0][0].stop
            step_size = None
            if len(splits) > 1:
                step_size = splits[1][0].start - splits[0][0].start
            split_params = {
                "train_size": int(train_size),
                "test_size": int(test_size),
                "step_size": int(step_size) if step_size is not None else None,
                "embargo": int(embargo),
            }

        config = {
            "seed": int(seed),
            "n_splits": len(splits),
            "split_params": split_params,
            "metadata": run_metadata or {},
        }

        save_json(out_path / "summary.json", {"summary": result["summary"], "metadata": run_metadata or {}})

        rows = []
        for split in per_split:
            res = split["result"]
            row = {
                "train_start": split["train_slice"].start,
                "train_stop": split["train_slice"].stop,
                "test_start": split["test_slice"].start,
                "test_stop": split["test_slice"].stop,
            }
            row.update(res["metrics"])
            for key in [
                "mean_transaction_cost",
                "mean_turnover",
                "mean_gross_exposure",
                "mean_net_exposure",
                "mean_weights_sum",
                "final_cash",
                "final_portfolio_value",
            ]:
                row[key] = res.get(key)
            rows.append(row)

        save_csv(out_path / "splits.csv", rows)
        save_json(out_path / "config.json", config)

    return result
