from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def save_json(path: str | Path, obj: Any) -> None:
    """Save object as JSON, converting NumPy types to native Python."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, sort_keys=True)


def save_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Save list of dict rows to CSV with stable column ordering.

    Columns are the sorted union of all keys across rows.
    Missing keys are written as blank fields.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    columns = sorted({key for row in rows for key in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out_row = {k: row.get(k, "") for k in columns}
            writer.writerow(out_row)


def make_run_id(prefix: str, seed: int, config: Dict[str, Any] | None = None) -> str:
    """Create a deterministic run id from prefix, seed, and config."""

    config = config or {}
    payload = json.dumps(_to_jsonable(config), sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_seed{seed}_{digest}"
