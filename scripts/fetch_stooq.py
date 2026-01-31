from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np


def fetch_csv(symbol: str, start: str, end: str) -> List[Dict[str, str]]:
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start}&d2={end}&i=d"
    with urllib.request.urlopen(url) as resp:
        data = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(data))
    rows = [row for row in reader]
    if not rows:
        raise RuntimeError(f"No data returned for {symbol}")
    return rows


def align_rows(rows_by_symbol: Dict[str, List[Dict[str, str]]]):
    dates_sets = []
    for rows in rows_by_symbol.values():
        dates_sets.append({r["Date"] for r in rows})
    common_dates = set.intersection(*dates_sets)
    if not common_dates:
        raise RuntimeError("No common dates across symbols")

    dates = sorted(common_dates)

    aligned = {}
    for symbol, rows in rows_by_symbol.items():
        row_map = {r["Date"]: r for r in rows}
        aligned[symbol] = [row_map[d] for d in dates]

    return dates, aligned


def to_arrays(dates, aligned, symbols):
    n_steps = len(dates)
    n_assets = len(symbols)
    def arr(field):
        out = np.zeros((n_steps, n_assets), dtype=float)
        for j, sym in enumerate(symbols):
            for i, row in enumerate(aligned[sym]):
                out[i, j] = float(row[field])
        return out

    open_ = arr("Open")
    high = arr("High")
    low = arr("Low")
    close = arr("Close")
    volume = arr("Volume")
    return open_, high, low, close, volume


def main():
    symbols = ["spy.us", "qqq.us"]
    start = "20150101"
    end = "20241231"

    rows_by_symbol = {s: fetch_csv(s, start, end) for s in symbols}
    dates, aligned = align_rows(rows_by_symbol)
    open_, high, low, close, volume = to_arrays(dates, aligned, symbols)

    out_dir = Path("data/real")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "ohlcv.npz",
        dates=np.array(dates, dtype="U10"),
        symbols=np.array(symbols, dtype="U10"),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    print(f"Saved aligned data to {out_dir / 'ohlcv.npz'}")


if __name__ == "__main__":
    main()
