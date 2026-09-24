"""Запись результатов прогона: summary.csv/json, timeseries.csv, trajectories.csv,
events.jsonl, run_manifest.json, world.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .sim import RunResult


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def write_run(result: RunResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", [result.summary])
    (out / "summary.json").write_text(json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "timeseries.csv", result.timeseries)
    if result.trajectories:
        write_csv(out / "trajectories.csv", result.trajectories)
    with (out / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in result.events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "run_manifest.json").write_text(json.dumps(result.manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    if result.world:
        (out / "world.json").write_text(json.dumps(result.world, ensure_ascii=False), encoding="utf-8")
    return out
