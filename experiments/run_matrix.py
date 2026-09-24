"""Испытательный стенд прогонов: матрица (варианты конфига × распределители × зёрна) → CSV + manifest.

Парный дизайн (docs/03): для каждого зерна мир одинаков во всех ячейках матрицы.

    python -m experiments.run_matrix --scenario scenarios/baseline.yaml --seeds 1-20 \
        --vary agents.count=5,10,20 --allocators static_roundrobin --plot agents.count:T_detect_90
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from swarmsim.io import write_csv
from swarmsim.scenario import apply_overrides, load_scenario, parse_override
from swarmsim.sim import run


def parse_seeds(text: str) -> list[int]:
    """'1-20' | '1,2,5' | '1-5,10'."""
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_vary(items: list[str]) -> dict[str, list[Any]]:
    """['agents.count=5,10,20'] -> {'agents.count': [5, 10, 20]}."""
    out = {}
    for item in items:
        key, raw = item.split("=", 1)
        out[key.strip()] = [yaml.safe_load(v) for v in raw.split(",")]
    return out


def _one(job: tuple[str, dict, int]) -> dict:
    path, overrides, seed = job
    scn = apply_overrides(load_scenario(path), overrides)
    summary = run(scn, seed).summary
    return {**{f"cfg:{k}": v for k, v in overrides.items()}, **summary}


def run_matrix(scenario_path: str, seeds: list[int], vary: dict[str, list[Any]] | None = None,
               allocators: list[str] | None = None, fixed: dict[str, Any] | None = None,
               jobs: int = 1) -> list[dict]:
    vary = dict(vary or {})
    if allocators:
        vary["allocator.kind"] = allocators
    keys = list(vary)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*vary.values())] or [{}]
    job_list = [(scenario_path, {**(fixed or {}), **c}, s) for c in combos for s in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_one, job_list))
    return [_one(j) for j in job_list]


def aggregate(rows: list[dict], group_by: list[str], metric: str) -> list[dict]:
    """Среднее и 95 % ДИ (нормальное приближение) по валидным прогонам."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(g) for g in group_by), []).append(r)
    out = []
    for key, rs in groups.items():
        vals = [r[metric] for r in rs if r.get("valid") and isinstance(r.get(metric), (int, float))]
        n = len(vals)
        mean = statistics.fmean(vals) if n else None
        ci = 1.96 * statistics.stdev(vals) / math.sqrt(n) if n > 1 else None
        out.append({**dict(zip(group_by, key)), "metric": metric, "n_valid": n,
                    "n_runs": len(rs), "mean": mean, "ci95": ci})
    return out


def plot(rows: list[dict], x: str, y: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xcol = x if x in rows[0] else f"cfg:{x}"
    series = "allocator"
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in sorted({r[series] for r in rows}):
        agg = aggregate([r for r in rows if r[series] == name], [xcol], y)
        agg = sorted((a for a in agg if a["mean"] is not None), key=lambda a: a[xcol])
        xs = [a[xcol] for a in agg]
        ax.errorbar(xs, [a["mean"] for a in agg], yerr=[a["ci95"] or 0 for a in agg],
                    marker="o", capsize=3, label=name)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"{y} vs {x} (среднее ± 95 % ДИ)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Матрица прогонов SwarmCore")
    add_matrix_args(p)
    return matrix_command(p.parse_args(argv))


def add_matrix_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scenario", required=True)
    p.add_argument("--seeds", default="1-20", help="например 1-20 или 1,2,3")
    p.add_argument("--vary", action="append", default=[], help="ключ=v1,v2,... (можно несколько)")
    p.add_argument("--set", action="append", default=[], help="ключ=значение, фиксированное")
    p.add_argument("--allocators", default=None, help="через запятую")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", default="runs/matrix")
    p.add_argument("--plot", default=None, help="x:y, например agents.count:T_detect_90")


def matrix_command(args) -> int:
    seeds = parse_seeds(args.seeds)
    vary = parse_vary(args.vary)
    fixed = dict(parse_override(s) for s in args.set)
    allocators = args.allocators.split(",") if args.allocators else None
    rows = run_matrix(args.scenario, seeds, vary, allocators, fixed, args.jobs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "results.csv", rows)
    manifest = {"scenario": args.scenario, "seeds": seeds, "vary": vary, "fixed": fixed,
                "allocators": allocators, "runs": len(rows),
                "config_hashes": sorted({r["config_hash"] for r in rows}),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    invalid = sum(not r["valid"] for r in rows)
    print(f"{len(rows)} прогонов: {out / 'results.csv'}  (invalid: {invalid})")
    if args.plot:
        x, y = args.plot.split(":")
        plot(rows, x, y, out / f"{y}_vs_{x}.png")
        print(f"график: {out / f'{y}_vs_{x}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
