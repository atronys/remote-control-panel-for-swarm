"""Командный интерфейс SwarmCore.

    swarmcore run     --scenario scenarios/baseline.yaml --allocator static_roundrobin --seed 1
    swarmcore compare --scenario scenarios/failures.yaml --allocators static_roundrobin,cbba --seeds 1-20
    swarmcore replay  runs/baseline_static_roundrobin_s1 --open     → интерактивный HTML
    swarmcore log     runs/baseline_static_roundrobin_s1 --agent 3  → хронология событий
    swarmcore matrix  --scenario scenarios/baseline.yaml --vary agents.count=5,10,20 --seeds 1-20 \
                      --plot agents.count:T_detect_90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apps.eventlog import add_log_args, log_command
from apps.replay import add_replay_args, replay_command
from experiments.run_matrix import add_matrix_args, aggregate, matrix_command, parse_seeds, run_matrix
from swarmsim.io import write_run
from swarmsim.scenario import apply_overrides, load_scenario, parse_override
from swarmsim.sim import run

KEY_METRICS = ("t_final", "T_detect_50", "T_detect_90", "detected_frac", "CR", "tasks_orphaned",
               "E_total_Wh", "N_col", "N_nm", "N_geo", "N_dup", "agents_failed", "valid",
               "realtime_factor")


def cmd_run(args) -> int:
    overrides = dict(parse_override(s) for s in args.set)
    if args.allocator:
        overrides["allocator.kind"] = args.allocator
    scn = apply_overrides(load_scenario(args.scenario), overrides)
    if args.seed is None:                       # без --seed каждый прогон свой; зерно печатаем для повтора
        import secrets
        args.seed = secrets.randbelow(1_000_000)
    print(f"зерно: {args.seed}   (повторить этот прогон: --seed {args.seed})")
    res = run(scn, args.seed)
    out = Path(args.out) if args.out else Path("runs") / f"{scn.name}_{res.summary['allocator']}_s{args.seed}"
    write_run(res, out)
    for k in KEY_METRICS:
        print(f"  {k:<16} {res.summary.get(k)}")
    print(f"результаты: {out}")
    if args.plot:
        from apps.viz import plot_run
        png = plot_run(out)
        print(f"картинка: {png}")
    if args.replay:
        from apps.replay import make_replay
        print(f"проигрыватель: {make_replay(out)}")
    return 0 if res.summary["valid"] else 2


def cmd_compare(args) -> int:
    allocators = args.allocators.split(",")
    rows = run_matrix(args.scenario, parse_seeds(args.seeds), allocators=allocators, jobs=args.jobs)
    print(f"{'allocator':<22}{'metric':<16}{'mean':>12}{'±95%':>10}{'valid/runs':>12}")
    for metric in args.metrics.split(","):
        for a in aggregate(rows, ["allocator"], metric):
            mean = "—" if a["mean"] is None else f"{a['mean']:.3f}"
            ci = "—" if a["ci95"] is None else f"{a['ci95']:.3f}"
            print(f"{a['allocator']:<22}{metric:<16}{mean:>12}{ci:>10}{a['n_valid']:>6}/{a['n_runs']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")    # консоль Windows (cp1251)
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="swarmcore")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="один прогон → CSV/JSON в папку")
    r.add_argument("--scenario", required=True)
    r.add_argument("--allocator", default=None)
    r.add_argument("--seed", type=int, default=None, help="зерно; без него — случайное (печатается)")
    r.add_argument("--set", action="append", default=[], help="ключ=значение, напр. agents.count=20")
    r.add_argument("--out", default=None)
    r.add_argument("--plot", action="store_true", help="сохранить картинку траекторий (matplotlib)")
    r.add_argument("--replay", action="store_true", help="собрать интерактивный replay.html")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="интерактивный HTML-проигрыватель прогона")
    add_replay_args(rp)
    rp.set_defaults(func=replay_command)

    lg = sub.add_parser("log", help="человекочитаемая хронология events.jsonl")
    add_log_args(lg)
    lg.set_defaults(func=log_command)

    c = sub.add_parser("compare", help="сравнить распределители на одних и тех же зёрнах")
    c.add_argument("--scenario", required=True)
    c.add_argument("--allocators", required=True)
    c.add_argument("--seeds", default="1-20")
    c.add_argument("--metrics", default="CR,T_detect_90,E_total_Wh")
    c.add_argument("--jobs", type=int, default=1)
    c.set_defaults(func=cmd_compare)

    sv = sub.add_parser("serve", help="страница в браузере: новые прогоны кнопкой + проигрыватель")
    from apps.serve import add_serve_args, serve_command
    add_serve_args(sv)
    sv.set_defaults(func=serve_command)

    m = sub.add_parser("matrix", help="матрица прогонов → results.csv (+ график)")
    add_matrix_args(m)
    m.set_defaults(func=matrix_command)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
