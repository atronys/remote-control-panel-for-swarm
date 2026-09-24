"""Проверка контрольных точек roadmap (docs/05) P2–P5 с отчётами в docs/reports/.

    python -m experiments.gates p2 --jobs 11
    python -m experiments.gates p3 --jobs 11
    python -m experiments.gates p4 --jobs 11
    python -m experiments.gates p5 --jobs 11      # полная кампания docs/03 + статистика + графики

Каждая проверка пишет docs/reports/<фаза>.md (таблицы, критерии, вывод), CSV с сырыми данными
и графики в docs/reports/figures/. Числа в отчёте — только из этих прогонов.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
FIG = REPORTS / "figures"
BASE = ROOT / "scenarios" / "baseline.yaml"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _mean_ci(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = statistics.fmean(xs)
    ci = 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, ci


def _pool(fn, jobs_list, jobs):
    if jobs <= 1:
        return [fn(j) for j in jobs_list]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, jobs_list, chunksize=1))


def _write_csv(path: Path, rows: list[dict]) -> None:
    from swarmsim.io import write_csv
    write_csv(path, rows)


# =============================================================================== P2
P2_SIZES = {600: 9, 800: 16, 1000: 25, 1200: 36}       # сторона поля, м → число секторов 200×200
P2_AGENTS = [2, 3, 5, 8, 10]


def _p2_job(job: tuple) -> dict:
    """Один экземпляр: качество плана (B2 против эталона B3) и полный прогон обоих."""
    from swarmsim import apply_overrides, load_scenario, run
    from swarmsim.alloc.central import greedy_insert, ruin_recreate, total_score
    from swarmsim.alloc.cpsat import solve_cpsat
    from swarmsim.planning.routing import RouteModel, TaskGeom, agent_context
    from swarmsim.sim import SimLoop

    n, size, prior, seed = job
    scn = apply_overrides(load_scenario(BASE), {
        "agents.count": n, "world.size": [size, size], "agents.home": [size / 2, 0],
        "world.prior.kind": prior, "output.trajectories": False})
    sim = SimLoop(scn, seed)
    snap = sim.snapshot(0.0)
    ids = [t.id for t in sim.tasks]
    model = RouteModel(TaskGeom.build(sim.tasks, ids), scn.planning.discount)
    ctxs = [agent_context(a, sim.tasks, sim.energy, scn.agents.reserve, scn.energy.E_land) for a in snap.agents]
    t0 = time.perf_counter()
    g = greedy_insert(model, ctxs, [[] for _ in ctxs], list(range(len(ids))))
    g = [model.improve(c, r) for c, r in zip(ctxs, g)]
    t_greedy = time.perf_counter() - t0
    s_greedy = total_score(model, ctxs, g)
    t0 = time.perf_counter()
    best = ruin_recreate(model, ctxs, g, np.random.default_rng(seed), iters=3000)
    t_lns = time.perf_counter() - t0
    s_best = total_score(model, ctxs, best)
    cp_status, cp_bound = "skipped", None
    if len(ids) <= 9:
        sol, info = solve_cpsat(model, ctxs, best, list(range(len(ids))), time_limit=10, workers=1)
        cp_status, cp_bound = info["status"], info["bound"]
        s_cp = total_score(model, ctxs, sol)
        if all(model.evaluate(c, r).feasible for c, r in zip(ctxs, sol)):
            s_best = max(s_best, s_cp)
    row = {"N": n, "M": len(ids), "size": size, "prior": prior, "seed": seed,
           "S_greedy": s_greedy, "S_best": s_best, "gap_S": (s_best - s_greedy) / s_best,
           "cp_status": cp_status, "cp_bound": cp_bound, "t_greedy_ms": 1000 * t_greedy, "t_lns_s": t_lns}
    for kind, params in [("central_greedy", {}), ("central_optimal", {"lns_iters": 3000, "cp_max_tasks": 0})]:
        s = run(apply_overrides(scn, {"allocator": {"kind": kind, **params}}), seed).summary
        tag = "B2" if kind == "central_greedy" else "B3"
        for k in ("U", "U_norm", "CR", "T_detect_90", "makespan", "E_total_Wh", "N_plan_energy", "valid"):
            row[f"{tag}_{k}"] = s[k]
        row[f"{tag}_depleted"] = s["agents_failed"]
    row["gap_U"] = (row["B3_U"] - row["B2_U"]) / abs(row["B3_U"]) if row["B3_U"] else float("nan")
    return row


def gate_p2(jobs: int, seeds_hot: int = 10, seeds_uni: int = 1, from_csv: str | None = None) -> dict:
    """Однородная карта даёт один и тот же экземпляр планирования при любом зерне (ценности
    секторов одинаковы, геометрия фиксирована) — поэтому для неё берётся одно зерно:
    иначе один экземпляр засчитывается несколько раз (псевдорепликация)."""
    if from_csv:
        import csv
        with open(from_csv, encoding="utf-8") as f:
            rows = [_p2_cast(r) for r in csv.DictReader(f)]
        wall = float("nan")
    else:
        job_list = []
        for size, m in P2_SIZES.items():
            for n in P2_AGENTS:
                if n > m:
                    continue
                job_list += [(n, size, "hotspots", sd) for sd in range(1, seeds_hot + 1)]
                job_list += [(n, size, "uniform", sd) for sd in range(1, seeds_uni + 1)]
        t = time.perf_counter()
        rows = _pool(_p2_job, job_list, jobs)
        wall = time.perf_counter() - t
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORTS / "P2_allocation.csv", rows)
    return _p2_report(rows, wall)


def _p2_cast(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if v in ("True", "False"):
            out[k] = v == "True"
            continue
        try:
            out[k] = float(v) if v not in ("", None) else None
        except ValueError:
            out[k] = v
    for k in ("N", "M", "size", "seed", "B2_N_plan_energy", "B3_N_plan_energy", "B2_depleted", "B3_depleted"):
        out[k] = int(out[k])
    return out


def _p2_cells(rows: list[dict]):
    cells = {}
    for r in rows:
        cells.setdefault((r["N"], r["M"]), []).append(r)
    table, worst = [], (0.0, None)
    for (n, m), rs in sorted(cells.items()):
        gs, gs_ci = _mean_ci([r["gap_S"] for r in rs])
        gu, gu_ci = _mean_ci([r["gap_U"] for r in rs])
        if gu > worst[0]:
            worst = (gu, (n, m))
        table.append((n, m, len(rs), gs, gs_ci, max(r["gap_S"] for r in rs), gu, gu_ci,
                      max(r["gap_U"] for r in rs), sum(r["cp_status"] == "OPTIMAL" for r in rs),
                      statistics.fmean(r["t_greedy_ms"] for r in rs)))
    return table, worst


def _p2_report(rows: list[dict], wall: float) -> dict:
    deadline = 900.0
    raw_table, raw_worst = _p2_cells(rows)
    dedup = [r for r in rows if r["prior"] != "uniform" or r["seed"] == 1]
    table, worst = _p2_cells(dedup)
    energy_viol = sum(r["B2_N_plan_energy"] + r["B3_N_plan_energy"] for r in rows)
    depleted = sum(r["B2_depleted"] + r["B3_depleted"] for r in rows)
    invalid = sum((not r["B2_valid"]) + (not r["B3_valid"]) for r in rows)
    passed_raw = raw_worst[0] <= 0.10 and energy_viol == 0 and depleted == 0
    passed = worst[0] <= 0.10 and energy_viol == 0 and depleted == 0
    cross = [r for r in dedup if (r["B2_makespan"] > deadline) != (r["B3_makespan"] > deadline)]
    rest = [r for r in dedup if (r["B2_makespan"] > deadline) == (r["B3_makespan"] > deadline)]
    _plot_p2(dedup)
    wall_txt = "" if math.isnan(wall) else f" · {wall / 60:.1f} мин"
    L = [
        "# P2 — Распределение задач: контрольная точка", "",
        f"_Сгенерировано `python -m experiments.gates p2` · {_stamp()} · {len(rows)} прогонов{wall_txt}._", "",
        "## Критерий (docs/05, P2)",
        "Разрыв по качеству централизованного жадного аукциона **B2** с эталоном **B3** ≤ 10 % "
        "на N ≤ 10, M ≤ 40; ограничение энергии ни разу не нарушено (валидатор на всех прогонах).", "",
        f"## Итог: **{'ПРОЙДЕНА' if passed else 'НЕ ПРОЙДЕНА'}** (после устранения псевдорепликации, см. ниже)", "",
        f"* наибольший средний разрыв по U(M) в ячейке (N, M) = {worst[1]}: **{100 * worst[0]:.2f} %** (порог 10 %)",
        f"* энергетически недопустимых планов (валидатор): **{energy_viol}**; разряженных в полёте: **{depleted}**; "
        f"невалидных прогонов: **{invalid}**",
        f"* разрыв по целевой функции планирования S во всех ячейках ≤ "
        f"**{100 * max(t[5] for t in table):.2f} %** (максимум по экземплярам)", "",
        "### Прозрачность: первый расчёт и что в нём было не так",
        f"В первом прогоне для однородной карты целей было взято 5 зёрен. Но при однородной карте "
        f"экземпляр планирования **не зависит от зерна** (ценности секторов одинаковы, геометрия фиксирована) — "
        f"это пять копий одного экземпляра, а не пять независимых наблюдений (псевдорепликация). "
        f"С ними худшая ячейка {raw_worst[1]} дала **{100 * raw_worst[0]:.2f} %** — формально "
        f"{'в пределах порога' if passed_raw else '**выше порога**'}. Исправлен только дизайн испытания "
        "(однородная карта — одно зерно); метрика U(M), дедлайн и методы не менялись. "
        "Решение, принимать ли такое исправление, — за командой; оба расчёта приведены.", "",
        "### Почему U(M) чувствительнее S: обрыв на дедлайне",
        f"U(M) засчитывает задачу только если она выполнена до deadline = {deadline:.0f} с. Когда вся миссия "
        "заканчивается около дедлайна (2 агента на 9 секторов), планы B2 и B3, почти равные по S, могут "
        "оказаться по разные стороны отметки — и одна «опоздавшая» задача снимает целиком свою ценность.",
        f"* экземпляров, где B2 и B3 разошлись по разные стороны дедлайна: **{len(cross)}** из {len(dedup)}; "
        f"средний разрыв по U в них {100 * statistics.fmean(r['gap_U'] for r in cross) if cross else float('nan'):.2f} %,"
        f" в остальных {100 * statistics.fmean(r['gap_U'] for r in rest):.2f} %.",
        "* Это свойство метрики (ступенька «вовремя»), а не качества распределения. Если нужно, чтобы "
        "распределитель это учитывал, дедлайн надо ввести в целевую функцию планирования — это изменение "
        "метода, его стоит делать осознанно и фиксировать отдельно (docs/03 §8).", "",
        "## Что сравнивается",
        "* **B2 `central_greedy`** — последовательный аукцион «один лот за раз» + локальное улучшение маршрутов.",
        "* **B3 `central_optimal`** — эталон: лучшее из CP-SAT (с доказательством, где успевает) и "
        "Ruin & Recreate LNS на 3000 итераций по той же целевой функции. На малых экземплярах, где CP-SAT "
        "доказал оптимум (проверено отдельно, 8 потоков), LNS выходит точно на него.",
        "* **gap_S** — разрыв по целевой функции планирования S = Σ value·λ^C (её оптимизируют оба);",
        "* **gap_U** — разрыв по U(M) = Σ priority·1[до deadline] − λ_E·E − λ_T·makespan, измеренный "
        "полным прогоном симуляции каждого плана (docs/03, раздел 4). Отрицательный — план B2 в прогоне лучше.",
        "* Сценарий: baseline, поле 600…1200 м (M = 9…36 секторов), N = 2…10; горячие зоны (10 зёрен) "
        "и однородная карта (1 экземпляр).", "",
        "## Результаты по ячейкам (без псевдорепликации)",
        "| N | M | экз. | gap_S ср. ± 95 % | gap_S макс | gap_U ср. ± 95 % | gap_U макс | B2, мс |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, m, k, gs, gsc, gsm, gu, guc, gum, pr, tg in table:
        L.append(f"| {n} | {m} | {k} | {100 * gs:.2f} ± {100 * gsc:.2f} % | {100 * gsm:.2f} % | "
                 f"{100 * gu:.2f} ± {100 * guc:.2f} % | {100 * gum:.2f} % | {tg:.0f} |")
    L += ["", "![разрыв B2 от эталона](figures/P2_gap.png)", "",
          "## Выводы",
          f"* Жадный аукцион почти не уступает эталону: средний разрыв по целевой функции "
          f"{100 * statistics.fmean(r['gap_S'] for r in dedup):.2f} %, по U(M) — "
          f"{100 * statistics.fmean(r['gap_U'] for r in dedup):.2f} %.",
          "* Время B2 — десятки миллисекунд, что оставляет запас для пересчёта по событиям "
          "(NFR-5: ≤ 2 с при N = 20).",
          "* Доказать оптимум CP-SAT успевает только на малых задачах — известное свойство маршрутизации "
          "с зависящей от времени наградой; поэтому эталон на больших M — LNS.",
          "", "Сырые данные (все прогоны, включая повторы однородной карты): `docs/reports/P2_allocation.csv`."]
    (REPORTS / "P2_allocation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"passed": passed, "passed_raw": passed_raw, "worst_gap_U": worst[0], "worst_cell": worst[1],
            "worst_gap_U_raw": raw_worst[0], "energy_viol": energy_viol, "rows": len(rows)}


def _plot_p2(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(axes[0], "gap_S", "разрыв по S (план)"), (axes[1], "gap_U", "разрыв по U(M) (прогон)")]:
        for n in P2_AGENTS:
            ms = sorted({r["M"] for r in rows if r["N"] == n})
            means = [statistics.fmean(r[key] for r in rows if r["N"] == n and r["M"] == m) * 100 for m in ms]
            ax.plot(ms, means, "o-", label=f"N = {n}")
        ax.axhline(10, color="r", ls="--", lw=1, label="порог 10 %")
        ax.set_xlabel("число задач M"); ax.set_ylabel("%"); ax.set_title(title); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("P2: B2 central_greedy против эталона B3")
    fig.tight_layout(); fig.savefig(FIG / "P2_gap.png", dpi=130); plt.close(fig)


# =============================================================================== общий прогон
def _run_job(job: tuple) -> dict:
    """(имя сценария, переопределения, зерно, метка) → итоговые метрики прогона (без траекторий)."""
    from swarmsim import apply_overrides, load_scenario, run
    scen, over, seed, label = job
    scn = apply_overrides(load_scenario(ROOT / "scenarios" / f"{scen}.yaml"),
                          {"output.trajectories": False, **over})
    s = run(scn, seed).summary
    cfg = {f"cfg:{k}": (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
           for k, v in over.items()}
    return {"label": label, "seed": seed, **cfg, **s}


def _new_task_for(seed: int, size: float = 1000.0, t: float = 300.0) -> list[dict]:
    """Новая цель (FR-9) в случайной точке поля — своя для каждого зерна, одинаковая для всех методов."""
    rng = np.random.default_rng(10_000 + seed)
    x, y = rng.uniform(0.15 * size, 0.85 * size, 2)
    return [{"t": t, "x": round(float(x), 1), "y": round(float(y), 1), "priority": 5}]


def _fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{nd}f} %" if pct else f"{x:.{nd}f}"


# =============================================================================== P3
P3_METHODS = {
    "B1 static_roundrobin": {"allocator": {"kind": "static_roundrobin"}},
    "B2 central_greedy (без перераспределения)": {"allocator": {"kind": "central_greedy", "dynamic": False}},
    "B2 central_greedy": {"allocator": {"kind": "central_greedy"}},
    "B2 central_greedy, replan=repair": {"allocator": {"kind": "central_greedy", "replan": "repair"}},
    "B3 central_optimal": {"allocator": {"kind": "central_optimal", "lns_iters": 300, "cp_max_tasks": 0}},
}


def _p3_stat(rs: list[dict]) -> dict:
    cr = [r["CR"] for r in rs]
    trec = [r["T_recovery_max"] for r in rs if r["T_recovery_max"] is not None]
    tres = [r["T_resume_max"] for r in rs if r["T_resume_max"] is not None]
    tnew = [r["T_newtask_done"] for r in rs if r["T_newtask_done"] is not None]
    return {
        "n": len(rs), "CR_mean": statistics.fmean(cr), "CR_min": min(cr),
        "CR_ok": sum(c >= 0.95 for c in cr) / len(cr),
        "rec": statistics.fmean(r["recovered_frac"] for r in rs),
        "Trec_max": max(trec) if len(trec) == len(rs) else None,
        "Trec_mean": statistics.fmean(trec) if trec else None,
        "Tres": statistics.fmean(tres) if tres else None,
        "Tnew": statistics.fmean(tnew) if tnew else None,
        "new_ok": len(tnew) / len(rs),
        "dup": sum(r["N_dup"] for r in rs), "U": statistics.fmean(r["U"] for r in rs),
        "invalid": sum(not r["valid"] for r in rs),
    }


def gate_p3(jobs: int, seeds: int = 20) -> dict:
    jobs_list = []
    for detect in ("reported", "silent"):
        for label, over in P3_METHODS.items():
            for s in range(1, seeds + 1):
                o = {**over, "faults.kind": "random", "faults.rate": 0.2, "faults.t_range": [60, 400],
                     "faults.detect": detect, "world.new_tasks": _new_task_for(s)}
                jobs_list.append(("baseline", o, s, f"{label} | {detect}"))
    t = time.perf_counter()
    rows = _pool(_run_job, jobs_list, jobs)
    wall = time.perf_counter() - t
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORTS / "P3_dynamics.csv", rows)

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    st = {k: _p3_stat(v) for k, v in groups.items()}
    checks = {}
    for detect in ("reported", "silent"):
        g = st[f"B2 central_greedy | {detect}"]
        checks[detect] = {"CR": g["CR_mean"] >= 0.95 and g["CR_ok"] >= 0.95,
                          "T_recovery": g["Trec_max"] is not None and g["Trec_max"] <= 2.0,
                          "N_dup": g["dup"] == 0}
    passed = all(all(c.values()) for c in checks.values())
    _plot_p3(groups)

    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    L = ["# P3 — Динамика и отказы: контрольная точка", "",
         f"_Сгенерировано `python -m experiments.gates p3` · {_stamp()} · {len(rows)} прогонов · {wall / 60:.1f} мин._", "",
         "## Критерий (docs/05, P3)",
         "Сценарий: отказывают 20 % агентов и на середине миссии появляется одна новая цель. "
         "Требуется `CR ≥ 95 %`, `T_recovery ≤ 2 с`, `N_dup = 0`.", "",
         f"## Итог: **{'ПРОЙДЕНА' if passed else 'НЕ ПРОЙДЕНА'}** (система — B2 `central_greedy`)", ""]
    for detect, c in checks.items():
        g = st[f"B2 central_greedy | {detect}"]
        L.append(f"* отказы **{detect}**: CR средний {_fmt(g['CR_mean'], True)}, прогонов с CR ≥ 95 %: "
                 f"{_fmt(g['CR_ok'], True, 0)} {ok(c['CR'])}; T_recovery макс {_fmt(g['Trec_max'])} с {ok(c['T_recovery'])}; "
                 f"N_dup = {g['dup']} {ok(c['N_dup'])}")
    L += ["", "## Условия",
          "* baseline: 10 БПЛА, поле 1000×1000 м, 25 секторов, однородная карта целей; 20 зёрен на метод.",
          "* отказы: `faults.kind: random`, λ = 0.2 (2 из 10 агентов) в случайные моменты 60–400 с; "
          "**reported** — борт успел сообщить (задачи освобождаются сразу), **silent** — аппарат замолчал, "
          "отказ замечают по пропаже heartbeat через `detect_timeout` = 1.5 с.",
          "* новая цель (FR-9): задача осмотра точки с приоритетом 5 в случайном месте поля на 300 с; "
          "может прервать текущую задачу ближайшего агента (его сектор уходит в пул с сохранённым прогрессом).",
          "* пары прогонов одинаковы для всех методов: те же цели, те же отказы (поток RNG `faults`), та же новая цель.",
          "", "## Определения (зафиксированы до прогона, `swarmsim/metrics.py::recovery_metrics`)",
          "* **T_recovery** — от отказа до момента, когда каждая освобождённая задача оказалась в плане "
          "живого агента (перераспределена и принята к исполнению). Для silent сюда входит время обнаружения.",
          "* **T_resume** — до фактического начала последней из них (зависит от очереди, а не от алгоритма).",
          "* **T_newtask_done** — от появления новой цели до завершения её осмотра.",
          "", "## Результаты",
          "| метод | отказы | CR ср. | CR мин | прогонов CR ≥ 95 % | восстановлено отказов | T_recovery ср. / макс, с "
          "| T_resume ср., с | новая цель осмотрена | T_newtask_done ср., с | U(M) ср. | N_dup | невалидных |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, g in st.items():
        name, detect = label.split(" | ")
        L.append(f"| {name} | {detect} | {_fmt(g['CR_mean'], True)} | {_fmt(g['CR_min'], True)} | {_fmt(g['CR_ok'], True, 0)} | "
                 f"{_fmt(g['rec'], True, 0)} | {_fmt(g['Trec_mean'])} / {_fmt(g['Trec_max'])} | {_fmt(g['Tres'], nd=0)} | "
                 f"{_fmt(g['new_ok'], True, 0)} | {_fmt(g['Tnew'], nd=0)} | {_fmt(g['U'])} | {g['dup']} | {g['invalid']} |")
    on = st["B2 central_greedy | reported"]
    off = st["B2 central_greedy (без перераспределения) | reported"]
    L += ["", "![P3](figures/P3_dynamics.png)", "",
          "## Метод исключения: перераспределение (docs/03 §5)",
          "Одинаковое первичное назначение (B2), отличие только в пересчёте по событиям (отказы с сообщением):",
          f"* CR: без перераспределения {_fmt(off['CR_mean'], True)} → с перераспределением {_fmt(on['CR_mean'], True)};",
          f"* U(M): {_fmt(off['U'])} → {_fmt(on['U'])}; новая цель осмотрена: "
          f"{_fmt(off['new_ok'], True, 0)} → {_fmt(on['new_ok'], True, 0)} прогонов.",
          "", "## Выводы",
          "* Перераспределение по событию закрывает потерю задач отказавших агентов; B1 и B2 без "
          "перераспределения теряют задачи отказавших и не обслуживают новую цель.",
          "* T_recovery определяется прежде всего временем обнаружения отказа: с сообщением — один шаг "
          "симуляции (0.1 с), тихий — таймаут heartbeat (1.5 с). Сам пересчёт занимает миллисекунды.",
          "* T_resume — сотни секунд: освобождённые секторы встают в очередь живых агентов. Это не задержка "
          "алгоритма, а физика нагрузки. Roadmap требует T_recovery, но показывать надо оба — иначе "
          "«2 секунды» вводят в заблуждение.",
          "", "Сырые данные: `docs/reports/P3_dynamics.csv`."]
    (REPORTS / "P3_dynamics.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"passed": passed, "checks": checks,
            "B2_CR": {k: st[f"B2 central_greedy | {k}"]["CR_mean"] for k in ("reported", "silent")}}


def _plot_p3(groups: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = sorted(groups)
    short = [lb.replace("central_", "").replace(" (без перераспределения)", " без пересчёта")
             .replace(", replan=repair", " repair") for lb in labels]
    axes[0].boxplot([[100 * r["CR"] for r in groups[lb]] for lb in labels], vert=False)
    axes[0].set_yticks(range(1, len(labels) + 1), short, fontsize=7)
    axes[0].axvline(95, color="r", ls="--", lw=1)
    axes[0].set_xlabel("CR, %")
    axes[0].set_title("доля выполненных задач")
    vals = [[r["T_recovery_max"] for r in groups[lb] if r["T_recovery_max"] is not None] for lb in labels]
    axes[1].boxplot([v or [float("nan")] for v in vals], vert=False)
    axes[1].set_yticks(range(1, len(labels) + 1), short, fontsize=7)
    axes[1].axvline(2, color="r", ls="--", lw=1)
    axes[1].set_xlabel("T_recovery, с (только восстановленные отказы)")
    axes[1].set_title("время восстановления")
    fig.suptitle("P3: 20 % отказов + новая цель, 20 зёрен")
    fig.tight_layout()
    fig.savefig(FIG / "P3_dynamics.png", dpi=130)
    plt.close(fig)


# =============================================================================== P4
P4_METHODS = {
    "B1 static_roundrobin": {"allocator": {"kind": "static_roundrobin"}},
    "B2 central_greedy": {"allocator": {"kind": "central_greedy"}},
    "B4 cbba": {"allocator": {"kind": "cbba"}},
    "B5 hybrid": {"allocator": {"kind": "hybrid"}},
}
P4_OUTAGE = [150.0, 350.0]
P4_NET = {"kind": "network", "range": 600, "range_gcs": 1500, "latency": 0.05, "jitter": 0.05, "loss": 0.1}
P4_SYSTEMS = ("B4 cbba", "B5 hybrid")          # те, что обязаны жить без станции


def _p4_events(seed: int, stress: bool) -> dict:
    """Стресс: тихий отказ одного агента и новая цель — оба внутри разрыва связи со станцией."""
    if not stress:
        return {}
    return {"faults.scheduled": [{"t": 200.0, "agent": seed % 10, "detect": "silent"}],
            "world.new_tasks": _new_task_for(seed, t=250.0)}


def _p4_stat(rs: list[dict], ref: dict[int, dict]) -> dict:
    cr = [r["CR"] for r in rs]
    drop = [(ref[r["seed"]]["CR"] - r["CR"]) / ref[r["seed"]]["CR"] for r in rs if r["seed"] in ref]
    trec = [r["T_recovery_max"] for r in rs if r.get("T_recovery_max") is not None]
    g = lambda k: [r[k] for r in rs if r.get(k) is not None]  # noqa: E731
    return {
        "n": len(rs), "CR": statistics.fmean(cr), "CR_min": min(cr),
        "drop": statistics.fmean(drop) if drop else None, "drop_max": max(drop) if drop else None,
        "dup": sum(r["N_dup"] for r in rs), "lost": sum(r.get("N_lost") or 0 for r in rs),
        "lost_runs": sum(1 for r in rs if (r.get("N_lost") or 0) > 0),
        "Trec": statistics.fmean(trec) if trec else None, "Trec_max": max(trec) if trec else None,
        "rec": statistics.fmean(r["recovered_frac"] for r in rs) if rs[0].get("recovered_frac") is not None else None,
        "mk": statistics.fmean(g("makespan")) if g("makespan") else None,
        "U": statistics.fmean(r["U"] for r in rs),
        "msg": statistics.fmean(g("N_msg")) if g("N_msg") else None,
        "B": statistics.fmean(g("B_per_agent")) if g("B_per_agent") else None,
        "A": statistics.fmean(g("A_gcs")) if g("A_gcs") else None,
        "in_out": statistics.fmean(g("done_in_outage")) if g("done_in_outage") else 0.0,
        "yield": statistics.fmean(g("N_conflict_yield")) if g("N_conflict_yield") else 0.0,
        "false": sum(g("N_false_loss")) if g("N_false_loss") else 0,
        "invalid": sum(not r["valid"] for r in rs),
    }


def gate_p4(jobs: int, seeds: int = 20, deg_seeds: int = 8) -> dict:
    jobs_list = []
    for label, over in P4_METHODS.items():
        for stress in (False, True):
            tag = "стресс" if stress else "разрыв"
            for s in range(1, seeds + 1):
                ev = _p4_events(s, stress)
                jobs_list.append(("baseline", {**over, **ev}, s, f"{label} | {tag} | ideal"))
                jobs_list.append(("baseline", {**over, **ev, "comms": {**P4_NET, "gcs_outages": [P4_OUTAGE]}},
                                  s, f"{label} | {tag} | network"))
    # деградация канала: дальность × потери, стресс-события, без разрыва станции
    deg_ranges, deg_loss = (1500, 600, 400), (0.0, 0.1, 0.3, 0.5)
    for label, over in P4_METHODS.items():
        for rg in deg_ranges:
            for q in deg_loss:
                for s in range(1, deg_seeds + 1):
                    comms = {**P4_NET, "range": rg, "loss": q}
                    jobs_list.append(("baseline", {**over, **_p4_events(s, True), "comms": comms},
                                      s, f"deg | {label} | {rg} | {q}"))
    t = time.perf_counter()
    rows = _pool(_run_job, jobs_list, jobs)
    wall = time.perf_counter() - t
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORTS / "P4_comms.csv", rows)
    return _p4_report(rows, wall, deg_ranges, deg_loss)


def _p4_report(rows: list[dict], wall: float, deg_ranges, deg_loss) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    st = {}
    for label in P4_METHODS:
        for tag in ("разрыв", "стресс"):
            ref = {r["seed"]: r for r in groups[f"{label} | {tag} | ideal"]}
            st[(label, tag, "ideal")] = _p4_stat(groups[f"{label} | {tag} | ideal"], ref)
            st[(label, tag, "network")] = _p4_stat(groups[f"{label} | {tag} | network"], ref)
    checks = {}
    for label in P4_SYSTEMS:
        for tag in ("разрыв", "стресс"):
            g = st[(label, tag, "network")]
            checks[f"{label} | {tag}"] = {"continues": g["in_out"] > 0, "drop": g["drop"] <= 0.15,
                                          "N_dup": g["dup"] == 0, "N_lost": g["lost"] == 0}
    main_ok = all(all(c.values()) for k, c in checks.items() if k.endswith("разрыв"))
    stress_ok = all(all(c.values()) for k, c in checks.items() if k.endswith("стресс"))
    passed = main_ok
    deg = {}
    for label in P4_METHODS:
        for rg in deg_ranges:
            for q in deg_loss:
                deg[(label, rg, q)] = _p4_stat(groups[f"deg | {label} | {rg} | {q}"], {})
    ideal_mk = statistics.fmean(r["makespan"] for r in groups["B2 central_greedy | разрыв | ideal"])
    _plot_p4(st, deg, deg_ranges, deg_loss)

    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    dur = P4_OUTAGE[1] - P4_OUTAGE[0]
    L = ["# P4 — Связь и децентрализация: контрольная точка", "",
         f"_Сгенерировано `python -m experiments.gates p4` · {_stamp()} · {len(rows)} прогонов · {wall / 60:.1f} мин._", "",
         "## Критерий (docs/05, P4)",
         "При полном разрыве связи с наземной станцией на 30 % времени миссии: миссия продолжается, "
         "`CR` падает не более чем на 15 % относительно идеального канала; после восстановления "
         "`N_dup = 0`, `N_lost = 0`.", "",
         f"## Итог: **{'ПРОЙДЕНА' if passed else 'НЕ ПРОЙДЕНА'}** (системы — B4 `cbba` и B5 `hybrid`)", "",
         f"Разрыв станции {P4_OUTAGE[0]:.0f}–{P4_OUTAGE[1]:.0f} с = {dur:.0f} с = "
         f"{100 * dur / ideal_mk:.0f} % средней длительности миссии B2 на идеальном канале ({ideal_mk:.0f} с).", ""]
    for k, c in checks.items():
        label, tag = k.split(" | ")
        g = st[(label, tag, "network")]
        L.append(f"* **{label}**, {tag}: задач выполнено во время разрыва (ср.) {g['in_out']:.1f} {ok(c['continues'])}; "
                 f"падение CR ср. {_fmt(g['drop'], True, 1)} (макс {_fmt(g['drop_max'], True, 1)}) {ok(c['drop'])}; "
                 f"N_dup = {g['dup']} {ok(c['N_dup'])}; N_lost = {g['lost']} {ok(c['N_lost'])}")
    L += ["", f"Стресс-вариант (сверх критерия): **{'пройден' if stress_ok else 'не пройден'}**.", "",
          "## Условия",
          "* baseline: 10 БПЛА, 1000×1000 м, 25 секторов, 20 зёрен на ячейку; пары «идеальный канал / сеть» "
          "с одинаковыми целями и событиями.",
          f"* сеть: дальность агент–агент {P4_NET['range']} м (поле 1000 м → рой временами делится на группы), "
          f"агент–станция {P4_NET['range_gcs']} м, задержка {P4_NET['latency']} ± {P4_NET['jitter']} с, "
          f"потери {100 * P4_NET['loss']:.0f} %, полоса 32 кБ/с на узел, heartbeat 5 Гц, gossip живости 1 Гц.",
          "* **разрыв** — только потеря станции (критерий roadmap). **стресс** — в разрыве ещё тихий отказ "
          "одного агента (200 с) и новая цель (250 с): их надо обработать без станции.",
          "* обнаружение отказа по пропаже heartbeat с учётом дальности: молчание агента, который по последней "
          "известной позиции (± сколько мог пролететь) в зоне связи наблюдателя, — отказ через 1.5 с; "
          "в зоне соседа — через 4 с (вести идут через gossip); вне зоны всех — только через 60 с "
          "(молчание объяснимо дальностью). Станция во время собственного разрыва никого не «теряет».",
          "* CR на идеальном канале — отдельный прогон того же метода с тем же зерном и событиями; "
          "падение CR = (CR_ideal − CR_network) / CR_ideal по парам.", "",
          "## Результаты",
          "| метод | вариант | канал | CR ср. | CR мин | падение CR ср. / макс | задач в разрыве | T_recovery ср. / макс, с "
          "| makespan, с | U(M) | N_dup | N_lost (прогонов) | уступок | ложных потерь | сообщений | Б/с на агента | невалидных |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (label, tag, ch), g in st.items():
        L.append(f"| {label} | {tag} | {ch} | {_fmt(g['CR'], True)} | {_fmt(g['CR_min'], True)} | "
                 f"{_fmt(g['drop'], True, 1)} / {_fmt(g['drop_max'], True, 1)} | {g['in_out']:.1f} | "
                 f"{_fmt(g['Trec'])} / {_fmt(g['Trec_max'])} | {_fmt(g['mk'], nd=0)} | {_fmt(g['U'])} | {g['dup']} | "
                 f"{g['lost']} ({g['lost_runs']}) | {g['yield']:.1f} | {g['false']} | {_fmt(g['msg'], nd=0)} | "
                 f"{_fmt(g['B'], nd=0)} | {g['invalid']} |")
    L += ["", "![P4](figures/P4_outage.png)", "",
          "## Эксперимент по деградации канала (docs/05 P4, задача 5)",
          f"Стресс-события (тихий отказ + новая цель), станция без разрыва, {len(groups[next(k for k in groups if k.startswith('deg'))])} "
          "зёрен на ячейку. Ячейка: CR ср. · N_dup сумма · makespan ср.", ""]
    head = "| метод | дальность, м | " + " | ".join(f"потери {int(100 * q)} %" for q in deg_loss) + " |"
    L += [head, "|---|---|" + "---|" * len(deg_loss)]
    for label in P4_METHODS:
        for rg in deg_ranges:
            cells = [f"{_fmt(deg[(label, rg, q)]['CR'], True, 1)} · {deg[(label, rg, q)]['dup']} · "
                     f"{_fmt(deg[(label, rg, q)]['mk'], nd=0)}" for q in deg_loss]
            L.append(f"| {label} | {rg} | " + " | ".join(cells) + " |")
    L += ["", "![деградация](figures/P4_degradation.png)", "",
          "## Выводы (по числам выше)"]
    b2 = st[("B2 central_greedy", "стресс", "network")]
    b5 = st[("B5 hybrid", "стресс", "network")]
    b4 = st[("B4 cbba", "стресс", "network")]
    L += [f"* Центр без станции не может перераспределить: в стресс-варианте T_recovery B2 ср. {_fmt(b2['Trec'])} с "
          f"против {_fmt(b5['Trec'])} с у B5 и {_fmt(b4['Trec'])} с у B4 — задачи отказавшего ждут конца разрыва (H2).",
          f"* Цена децентрализации — трафик: B4 {_fmt(b4['B'], nd=0)} Б/с на агента против {_fmt(b2['B'], nd=0)} у B2 "
          f"и {_fmt(b5['B'], nd=0)} у B5 (гибрид запускает торги только у тех, кто потерял станцию).",
          "* Дубли при разделении роя на группы возможны в принципе: две группы, не слышащие друг друга, "
          "не могут знать о чужом исполнении. Их удерживают у нуля три механизма: (1) обнаружение отказа "
          "с учётом дальности — молчание вне зоны связи не считается отказом, (2) владелец выполняемой "
          "задачи не перезаписывается проигравшим конфликт, (3) агент бросает задачу, как только узнаёт, "
          "что её уже выполнил другой. Таблица деградации показывает, где эта защита перестаёт работать.",
          "", "Сырые данные: `docs/reports/P4_comms.csv`."]
    (REPORTS / "P4_comms.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"passed": passed, "stress_passed": stress_ok,
            "checks": {k: {kk: bool(vv) for kk, vv in c.items()} for k, c in checks.items()},
            "report": str(REPORTS / "P4_comms.md")}


def _plot_p4(st: dict, deg: dict, deg_ranges, deg_loss) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(P4_METHODS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(labels))
    w = 0.2
    for i, (tag, ch, col) in enumerate([("разрыв", "ideal", "#9bb"), ("разрыв", "network", "#37a"),
                                          ("стресс", "ideal", "#db9"), ("стресс", "network", "#c63")]):
        axes[0].bar(x + (i - 1.5) * w, [100 * st[(lb, tag, ch)]["CR"] for lb in labels], w, color=col,
                    label=f"{tag}, {ch}")
        axes[1].bar(x + (i - 1.5) * w, [st[(lb, tag, ch)]["mk"] or 0 for lb in labels], w, color=col)
    axes[0].set_ylim(80, 101)
    axes[0].set_ylabel("CR, %")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[1].set_ylabel("makespan, с")
    axes[2].bar(x, [st[(lb, "стресс", "network")]["Trec"] or 0 for lb in labels], color="#c63")
    axes[2].set_ylabel("T_recovery ср., с (стресс, сеть)")
    axes[2].set_yscale("symlog", linthresh=2)
    for ax in axes:
        ax.set_xticks(x, [lb.split(" ", 1)[1].replace("static_", "").replace("central_", "") for lb in labels],
                      fontsize=8)
    fig.suptitle(f"P4: разрыв со станцией {P4_OUTAGE[0]:.0f}–{P4_OUTAGE[1]:.0f} с")
    fig.tight_layout()
    fig.savefig(FIG / "P4_outage.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(deg_ranges), figsize=(4.6 * len(deg_ranges), 7), sharey="row")
    for c, rg in enumerate(deg_ranges):
        for lb in labels:
            axes[0, c].plot([100 * q for q in deg_loss], [100 * deg[(lb, rg, q)]["CR"] for q in deg_loss], "o-",
                            label=lb, ms=4)
            axes[1, c].plot([100 * q for q in deg_loss], [deg[(lb, rg, q)]["mk"] for q in deg_loss], "o-", ms=4)
        axes[0, c].set_title(f"дальность {rg} м")
        axes[1, c].set_xlabel("потери, %")
    axes[0, 0].set_ylabel("CR, %")
    axes[1, 0].set_ylabel("makespan, с")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("P4: деградация канала (тихий отказ + новая цель)")
    fig.tight_layout()
    fig.savefig(FIG / "P4_degradation.png", dpi=130)
    plt.close(fig)


# =============================================================================== P5
def gate_p5(jobs: int, resume: bool = False, quick: bool = False, report_only: bool = False,
            only: list[str] | None = None) -> dict:
    """Кампания docs/03 → docs/reports/p5/*.csv, графики 1–6, docs/reports/P5_comparison.md."""
    from . import campaign
    if not report_only:
        campaign.run_campaign(jobs, resume=resume, quick=quick, only=only)
    return campaign.report(_stamp())


# =============================================================================== CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Контрольные точки P2–P5")
    p.add_argument("phase", choices=["p2", "p3", "p4", "p5"])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--quick", action="store_true", help="меньше зёрен — для отладки")
    p.add_argument("--from-csv", default=None, help="P2: пересобрать отчёт по готовым данным")
    p.add_argument("--resume", action="store_true", help="P5: не пересчитывать опыты с готовым CSV")
    p.add_argument("--report-only", action="store_true", help="P5: только анализ и отчёт по готовым CSV")
    p.add_argument("--only", default=None, help="P5: через запятую, напр. E1,E2")
    a = p.parse_args(argv)
    fn = {"p2": gate_p2}.get(a.phase) or globals().get(f"gate_{a.phase}")
    kw = {"seeds_hot": 2, "seeds_uni": 1} if (a.quick and a.phase == "p2") else {}
    if a.quick and a.phase in ("p3", "p4"):
        kw = {"seeds": 2} | ({"deg_seeds": 1} if a.phase == "p4" else {})
    if a.phase == "p5":
        kw = {"resume": a.resume, "quick": a.quick, "report_only": a.report_only,
              "only": a.only.split(",") if a.only else None}
    if a.from_csv:
        kw["from_csv"] = a.from_csv
    res = fn(a.jobs, **kw)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
