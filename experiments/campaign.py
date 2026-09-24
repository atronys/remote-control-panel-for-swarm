"""P5 — сравнение архитектур: кампания docs/03 (последовательное планирование), статистика, графики 1–6.

    python -m experiments.gates p5 --jobs 11             # всё: прогоны + анализ + отчёт
    python -m experiments.gates p5 --jobs 11 --resume    # досчитать недостающие опыты, готовые CSV не трогать
    python -m experiments.gates p5 --report-only         # только анализ по готовым CSV

Дизайн зафиксирован в этом файле ДО прогона (docs/03 §8, правило 5): опыты, уровни, число зёрен,
метрики, правила вердиктов по гипотезам. Все методы в опыте — на одних и тех же зёрнах (парный дизайн).

Система во всех опытах (кроме A4): эшелоны по высоте (стратегическая деконфликтуация) + ORCA как
страховка (тактическое избегание) — жёсткое ограничение N_col = 0 должно выполняться при любом N.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from .stats import describe, effect_word, paired

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
DATA = REPORTS / "p5"
FIG = REPORTS / "figures"

METHODS = {
    "B0": {"allocator": {"kind": "static_oracle"}},
    "B1": {"allocator": {"kind": "static_roundrobin"}},
    "B2": {"allocator": {"kind": "central_greedy"}},
    "B3": {"allocator": {"kind": "central_optimal"}},
    "B4": {"allocator": {"kind": "cbba"}},
    "B5": {"allocator": {"kind": "hybrid"}},
}
NAMES = {"B0": "B0 static_oracle", "B1": "B1 static_roundrobin", "B2": "B2 central_greedy",
         "B3": "B3 central_optimal", "B4": "B4 cbba", "B5": "B5 hybrid"}
CORE = ["B0", "B1", "B2", "B4", "B5"]
SYSTEM = {"safety.filter": "orca"}
COLORS = {"B0": "#444444", "B1": "#aaaaaa", "B2": "#1f77b4", "B3": "#17becf", "B4": "#d62728", "B5": "#2ca02c"}

# каналы (E3): связный рой (дальность 2000 м), меняется только качество канала
NET = {"kind": "network", "range": 2000, "range_gcs": 2000}
CHANNELS = {
    "ideal": {"kind": "ideal"},
    "delay200+loss10": {**NET, "latency": 0.2, "jitter": 0.05, "loss": 0.1},
    "loss30": {**NET, "latency": 0.05, "jitter": 0.02, "loss": 0.3},
    "gcs_outage30": {**NET, "latency": 0.02, "jitter": 0.01, "loss": 0.0, "gcs_outages": [[150, 350]]},
}
LOSS_CURVE = (0.0, 0.1, 0.2, 0.3, 0.4)


def new_tasks(seed: int, times=(200.0, 400.0), k: int = 3, size: float = 1000.0) -> list[dict]:
    """Нестационарность D: k новых целей в моменты times (≈ 1/3 и 2/3 миссии), своё место на зерно."""
    rng = np.random.default_rng(20_000 + seed)
    out = []
    for t in times:
        for _ in range(k):
            x, y = rng.uniform(0.1 * size, 0.9 * size, 2)
            out.append({"t": t, "x": round(float(x), 1), "y": round(float(y), 1), "priority": 5})
    return out


def disturbed(seed: int) -> dict:
    """Режим «с нарушениями»: 20 % тихих отказов в 60–400 с + новые цели на 1/3 и 2/3 миссии."""
    return {"faults.kind": "random", "faults.rate": 0.2, "faults.t_range": [60, 400], "faults.detect": "silent",
            "world.new_tasks": new_tasks(seed)}


# ------------------------------------------------------------------ план опытов
def plan(seeds: int = 20, quick: bool = False) -> dict[str, list[tuple]]:
    S = 2 if quick else seeds
    S10 = 1 if quick else max(seeds // 2, 1)
    exps: dict[str, list[tuple]] = {}

    def add(exp, cond, m, over, n_seeds):
        for s in range(1, n_seeds + 1):
            o = {**SYSTEM, **METHODS[m], **over}
            if callable(over.get("_dyn")):
                o = {**o, **over["_dyn"](s)}
                o.pop("_dyn", None)
            exps.setdefault(exp, []).append(("baseline", o, s, f"{exp}|{cond}|{m}"))

    # E1: масштаб, базовая линия (A = 1000², κ однородная, λ = 0, идеальный канал, D = 0), M = 25
    for N in (5, 10, 20, 50):
        for m in CORE + (["B3"] if N <= 10 else []):
            add("E1", f"N={N}", m, {"agents.count": N}, S)
    # E1h: то же на мире, который меняется от зерна (3 «горячие» зоны в случайных местах): при однородной
    # карте план не зависит от зерна и 20 зёрен — один и тот же прогон (псевдоповторность, см. P2)
    for N in (5, 10, 20, 50):
        for m in CORE + (["B3"] if N <= 10 else []):
            add("E1h", f"N={N}", m, {"agents.count": N, "world.prior.kind": "hotspots"}, S)
    # E1b: масштаб при большей нагрузке распределения: M = 100 (секторы 100 м)
    for N in (10, 20, 50):
        for m in ("B0", "B2", "B4", "B5"):
            add("E1b", f"N={N}", m, {"agents.count": N, "mission.sector_size": 100}, S10)
    # E2: устойчивость к отказам λ (тихие, 60–400 с)
    for lam in (0.0, 0.1, 0.2, 0.3):
        for m in CORE:
            add("E2", f"λ={lam}", m, {"faults.kind": "random", "faults.rate": lam, "faults.t_range": [60, 400],
                                       "faults.detect": "silent"}, S)
    # E3: качество канала q — в номинале и с нарушениями; кривая потерь — с нарушениями
    for q, comms in CHANNELS.items():
        for m in CORE:
            add("E3", f"{q}|nominal", m, {"comms": comms}, S10)
            add("E3", f"{q}|disturbed", m, {"comms": comms, "_dyn": disturbed}, S)
    for loss in LOSS_CURVE:
        for m in CORE:
            add("E3L", f"loss={loss}", m, {"comms": {**NET, "latency": 0.05, "jitter": 0.02, "loss": loss},
                                           "_dyn": disturbed}, S10)
    # E4: нестационарность D
    for D in (0, 1):
        for m in CORE:
            add("E4", f"D={D}", m, {"_dyn": (lambda s: {"world.new_tasks": new_tasks(s)})} if D else {}, S)
    # E5: неоднородность κ
    for kappa in ("uniform", "hotspots"):
        for m in CORE:
            add("E5", f"κ={kappa}", m, {"world.prior.kind": kappa}, S)
    # Абляции (docs/03 §5 и график 5)
    fl = {"faults.kind": "random", "faults.rate": 0.2, "faults.t_range": [60, 400], "faults.detect": "silent"}
    add("A1", "с перераспределением", "B2", fl, S)
    add("A1", "без перераспределения", "B2", {**fl, "allocator": {"kind": "central_greedy", "dynamic": False}}, S)
    degr = lambda s: {"faults.scheduled": [{"t": 120.0, "agent": s % 10, "type": "degrade", "factor": 0.5},  # noqa: E731
                                           {"t": 120.0, "agent": (s + 5) % 10, "type": "degrade", "factor": 0.5}]}
    out = {"comms": {**NET, "latency": 0.05, "jitter": 0.02, "loss": 0.1, "gcs_outages": [[100, 400]]}}
    add("A2", "с локальным ремонтом", "B5", {**out, "agents.local_repair": True, "_dyn": degr}, S)
    add("A2", "без локального ремонта", "B5", {**out, "agents.local_repair": False, "_dyn": degr}, S)
    hs = {"world.prior.kind": "hotspots"}
    add("A3", "Pd-планирование", "B2", {**hs, "planning.value": "prior"}, S)
    add("A3", "по покрытию", "B2", {**hs, "planning.value": "uniform"}, S)
    for layers in (10, 1):
        for f in ("orca", "none"):
            add("A4", f"эшелоны={layers}|{f}", "B2", {"agents.count": 20, "agents.altitude_layers": layers,
                                                     "safety.filter": f, "world.prior.kind": "hotspots"}, S)
    return exps


# ------------------------------------------------------------------ прогон
def _job(job: tuple) -> dict:
    from swarmsim import apply_overrides, load_scenario, run
    scen, over, seed, label = job
    scn = apply_overrides(load_scenario(ROOT / "scenarios" / f"{scen}.yaml"), {"output.trajectories": False, **over})
    s = run(scn, seed).summary
    parts = label.split("|")
    exp, cond, m = parts[0], "|".join(parts[1:-1]), parts[-1]
    return {"exp": exp, "cond": cond, "method": m, "seed": seed, **s}


def _pool(jobs_list, jobs):
    from concurrent.futures import ProcessPoolExecutor
    if jobs <= 1:
        return [_job(j) for j in jobs_list]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(_job, jobs_list, chunksize=1))


def _cast(v: str):
    if v in ("", "None"):
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except ValueError:
        return v


def load(exp: str) -> list[dict] | None:
    p = DATA / f"P5_{exp}.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return [{k: _cast(v) for k, v in r.items()} for r in csv.DictReader(f)]


def run_campaign(jobs: int, resume: bool = False, quick: bool = False, only: list[str] | None = None) -> dict:
    from swarmsim.io import write_csv
    DATA.mkdir(parents=True, exist_ok=True)
    walls = {}
    for exp, jl in plan(quick=quick).items():
        if only and exp not in only:
            continue
        if resume and load(exp) is not None:
            continue
        t = time.perf_counter()
        rows = _pool(jl, jobs)
        walls[exp] = time.perf_counter() - t
        write_csv(DATA / f"P5_{exp}.csv", rows)
        print(f"{exp}: {len(rows)} прогонов, {walls[exp] / 60:.1f} мин", flush=True)
    meta = DATA / "walls.json"
    old = json.loads(meta.read_text()) if meta.exists() else {}
    meta.write_text(json.dumps({**old, **walls}, indent=1))
    return walls


# ------------------------------------------------------------------ анализ
def _by(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def _series(rows, cond, m, key, valid_only=True):
    rs = sorted(_by(rows, cond=cond, method=m), key=lambda r: r["seed"])
    return {r["seed"]: (r.get(key) if (r["valid"] or not valid_only) else None) for r in rs}


def _pair(rows, cond, a, b, key, cond_b=None):
    xa = _series(rows, cond, a, key)
    xb = _series(rows, cond_b or cond, b, key)
    seeds = sorted(set(xa) & set(xb))
    return paired([xa[s] for s in seeds], [xb[s] for s in seeds])


def _f(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{nd}f} %" if pct else f"{x:.{nd}f}"


def _p(p, det=False):
    if det:
        return "детерм."
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def _table(rows, conds, methods, key, label, nd=2):
    L = [f"**{label}** — mean [CI95 бутстреп] · median (IQR) · n валидных / невалидных", "",
         "| условие | " + " | ".join(NAMES[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    for c in conds:
        cells = []
        for m in methods:
            rs = _by(rows, cond=c, method=m)
            if not rs:
                cells.append("—")
                continue
            d = describe([r.get(key) for r in rs if r["valid"]])
            inv = sum(not r["valid"] for r in rs)
            cells.append(f"{_f(d['mean'], nd)} [{_f(d['lo'], nd)}; {_f(d['hi'], nd)}] · {_f(d['median'], nd)} "
                         f"({_f(d['q1'], nd)}–{_f(d['q3'], nd)}) · {d['n']}/{inv}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    return L + [""]


def _cmp_table(rows, conds, pairs_, key, label):
    L = [f"**{label}: парные сравнения** (d = первый − второй; Уилкоксон; r_rb — величина эффекта)", "",
         "| условие | сравнение | n | средняя d [CI95] | отн. | p | r_rb (эффект) |", "|---|---|---|---|---|---|---|"]
    for c in conds:
        for a, b in pairs_:
            if not _by(rows, cond=c, method=a) or not _by(rows, cond=c, method=b):
                continue
            r = _pair(rows, c, a, b, key)
            L.append(f"| {c} | {a} − {b} | {r['n']} | {_f(r['diff'], 3)} [{_f(r['lo'], 3)}; {_f(r['hi'], 3)}] | "
                     f"{_f(r['rel'], 1, True)} | {_p(r['p'], r.get('det'))} | "
                     + ("n_eff = 1 |" if r.get("det") else f"{_f(r['r_rb'])} ({effect_word(r['r_rb'])}) |"))
    return L + ["", "_«детерм.» — все пары дают одну и ту же разность: мир и метод не зависят от зерна, "
                "эффективный объём выборки 1; разность точная для этого мира, но обобщать её нельзя._", ""]


def _mean(rows, cond, m, key):
    xs = [r.get(key) for r in _by(rows, cond=cond, method=m) if r["valid"] and r.get(key) is not None]
    return statistics.fmean(xs) if xs else math.nan


def _p_complete(rows, cond, m, thr=0.95):
    rs = _by(rows, cond=cond, method=m)
    return sum(1 for r in rs if r["valid"] and r["CR"] >= thr) / len(rs) if rs else math.nan


# ------------------------------------------------------------------ графики
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ci_of(rows, cond, m, key, norm=None):
    vals = []
    for r in _by(rows, cond=cond, method=m):
        if not r["valid"] or r.get(key) is None:
            continue
        v = r[key]
        if norm is not None:
            b = norm.get(r["seed"])
            if b in (None, 0):
                continue
            v = v / b
        vals.append(v)
    d = describe(vals)
    return d["mean"], max(d["mean"] - d["lo"], 0.0), max(d["hi"] - d["mean"], 0.0)


def plot1(e1, e1h, e1b):
    plt = _plt()
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    for k, (rows, ns, title) in enumerate([(e1, (5, 10, 20, 50), "M = 25, однородная карта"),
                                           (e1h, (5, 10, 20, 50), "M = 25, 3 горячие зоны"),
                                           (e1b, (10, 20, 50), "M = 100, однородная")]):
        if not rows:
            continue
        for m in ("B0", "B1", "B2", "B3", "B4", "B5"):
            xs, ys, lo, hi = [], [], [], []
            for N in ns:
                c = f"N={N}"
                base = {r["seed"]: r["U"] for r in _by(rows, cond=c, method="B0") if r["valid"]}
                if not _by(rows, cond=c, method=m):
                    continue
                mu, a, b = _ci_of(rows, c, m, "U", norm=base)
                xs.append(N); ys.append(mu); lo.append(a); hi.append(b)
            if xs:
                ax[k].errorbar(xs, ys, yerr=[lo, hi], fmt="o-", capsize=3, color=COLORS[m], label=NAMES[m], ms=4)
        ax[k].set_xscale("log")
        ax[k].set_xticks(ns, [str(n) for n in ns])
        ax[k].set_xlabel("N агентов")
        ax[k].set_ylabel("U(M) / U(B0)")
        ax[k].set_title(f"1. Пересечение по масштабу: {title}")
        ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    # вычисления: центр — время одного пересчёта; CBBA — один пересчёт пакета на борту
    rows = e1b or e1
    ns = (10, 20, 50) if e1b else (5, 10, 20, 50)
    for m, key, lab in (("B2", "alloc_ms_max", "B2: пересчёт на станции"), ("B0", "alloc_ms_max", "B0: офлайн-оракул"),
                        ("B4", "node_ms_max", "B4: пересчёт пакета на борту (макс)"),
                        ("B5", "alloc_ms_max", "B5: пересчёт на станции")):
        ys = [_mean(rows, f"N={N}", m, key) for N in ns]
        ax[3].plot(ns, ys, "o-", color=COLORS[m], label=lab, ms=4)
    ax[3].set_xscale("log"); ax[3].set_yscale("log")
    ax[3].set_xticks(ns, [str(n) for n in ns])
    ax[3].set_xlabel("N агентов"); ax[3].set_ylabel("мс (настенное время)")
    ax[3].set_title("T_compute" + (" (M = 100)" if e1b else ""))
    ax[3].legend(fontsize=7); ax[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "P5_1_crossover.png", dpi=130)
    plt.close(fig)


def plot2(e2):
    plt = _plt()
    lams = (0.0, 0.1, 0.2, 0.3)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for m in CORE:
        ax[0].plot(lams, [_p_complete(e2, f"λ={x}", m) for x in lams], "o-", color=COLORS[m], label=NAMES[m])
        mu = [_ci_of(e2, f"λ={x}", m, "U") for x in lams]
        ax[1].errorbar(lams, [a for a, _, _ in mu], yerr=[[b for _, b, _ in mu], [c for _, _, c in mu]], fmt="o-",
                       capsize=3, color=COLORS[m], label=NAMES[m])
    ax[0].set_xlabel("λ — доля отказавших агентов"); ax[0].set_ylabel("P(mission complete) = P(CR ≥ 95 %)")
    ax[1].set_xlabel("λ"); ax[1].set_ylabel("U(M)")
    ax[0].set_title("2. Кривая устойчивости"); ax[1].set_title("U(M) при отказах")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "P5_2_resilience.png", dpi=130)
    plt.close(fig)


def plot3(e3, e3l):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    if e3l:
        for m in CORE:
            mu = [_ci_of(e3l, f"loss={q}", m, "U") for q in LOSS_CURVE]
            ax[0].errorbar([100 * q for q in LOSS_CURVE], [a for a, _, _ in mu],
                           yerr=[[b for _, b, _ in mu], [c for _, _, c in mu]], fmt="o-", capsize=3,
                           color=COLORS[m], label=NAMES[m])
        ax[0].set_xlabel("потери пакетов, %"); ax[0].set_ylabel("U(M)")
        ax[0].set_title("3. Деградация канала (с нарушениями)")
        ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    qs = list(CHANNELS)
    x = np.arange(len(qs))
    w = 0.16
    for i, m in enumerate(CORE):
        mu = [_ci_of(e3, f"{q}|disturbed", m, "U") for q in qs]
        ax[1].bar(x + (i - 2) * w, [a for a, _, _ in mu], w, yerr=[[b for _, b, _ in mu], [c for _, _, c in mu]],
                  color=COLORS[m], label=NAMES[m], capsize=2)
    ax[1].set_xticks(x, qs, fontsize=8)
    ax[1].set_ylabel("U(M)")
    ax[1].set_title("уровни q (с нарушениями), вкл. разрыв со станцией 30 %")
    lo = min(_mean(e3, f"{q}|disturbed", m, "U") for q in qs for m in CORE)
    ax[1].set_ylim(lo * 0.9, None)
    fig.tight_layout()
    fig.savefig(FIG / "P5_3_channel.png", dpi=130)
    plt.close(fig)


def plot4(e3, e1b):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for k, (rows, cond, title) in enumerate([(e3, "delay200+loss10|disturbed", "N = 10, канал 200 мс + 10 %, с нарушениями"),
                                             (e1b, "N=50", "N = 50, M = 100, номинал")]):
        if not rows:
            continue
        for m in CORE:
            if not _by(rows, cond=cond, method=m):
                continue
            U = _mean(rows, cond, m, "U")
            B = _mean(rows, cond, m, "B_per_agent")
            comp = _mean(rows, cond, m, "node_ms_max" if m == "B4" else "alloc_ms_max")
            if math.isnan(B):
                B = 0.0
            s = 30 + 25 * math.log10(1 + (comp if not math.isnan(comp) else 0))
            ax[k].scatter(B, U, s=s * 3, color=COLORS[m], alpha=0.8, edgecolor="k")
            ax[k].annotate(f"{m}\n{comp:.0f} мс" if not math.isnan(comp) else m, (B, U), fontsize=8,
                           xytext=(6, 4), textcoords="offset points")
        ax[k].set_xlabel("B — трафик, байт/с на агента (0 — канал не моделировался)")
        ax[k].set_ylabel("U(M)")
        ax[k].set_title(f"4. Парето: качество / трафик / вычисления\n{title}", fontsize=9)
        ax[k].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "P5_4_pareto.png", dpi=130)
    plt.close(fig)


def ablations(a):
    """[(имя, модуль, условие «с», условие «без», метод, метрика, лучше_больше)] → парные разности."""
    spec = [("A1", "перераспределение", "с перераспределением", "без перераспределения", "B2", "U", True),
            ("A2", "локальный ремонт", "с локальным ремонтом", "без локального ремонта", "B5", "U", True),
            ("A3", "Pd-планирование", "Pd-планирование", "по покрытию", "B2", "U", True),
            ("A3", "Pd-планирование (T_detect 90 %)", "Pd-планирование", "по покрытию", "B2", "T_detect_90", False),
            ("A4", "тактическое избегание при эшелонах", "эшелоны=10|orca", "эшелоны=10|none", "B2", "U", True),
            ("A4", "тактическое избегание без эшелонов", "эшелоны=1|orca", "эшелоны=1|none", "B2", "U", True)]
    out = []
    for exp, name, c_on, c_off, m, key, higher in spec:
        rows = a.get(exp)
        if not rows:
            continue
        on = {r["seed"]: r for r in _by(rows, cond=c_on, method=m)}
        off = {r["seed"]: r for r in _by(rows, cond=c_off, method=m)}
        seeds = sorted(set(on) & set(off))
        # для A4 невалидные (столкновения) — это и есть эффект: сравниваем по всем прогонам
        keep_all = exp == "A4"
        xs = [on[s].get(key) if (keep_all or on[s]["valid"]) else None for s in seeds]
        ys = [off[s].get(key) if (keep_all or off[s]["valid"]) else None for s in seeds]
        r = paired(xs, ys)
        out.append({"exp": exp, "name": name, "key": key, "higher": higher, **r,
                    "inv_on": sum(not on[s]["valid"] for s in seeds), "inv_off": sum(not off[s]["valid"] for s in seeds),
                    "col_on": sum(on[s]["N_col"] for s in seeds), "col_off": sum(off[s]["N_col"] for s in seeds),
                    "nm_on": sum(on[s]["N_nm"] for s in seeds), "nm_off": sum(off[s]["N_nm"] for s in seeds)})
    return out


def plot5(ab):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    rows = [r for r in ab if r["key"] == "U"]
    y = np.arange(len(rows))
    ax[0].barh(y, [r["diff"] for r in rows], xerr=[[r["diff"] - r["lo"] for r in rows], [r["hi"] - r["diff"] for r in rows]],
               color=["#2ca02c" if r["diff"] >= 0 else "#d62728" for r in rows], capsize=3)
    ax[0].set_yticks(y, [f"{r['exp']}: {r['name']}" for r in rows], fontsize=8)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("ΔU(M) = с модулем − без (среднее, CI95)")
    ax[0].set_title("5. Вклад модулей (метод исключения)")
    a4 = [r for r in ab if r["exp"] == "A4" and r["key"] == "U"]
    if a4:
        lab = ["эшелоны", "один эшелон"]
        x = np.arange(len(a4))
        ax[1].bar(x - 0.2, [r["col_off"] for r in a4], 0.4, label="без ORCA: столкновения", color="#d62728")
        ax[1].bar(x + 0.2, [r["col_on"] for r in a4], 0.4, label="с ORCA: столкновения", color="#2ca02c")
        ax[1].set_xticks(x, lab[:len(a4)])
        ax[1].set_ylabel("N_col (сумма по зёрнам)")
        ax[1].set_title("A4: тактическое избегание (N = 20)")
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "P5_5_ablation.png", dpi=130)
    plt.close(fig)


def plot6(seed: int = 1):
    """Карта миссии: траектории, места отказов, новые цели, интервал разрыва связи со станцией."""
    from swarmsim import apply_overrides, load_scenario, run
    outage = [150.0, 350.0]
    over = {**SYSTEM, **METHODS["B5"], **disturbed(seed), "output.trajectories": True,
            "comms": {**NET, "latency": 0.05, "jitter": 0.02, "loss": 0.1, "gcs_outages": [outage]}}
    res = run(apply_overrides(load_scenario(ROOT / "scenarios" / "baseline.yaml"), over), seed)
    plt = _plt()
    fig = plt.figure(figsize=(14, 6.4))
    ax = fig.add_subplot(1, 2, 1)
    tr = res.trajectories
    n = max(r["agent"] for r in tr) + 1
    cmap = plt.get_cmap("tab10")
    fail_t = {e["agent"]: e["t"] for e in res.events if e["type"] == "agent_failed"}
    for i in range(n):
        pts = [(r["x"], r["y"], r["t"]) for r in tr if r["agent"] == i]
        xs, ys, ts = zip(*pts)
        ax.plot(xs, ys, lw=0.8, color=cmap(i % 10), label=f"БПЛА {i}")
        ins = [(x, y) for x, y, t in pts if outage[0] <= t < outage[1]]
        if ins:
            ax.plot(*zip(*ins), lw=2.2, color=cmap(i % 10), alpha=0.35)
        if i in fail_t:
            k = min(range(len(ts)), key=lambda j: abs(ts[j] - fail_t[i]))
            ax.plot(xs[k], ys[k], "kx", ms=12, mew=3)
            ax.annotate(f"отказ {fail_t[i]:.0f} с", (xs[k], ys[k]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    for e in res.events:
        if e["type"] == "new_task":
            ax.plot(e["x"], e["y"], "*", color="gold", ms=16, mec="k")
            ax.annotate(f"новая цель {e['t']:.0f} с", (e["x"], e["y"]), fontsize=7, xytext=(6, -10),
                        textcoords="offset points")
    ax.set_aspect("equal"); ax.set_xlim(-20, 1020); ax.set_ylim(-40, 1020)
    ax.set_title("6. Карта миссии (B5 hybrid): жирно — полёт во время разрыва со станцией")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax2 = fig.add_subplot(1, 2, 2)
    done = sorted(e["t"] for e in res.events if e["type"] == "task_done")
    ax2.step(done, range(1, len(done) + 1), where="post", color="k", label="выполнено задач")
    ax2.axvspan(*outage, color="#d62728", alpha=0.15, label="нет связи со станцией")
    for i, t in fail_t.items():
        ax2.axvline(t, color="k", ls=":", lw=1)
        ax2.annotate(f"отказ {i}", (t, 1), rotation=90, fontsize=7)
    for e in res.events:
        if e["type"] == "new_task":
            ax2.axvline(e["t"], color="gold", lw=1)
    ax2.set_xlabel("t, с"); ax2.set_ylabel("задач выполнено")
    ax2.set_title(f"ход миссии: CR {100 * res.summary['CR']:.0f} %, N_dup {res.summary['N_dup']}, "
                  f"N_lost {res.summary.get('N_lost')}")
    ax2.legend(fontsize=8, loc="lower right"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "P5_6_mission_map.png", dpi=130)
    plt.close(fig)
    return res.summary


# ------------------------------------------------------------------ отчёт
def _notes(e1, e1b, V) -> list[str]:
    """Замечания к интерпретации — написаны после прогона; метрики и правила вердиктов не менялись."""
    L = ["", "## Замечания к интерпретации (добавлены после прогона; метрики и правила вердиктов не менялись)", ""]
    if e1:
        mk = {m: _mean(e1, "N=5", m, "makespan") for m in ("B0", "B1", "B2")}
        L.append(f"* **N = 5, «B1 лучше оракула»:** миссия длится дольше срока U (900 с): makespan B1 {mk['B1']:.0f} с, "
                 f"B2 {mk['B2']:.0f} с, B0 {mk['B0']:.0f} с. Планировщики оптимизируют гладкий дисконт λ^C "
                 "(сначала ценное), а U штрафует ступенькой «после 900 с — ноль»; равномерный B1 случайно укладывает всё "
                 "в срок. Это расхождение целевой функции планирования и метрики оценки (тот же обрыв, что в P2), "
                 "а не превосходство B1. При N ≥ 10 миссия короче срока, и эффект исчезает.")
    L.append("* **H3 «да» по формальному правилу**, но величина эффекта практически нулевая: во всех ячейках E1/E1h/E1b "
             "разница U между B2 и B4 меньше 1 %, а время пересчёта станции не превышает сотни миллисекунд даже при "
             "N = 50, M = 100. Честный вывод: в исследованном диапазоне точки перехода по качеству нет — B4 и B2 "
             "практически равны; различие — в трафике, вычислениях и поведении без станции.")
    L.append("* **H1 «нет», H2 «нет»**: централизованный B2 не лучше CBBA в номинале (разрыв < 1 %, критерий остановки №1), "
             "а в связном рое (дальность 2 км) B2 при потерях и разрыве станции тоже не проигрывает статистически "
             "значимо: повтор доставки плана и 20 % отказов при избыточной ёмкости роя отыгрываются к концу миссии. "
             "Выгода децентрализации видна во времени восстановления (P4: B2 ждёт конца разрыва — T_recovery ~150 с, "
             "B4/B5 — 1.5 с), но в U за миссию она почти не проявляется. Для U нужен сценарий, где задержка стоит "
             "дорого (срочные цели, длинный разрыв, меньшая ёмкость).")
    L.append("* **Практический вывод для бортовой реализации:** B5 (гибрид) не хуже B2 в номинале, не хуже B4 при "
             "потере станции, и тратит меньше трафика, чем B4. Он — рабочая архитектура для переноса на компаньон-компьютер.")
    return L


def report(stamp: str) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    D = {e: load(e) for e in ("E1", "E1h", "E1b", "E2", "E3", "E3L", "E4", "E5", "A1", "A2", "A3", "A4")}
    missing = [k for k, v in D.items() if v is None]
    walls = json.loads((DATA / "walls.json").read_text()) if (DATA / "walls.json").exists() else {}
    e1, e1h, e1b, e2, e3, e3l, e4, e5 = (D[k] or [] for k in ("E1", "E1h", "E1b", "E2", "E3", "E3L", "E4", "E5"))
    ab = ablations(D)
    plots = {}
    for name, fn in (("1", lambda: plot1(e1, e1h, e1b)), ("2", lambda: plot2(e2)), ("3", lambda: plot3(e3, e3l)),
                     ("4", lambda: plot4(e3, e1b)), ("5", lambda: plot5(ab)), ("6", plot6)):
        try:
            fn()
            plots[name] = True
        except Exception as exc:  # график не должен ронять весь отчёт — но факт фиксируем
            plots[name] = f"ошибка: {exc!r}"
    all_rows = [r for v in D.values() if v for r in v]
    n_runs = len(all_rows)
    # жёсткие ограничения — по всем прогонам, кроме плеча A4 «один эшелон без ORCA» (там столкновения — измеряемый эффект)
    gate_rows = [r for r in all_rows if not (r["exp"] == "A4" and r["cond"] == "эшелоны=1|none")]
    n_col = sum(r["N_col"] for r in gate_rows)
    n_dup = sum(r["N_dup"] for r in gate_rows)
    n_invalid = sum(not r["valid"] for r in gate_rows)
    passed = not missing and all(v is True for v in plots.values()) and n_col == 0

    # ---------------- вердикты по гипотезам (правила зафиксированы здесь до прогона)
    V = {}
    # H1: B2 vs B4, идеальный канал, N ≤ 20. Статистический вывод — по E1h (мир меняется от зерна);
    # E1 (однородная карта) детерминирован и приводится как точная разность для одного мира.
    src1 = e1h or e1
    h1 = {N: _pair(src1, f"N={N}", "B2", "B4", "U") for N in (5, 10, 20)} if src1 else {}
    if h1:
        gaps = {N: r["rel"] for N, r in h1.items()}
        sig = {N: (not r.get("det")) and r["p"] < 0.05 and r["diff"] > 0 for N, r in h1.items()}
        V["H1"] = ("да" if all(sig.values()) else "частично" if any(sig.values()) else "нет",
                   ("E1h: " if e1h else "E1: ")
                   + "; ".join(f"N={N}: U(B2)−U(B4) = {_f(h1[N]['diff'], 3)} ({_f(gaps[N], 2, True)}), "
                               f"p {_p(h1[N]['p'], h1[N].get('det'))}, r_rb {_f(h1[N]['r_rb'])}" for N in h1)
                   + ("; однородная карта (E1, детерм.): " + ", ".join(
                       f"N={N} {_f(_pair(e1, f'N={N}', 'B2', 'B4', 'U')['rel'], 2, True)}" for N in (5, 10, 20)) if e1 and e1h else "")
                   + (". Разрыв ≤ 5 % → децентрализация «почти бесплатна» (критерий остановки №1 сработал)"
                      if all(abs(g) <= 0.05 for g in gaps.values()) else ""))
    # H2: B4/B5 vs B2 при потерях ≥ 30 % и разрыве станции (с нарушениями)
    if e3:
        parts, wins = [], []
        for q in ("loss30", "gcs_outage30"):
            for m in ("B4", "B5"):
                r = _pair(e3, f"{q}|disturbed", m, "B2", "U")
                wins.append((not r.get("det")) and r["p"] < 0.05 and r["diff"] > 0)
                parts.append(f"{q}: U({m})−U(B2) = {_f(r['diff'], 3)} [{_f(r['lo'], 3)}; {_f(r['hi'], 3)}], p {_p(r['p'], r.get('det'))}, "
                             f"r_rb {_f(r['r_rb'])}")
        V["H2"] = ("да" if all(wins) else "частично" if any(wins) else "нет", "; ".join(parts))
    # H3: точка перехода N*: наименьшее N, где U(B4) ≥ U(B2) (среднее, M = 100 и M = 25)
    if e1 or e1h or e1b:
        txt = []
        verdict = "нет"
        for rows, ns, tag in ((e1h, (5, 10, 20, 50), "M=25, горячие зоны"), (e1, (5, 10, 20, 50), "M=25, однородная"),
                              (e1b, (10, 20, 50), "M=100")):
            if not rows:
                continue
            nstar = next((N for N in ns if _mean(rows, f"N={N}", "B4", "U") >= _mean(rows, f"N={N}", "B2", "U")), None)
            rr = _pair(rows, f"N={ns[-1]}", "B4", "B2", "U")
            txt.append(f"{tag}: N* = {nstar if nstar else 'не найдено'}; при N={ns[-1]} U(B4)−U(B2) = {_f(rr['diff'], 3)} "
                       f"({_f(rr['rel'], 2, True)}), p {_p(rr['p'], rr.get('det'))}; T_compute B2 {_f(_mean(rows, f'N={ns[-1]}', 'B2', 'alloc_ms_max'), 0)} мс "
                       f"vs бортовой пересчёт B4 {_f(_mean(rows, f'N={ns[-1]}', 'B4', 'node_ms_max'), 1)} мс")
            if nstar is not None and nstar > ns[0]:
                verdict = "да"
            elif nstar == ns[0] and verdict != "да":
                verdict = "B4 не хуже с самого малого N"
        V["H3"] = (verdict, "; ".join(txt))
    # H4: P(complete) при λ = 0.3 у B2/B4/B5 ≥ 0.95
    if e2:
        pc = {m: _p_complete(e2, "λ=0.3", m) for m in CORE}
        ok4 = all(pc[m] >= 0.95 for m in ("B2", "B4", "B5"))
        V["H4"] = ("да" if ok4 else "нет", ", ".join(f"{m}: {_f(pc[m], 0, True)}" for m in CORE)
                   + f"; T_recovery макс (λ=0.3): " + ", ".join(
                       f"{m} {_f(max([r['T_recovery_max'] for r in _by(e2, cond='λ=0.3', method=m) if r.get('T_recovery_max') is not None] or [math.nan]))} с"
                       for m in ("B2", "B4", "B5")))
    # H5: B0 лучше роя в номинале и хуже при нарушениях
    if (e1h or e1) and e2:
        nom = {m: _pair(e1h or e1, "N=10", "B0", m, "U") for m in ("B2", "B4", "B5")}
        dis = {m: _pair(e2, "λ=0.2", "B0", m, "U") for m in ("B2", "B4", "B5")}
        better_nom = all(r["diff"] > 0 for r in nom.values())
        worse_dis = all(r["diff"] < 0 and (not r.get("det")) and r["p"] < 0.05 for r in dis.values())
        V["H5"] = ("да" if better_nom and worse_dis else "частично" if (better_nom or worse_dis) else "нет",
                   f"номинал ({'E1h' if e1h else 'E1'}, N=10): " + ", ".join(
                       f"U(B0)−U({m}) = {_f(r['diff'], 3)} p {_p(r['p'], r.get('det'))}" for m, r in nom.items())
                   + "; отказы λ=0.2 (E2): " + ", ".join(
                       f"U(B0)−U({m}) = {_f(r['diff'], 3)} p {_p(r['p'], r.get('det'))}" for m, r in dis.items()))
    # H6: тактическое избегание не даёт выигрыша при корректной стратегической деконфликтуации
    a4 = {r["name"]: r for r in ab if r["exp"] == "A4"}
    if a4:
        lay = a4.get("тактическое избегание при эшелонах")
        one = a4.get("тактическое избегание без эшелонов")
        no_gain = lay and lay["col_off"] == 0 and (lay["p"] >= 0.05 or abs(lay["rel"]) < 0.01)
        V["H6"] = ("да (выигрыша нет)" if no_gain else "нет",
                   (f"с эшелонами: столкновений без ORCA {lay['col_off']}, с ORCA {lay['col_on']}; ΔU {_f(lay['diff'], 4)} "
                    f"({_f(lay['rel'], 2, True)}), p {_p(lay['p'], lay.get('det'))}. " if lay else "")
                   + (f"Один эшелон: без ORCA {one['col_off']} столкновений ({one['inv_off']} невалидных прогонов), с ORCA "
                      f"{one['col_on']}; цена ΔU {_f(one['diff'], 4)} ({_f(one['rel'], 2, True)})." if one else ""))
    V["H7"] = ("не проверялась", "ML — фаза P8 (после P5), в эту кампанию не входит")

    # ---------------- критерии остановки (docs/03 §9)
    stops = []
    if h1 and all(abs(r["rel"]) <= 0.05 for r in h1.values()):
        stops.append("№1: B4 в пределах 5 % от B2 при идеальном канале и N ≤ 20 → вклад переформулировать: "
                     "система сразу децентрализованная, центр — для оператора и мониторинга.")
    if e1:
        t20 = max([r.get("alloc_ms_max") or 0 for r in _by(e1, cond="N=20", method="B2")] or [0])
        stops.append(f"№2: T_realloc при N = 20: макс пересчёт B2 {t20:.0f} мс — "
                     + ("больше 10 с, критерий сработал." if t20 > 10_000 else "меньше 10 с, критерий не сработал."))

    L = ["# P5 — Сравнение архитектур: отчёт кампании", "",
         f"_Сгенерировано `python -m experiments.gates p5` · {stamp} · {n_runs} прогонов"
         + (f" · {sum(walls.values()) / 60:.0f} мин счёта" if walls else "") + "._", "",
         "## Критерий (docs/05, P5)", "Отчёт с графиками 1–6 из `docs/03`, статистика, 0 столкновений.", "",
         f"## Итог: **{'ПРОЙДЕНА' if passed else 'НЕ ПРОЙДЕНА'}**", "",
         f"* графики 1–6: " + ", ".join(f"{k} {'✅' if v is True else '❌ ' + str(v)}" for k, v in plots.items()),
         f"* столкновения во всех прогонах системы: **{n_col}** {'✅' if n_col == 0 else '❌'} "
         "(не считая плеча A4 «один эшелон без ORCA», где столкновения — измеряемая величина)",
         f"* N_dup во всех прогонах системы: {n_dup}; невалидных прогонов: {n_invalid}",
         f"* статистика: парный Уилкоксон, бутстреп CI95 (1000), r_rb, median/IQR — ниже по каждому опыту",
         *( [f"* **не хватает данных:** {', '.join(missing)}"] if missing else []), "",
         "## Вердикты по гипотезам (правила вердиктов зафиксированы в `experiments/campaign.py` до прогона)", "",
         "| H | вердикт | числа |", "|---|---|---|"]
    for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7"):
        if h in V:
            L.append(f"| {h} | **{V[h][0]}** | {V[h][1]} |")
    L += ["", "## Критерии остановки (docs/03 §9)", *[f"* {s}" for s in stops], "",
          "## Что знает каждая архитектура (docs/03 §8, правило 2)", "",
          "| метод | знание мира при планировании | связь | перераспределение |", "|---|---|---|---|",
          "| B0 static_oracle | **истинные** позиции целей (оракул, нечестное преимущество — это ориентир) | не нужна | нет |",
          "| B1 static_roundrobin | только геометрия секторов | не нужна | нет |",
          "| B2 central_greedy | априорная карта + состояние агентов **глазами станции** (что дошло по каналу) | агент↔станция | по событиям, на станции |",
          "| B3 central_optimal | как B2 | как B2 | как B2 (LNS/CP-SAT) |",
          "| B4 cbba | априорная карта + своя реплика (heartbeat/gossip/заявки соседей) | агент↔агент | непрерывно, на борту |",
          "| B5 hybrid | B2 при связи со станцией; без неё — реплика и CBBA среди «сирот» | оба | станция + борт |",
          "", "Во всех методах одна целевая функция планирования (дисконт λ^C по ценности сектора), одна проверка "
          "плана (энергия, геозона) и одинаковый бюджет настройки: параметры по умолчанию, подобранные на P2–P4 "
          "до этой кампании; в кампании ничего не настраивалось.", "",
          "## Дизайн (последовательное планирование, docs/03 §3)", "",
          "* базовая линия: A = 1000×1000 м, κ однородная, λ = 0, канал идеальный, D = 0, M = 25 секторов 200×200 м;",
          "* E1 — N ∈ {5, 10, 20, 50}, B0 B1 B2 B4 B5 (+ B3 при N ≤ 10), 20 зёрен;",
          "* E1b — нагрузка: M = 100 (секторы 100 м), N ∈ {10, 20, 50}, B0 B2 B4 B5, 10 зёрен;",
          "* E2 — λ ∈ {0, 0.1, 0.2, 0.3} тихих отказов в 60–400 с, 20 зёрен;",
          "* E3 — q ∈ {идеальный, 200 мс + 10 %, потери 30 %, разрыв со станцией 30 % (150–350 с)} в номинале (10 зёрен) "
          "и с нарушениями (λ = 0.2 + новые цели, 20 зёрен); E3L — кривая потерь 0–40 % с нарушениями, 10 зёрен;",
          "* E4 — D ∈ {0; по 3 новые цели на 200 и 400 с}; E5 — κ ∈ {однородная, 3 «горячие» зоны}; 20 зёрен;",
          "* A1–A4 — методы исключения (см. раздел «Абляции»), 20 зёрен;",
          "* «с нарушениями» = 20 % тихих отказов в 60–400 с + по 3 новые цели на 200 и 400 с (≈ 1/3 и 2/3 миссии);",
          "* система: эшелоны по высоте + ORCA (A4 проверяет, нужно ли второе).",
          "* **P(mission complete)** = доля прогонов с CR ≥ 95 % (порог P3). Невалидные прогоны (N_col, N_dup, геозона) "
          "исключаются из статистики качества и показаны отдельно (n валидных / невалидных).", ""]

    def section(title, rows, conds, methods, keys, cmp_pairs, img=None):
        nonlocal L
        if not rows:
            L += [f"## {title}", "", "_нет данных_", ""]
            return
        L += [f"## {title}", ""]
        if img:
            L += [f"![{title}](figures/{img})", ""]
        for key, lab, nd in keys:
            L += _table(rows, conds, methods, key, lab, nd)
        for key, lab in cmp_pairs[1]:
            L += _cmp_table(rows, conds, cmp_pairs[0], key, lab)

    ms = CORE + ["B3"]
    section("E1. Масштаб (график 1)", e1, [f"N={N}" for N in (5, 10, 20, 50)], ms,
            [("U", "U(M)", 3), ("CR", "CR", 3), ("makespan", "makespan, с", 0), ("alloc_ms_max", "T_compute станции, мс", 1),
             ("node_ms_max", "T_compute борта (CBBA), мс", 2)],
            ([("B2", "B4"), ("B5", "B2"), ("B0", "B2"), ("B3", "B2")], [("U", "U(M)")]), "P5_1_crossover.png")
    section("E1h. Масштаб на мире с горячими зонами (меняется от зерна)", e1h, [f"N={N}" for N in (5, 10, 20, 50)], ms,
            [("U", "U(M)", 3), ("CR", "CR", 3), ("makespan", "makespan, с", 0), ("T_detect_90", "T_detect(90 %), с", 0)],
            ([("B2", "B4"), ("B5", "B2"), ("B0", "B2"), ("B3", "B2")], [("U", "U(M)"), ("T_detect_90", "T_detect_90")]))
    section("E1b. Масштаб при M = 100", e1b, [f"N={N}" for N in (10, 20, 50)], ["B0", "B2", "B4", "B5"],
            [("U", "U(M)", 3), ("makespan", "makespan, с", 1), ("alloc_ms_max", "T_compute станции, мс", 0),
             ("node_ms_max", "T_compute борта (CBBA), мс", 2), ("B_per_agent", "B, байт/с на агента", 0)],
            ([("B4", "B2"), ("B5", "B2"), ("B0", "B2")], [("U", "U(M)"), ("makespan", "makespan")]))
    section("E2. Отказы (график 2)", e2, [f"λ={x}" for x in (0.0, 0.1, 0.2, 0.3)], CORE,
            [("U", "U(M)", 3), ("CR", "CR", 3), ("T_recovery_max", "T_recovery макс, с", 2)],
            ([("B2", "B0"), ("B4", "B0"), ("B5", "B0"), ("B4", "B2")], [("U", "U(M)"), ("CR", "CR")]), "P5_2_resilience.png")
    if e2:
        L += ["**P(mission complete)** = P(CR ≥ 95 %):", "", "| λ | " + " | ".join(NAMES[m] for m in CORE) + " |",
              "|---|" + "---|" * len(CORE)]
        for x in (0.0, 0.1, 0.2, 0.3):
            L.append(f"| {x} | " + " | ".join(_f(_p_complete(e2, f'λ={x}', m), 0, True) for m in CORE) + " |")
        L.append("")
    qc = [f"{q}|{r}" for r in ("nominal", "disturbed") for q in CHANNELS]
    section("E3. Канал связи (график 3)", e3, qc, CORE,
            [("U", "U(M)", 3), ("CR", "CR", 3), ("A_gcs", "A_gcs (темп в разрыве / вне)", 2),
             ("B_per_agent", "B, байт/с на агента", 0), ("N_dup", "N_dup", 2), ("N_lost", "N_lost", 2)],
            ([("B4", "B2"), ("B5", "B2")], [("U", "U(M)"), ("CR", "CR")]), "P5_3_channel.png")
    section("E3L. Кривая потерь (с нарушениями)", e3l, [f"loss={q}" for q in LOSS_CURVE], CORE,
            [("U", "U(M)", 3), ("CR", "CR", 3)], ([("B4", "B2"), ("B5", "B2")], [("U", "U(M)")]))
    L += ["## График 4. Фронт Парето", "", "![Парето](figures/P5_4_pareto.png)", "",
          "Размер точки ~ log(вычисления на пересчёт); подпись — мс. Для B0/B1 трафик распределения нулевой "
          "(план выдан до вылета), для B4 вычисления распределены по бортам.", ""]
    section("E4. Нестационарность", e4, ["D=0", "D=1"], CORE, [("U", "U(M)", 3), ("CR", "CR", 3),
            ("T_newtask_done", "T_newtask_done, с", 0)], ([("B2", "B0"), ("B4", "B2"), ("B5", "B2")], [("U", "U(M)")]))
    section("E5. Неоднородность целей κ", e5, ["κ=uniform", "κ=hotspots"], CORE,
            [("U", "U(M)", 3), ("T_detect_90", "T_detect(90 %), с", 0)],
            ([("B0", "B2"), ("B4", "B2"), ("B2", "B1")], [("U", "U(M)"), ("T_detect_90", "T_detect_90")]))
    L += ["## Абляции (график 5)", "", "![абляции](figures/P5_5_ablation.png)", "",
          "| опыт | модуль | метрика | n | с − без [CI95] | отн. | p | r_rb | невалидных с / без | N_col с / без | N_nm с / без |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ab:
        L.append(f"| {r['exp']} | {r['name']} | {r['key']} | {r['n']} | {_f(r['diff'], 3)} [{_f(r['lo'], 3)}; {_f(r['hi'], 3)}] | "
                 f"{_f(r['rel'], 2, True)} | {_p(r['p'], r.get('det'))} | {_f(r['r_rb'])} ({effect_word(r['r_rb'])}) | "
                 f"{r['inv_on']} / {r['inv_off']} | {r['col_on']} / {r['col_off']} | {r['nm_on']} / {r['nm_off']} |")
    L += ["", "* A1 — B2 с пересчётом по событиям и без него, 20 % тихих отказов.",
          "* A2 — B5 во время разрыва со станцией (100–400 с, потери 10 %), два агента в 120 с теряют половину скорости; "
          "локальный ремонт (отрезать хвост плана, не проходящий по энергии) включён/выключен.",
          "* A3 — B2 на карте с 3 «горячими» зонами: ценность сектора = доля вероятности целей (Pd) или 1 (покрытие).",
          "* A4 — B2, N = 20: эшелоны (10 высот) / один эшелон × ORCA / без. Здесь невалидные прогоны учтены в сравнении — "
          "столкновение и есть измеряемый эффект.", "",
          "## График 6. Карта миссии", "", "![карта](figures/P5_6_mission_map.png)", "",
          "Сырые данные: `docs/reports/p5/P5_*.csv` (по одному на опыт)."]
    L += _notes(e1, e1b, V)
    (REPORTS / "P5_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"passed": passed, "runs": n_runs, "N_col": n_col, "N_dup": n_dup, "invalid": n_invalid,
            "plots": plots, "missing": missing, "verdicts": {k: v[0] for k, v in V.items()},
            "report": str(REPORTS / "P5_comparison.md")}
