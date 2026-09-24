"""Статистика отчёта P5 (docs/03, раздел 7).

* парный критерий Уилкоксона (signed-rank) по зёрнам — непараметрический, без нормальности;
* доверительные интервалы — бутстреп, 1000 ресемплов (percentile);
* величина эффекта — ранговая бисериальная корреляция для парных данных (r_rb ∈ [−1, 1])
  и относительная разница средних; p-value без эффекта не интерпретируется;
* описание конфигурации: mean ± CI95, median, IQR, n, число невалидных.
"""
from __future__ import annotations

import math

import numpy as np

B_RESAMPLES = 1000


def _clean(xs) -> np.ndarray:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    return a[~np.isnan(a)]


def bootstrap_ci(xs, stat=np.mean, n: int = B_RESAMPLES, seed: int = 0, alpha: float = 0.05):
    a = _clean(xs)
    if len(a) == 0:
        return (math.nan, math.nan)
    if len(a) == 1 or np.all(a == a[0]):
        return (float(a[0]), float(a[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    boots = stat(a[idx], axis=1)
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def describe(xs) -> dict:
    a = _clean(xs)
    if len(a) == 0:
        return {"n": 0, "mean": math.nan, "lo": math.nan, "hi": math.nan, "median": math.nan,
                "q1": math.nan, "q3": math.nan}
    lo, hi = bootstrap_ci(a)
    q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
    return {"n": int(len(a)), "mean": float(a.mean()), "lo": lo, "hi": hi,
            "median": float(med), "q1": float(q1), "q3": float(q3)}


def paired(x, y) -> dict:
    """Сравнение x против y по парам (одинаковые зёрна). Разность d = x − y.

    Возвращает n, среднюю разность и её бутстреп-CI95, p (Уилкоксон, двусторонний),
    r_rb (парная ранговая бисериальная корреляция: +1 — x всегда больше), rel = mean(d)/|mean(y)|.
    """
    from scipy.stats import rankdata, wilcoxon

    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
             and not (isinstance(a, float) and math.isnan(a)) and not (isinstance(b, float) and math.isnan(b))]
    if not pairs:
        return {"n": 0, "diff": math.nan, "lo": math.nan, "hi": math.nan, "p": math.nan, "r_rb": math.nan,
                "rel": math.nan}
    xa = np.array([p[0] for p in pairs], float)
    ya = np.array([p[1] for p in pairs], float)
    d = xa - ya
    my = float(ya.mean())
    if len(d) > 1 and np.all(np.abs(d - d[0]) <= 1e-9 * max(1.0, abs(d[0]))):
        # все пары дают одну и ту же разность: мир не меняется от зерна (детерминированный метод
        # на детерминированном мире) — эффективный объём выборки 1, p-value не имеет смысла
        return {"n": int(len(d)), "diff": float(d[0]), "lo": float(d[0]), "hi": float(d[0]), "p": math.nan,
                "r_rb": float(np.sign(d[0])), "rel": float(d[0] / abs(my)) if abs(my) > 1e-12 else math.nan,
                "det": True}
    lo, hi = bootstrap_ci(d)
    nz = d[np.abs(d) > 1e-12]
    if len(nz) == 0:
        p, r = 1.0, 0.0
    else:
        p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else 1.0
        ranks = rankdata(np.abs(nz))
        r = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return {"n": int(len(d)), "diff": float(d.mean()), "lo": lo, "hi": hi, "p": p, "r_rb": r,
            "rel": float(d.mean() / abs(my)) if abs(my) > 1e-12 else math.nan, "det": False}


def effect_word(r: float) -> str:
    """Словесная шкала для |r_rb| (Kerby 2014 / Cohen-подобные пороги)."""
    if r is None or math.isnan(r):
        return "—"
    a = abs(r)
    return "нет" if a < 0.1 else "малый" if a < 0.3 else "средний" if a < 0.5 else "большой"
