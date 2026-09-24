"""Отладочная картинка прогона по trajectories.csv (не GUI; полноценная визуализация — позже).

Тот же trajectories.csv можно проигрывать во внешнем 3D-просмотрщике (например, Unity):
в нём есть t, agent, x, y, alt, vx, vy, mode.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def plot_run(run_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    world = manifest["config"]["world"]
    tracks: dict[int, list[tuple[float, float]]] = {}
    with (run_dir / "trajectories.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tracks.setdefault(int(row["agent"]), []).append((float(row["x"]), float(row["y"])))

    fig, ax = plt.subplots(figsize=(7, 7))
    for poly in world["nofly"]:
        xs, ys = zip(*(poly + [poly[0]]))
        ax.fill(xs, ys, color="tab:red", alpha=0.2, label="бесполётная зона")
    for aid, pts in sorted(tracks.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, lw=0.8, label=f"БПЛА {aid}")
    w, h = world["size"]
    ax.set_xlim(-10, w + 10)
    ax.set_ylim(-10, h + 10)
    ax.set_aspect("equal")
    ax.set_title(f"{manifest['scenario']} · {manifest['allocator']} · seed {manifest['seed']}")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    out = run_dir / "trajectories.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
