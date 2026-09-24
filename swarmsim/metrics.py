"""MetricsRecorder. Владелец — R4; здесь базовый набор метрик из docs/03, раздел 4,
который ядро умеет считать само (обнаружение, задачи, энергия, жёсткие ограничения).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .datatypes import Mode, TaskSet, TaskStatus

if TYPE_CHECKING:
    from .agents.base import Agent
    from .world import World

DETECT_LEVELS = (0.5, 0.9)
# Поля, зависящие от настенного времени, — исключаются из проверки детерминизма.
WALLCLOCK_FIELDS = ("wall_time_s", "realtime_factor", "alloc_ms_mean", "alloc_ms_max", "node_ms_mean", "node_ms_max")


class MetricsRecorder:
    def __init__(self, record_every: float, trajectories: bool):
        self.record_every = record_every
        self.keep_traj = trajectories
        self._next_t = 0.0
        self.timeseries: list[dict] = []
        self.trajectories: list[dict] = []
        self.t_detect: dict[str, float | None] = {}
        for x in DETECT_LEVELS:
            self.t_detect[f"T_detect_{int(x * 100)}"] = None
            self.t_detect[f"T_detect_{int(x * 100)}_realized"] = None

    def record(self, t: float, world: "World", agents: list["Agent"], tasks: TaskSet) -> None:
        exp_frac = world.expected_detected_fraction()
        real_frac = world.detected_fraction()
        for x in DETECT_LEVELS:
            k = f"T_detect_{int(x * 100)}"
            if self.t_detect[k] is None and exp_frac >= x:
                self.t_detect[k] = round(t, 6)
            if self.t_detect[k + "_realized"] is None and real_frac >= x:
                self.t_detect[k + "_realized"] = round(t, 6)

        if t + 1e-9 < self._next_t:
            return
        self._next_t += self.record_every
        self.timeseries.append({
            "t": round(t, 6),
            "detected_frac": round(real_frac, 6),
            "expected_detect_frac": round(exp_frac, 6),
            "tasks_done": len(tasks.with_status(TaskStatus.DONE)),
            "airborne": sum(a.airborne for a in agents),
            "energy_used_wh": round(sum(a.energy_used_j for a in agents) / 3600.0, 6),
        })
        if self.keep_traj:
            for a in agents:
                s = a.state
                self.trajectories.append({
                    "t": round(t, 6), "agent": a.id, "x": round(float(s.p[0]), 4),
                    "y": round(float(s.p[1]), 4), "alt": s.alt, "vx": round(float(s.v[0]), 4),
                    "vy": round(float(s.v[1]), 4), "mode": a.mode.value,
                    "task": a.current if a.current is not None else "",
                    "energy_frac": round(a.e_j / a.capacity_j, 6),
                    "wp": a.wp_idx if a.current is not None else "",   # индекс путевой точки в задаче
                })

    def finalize(self, t_final: float, world: "World", agents: list["Agent"], tasks: TaskSet,
                 extra: dict) -> dict:
        done = tasks.with_status(TaskStatus.DONE)
        n = len(tasks)
        makespan = max((t.t_done for t in done), default=None)
        e_total = sum(a.energy_used_j for a in agents) / 3600.0
        n_dup = sum(1 for t in tasks if t.times_done > 1)
        orphaned = sum(1 for t in tasks if t.status == TaskStatus.UNASSIGNED)
        summary = {
            **extra,
            "t_final": round(t_final, 6),
            **self.t_detect,
            "detected_frac": round(world.detected_fraction(), 6),
            "expected_detect_frac": round(world.expected_detected_fraction(), 6),
            "targets": world.n_active_targets,
            "tasks_total": n,
            "tasks_done": len(done),
            "tasks_orphaned": orphaned,
            "CR": round(len(done) / n, 6) if n else 1.0,
            "makespan": round(makespan, 6) if makespan is not None else None,
            "E_total_Wh": round(e_total, 6),
            "distance_km": round(sum(a.distance_m for a in agents) / 1000.0, 6),
            "agents_failed": sum(a.mode == Mode.FAILED for a in agents),
            "agents_landed": sum(a.mode == Mode.LANDED for a in agents),
            "N_col": world.n_collisions,
            "N_nm": world.n_near_miss,
            "N_geo": world.n_geofence,
            "N_dup": n_dup,
        }
        summary["valid"] = summary["N_col"] == 0 and summary["N_dup"] == 0 and summary["N_geo"] == 0
        return summary



def recovery_metrics(events: list[dict]) -> dict:
    """Метрики динамики по журналу событий (docs/03, раздел 4; определения зафиксированы до P3).

    T_recovery — от отказа до момента, когда КАЖДАЯ задача, освобождённая этим отказом, оказалась
                 в плане (событие plan) или в работе (task_started) у другого живого агента:
                 «перераспределены и приняты к исполнению». Если хоть одна так и не нашла
                 исполнителя — отказ не восстановлен (T_recovery = None, recovered_frac < 1).
    T_resume   — то же, но до фактического начала (task_started) последней из них.
    T_realloc  — от момента, когда система УЗНАЛА об отказе (agent_failed с сообщением или
                 failure_detected), до ближайшего распределения.
    T_response_new — от появления новой задачи (FR-9) до начала её выполнения.
    """
    fails = [e for e in events if e["type"] == "agent_failed"]
    rec, res, realloc, recovered = [], [], [], 0
    allocs = [e["t"] for e in events if e["type"] == "allocation"]
    for f in fails:
        aid, tf = f["agent"], f["t"]
        why = ("injected", "failure_detected", "battery_depleted")
        released = {e["task"] for e in events if e["type"] == "task_released" and e["agent"] == aid
                    and e["t"] >= tf - 1e-9 and e.get("reason") in why}
        known = f["t"] if f.get("detect", "reported") == "reported" else next(
            (e["t"] for e in events if e["type"] == "failure_detected" and e["agent"] == aid), None)
        if known is not None:
            nxt = next((ta for ta in allocs if ta >= known - 1e-9), None)
            if nxt is not None:
                realloc.append(nxt - known)
        if not released:
            recovered += 1
            rec.append(0.0)
            continue
        t_plan, t_start = {}, {}
        for e in events:
            if e["t"] < tf - 1e-9:
                continue
            if e["type"] == "plan" and e["agent"] != aid:
                for tid in e["tasks"]:
                    if tid in released and tid not in t_plan:
                        t_plan[tid] = e["t"]
            elif e["type"] == "task_started" and e["agent"] != aid and e["task"] in released:
                t_plan.setdefault(e["task"], e["t"])
                t_start.setdefault(e["task"], e["t"])
        if len(t_plan) == len(released):
            recovered += 1
            rec.append(max(t_plan.values()) - tf)
            if len(t_start) == len(released):
                res.append(max(t_start.values()) - tf)
    new = {e["task"]: e["t"] for e in events if e["type"] == "new_task"}
    resp, done_new = [], []
    for tid, ta in new.items():
        ts = next((e["t"] for e in events if e["type"] == "task_started" and e["task"] == tid), None)
        td = next((e["t"] for e in events if e["type"] == "task_done" and e["task"] == tid), None)
        if ts is not None:
            resp.append(ts - ta)                 # решение принято и исполнитель вылетел к цели
        if td is not None:
            done_new.append(td - ta)             # цель осмотрена
    out = {
        "n_failures": len(fails),
        "recovered_frac": round(recovered / len(fails), 6) if fails else 1.0,
        "T_recovery_mean": round(sum(rec) / len(rec), 6) if rec else None,
        "T_recovery_max": round(max(rec), 6) if rec else None,
        "T_resume_max": round(max(res), 6) if res else None,
        "T_realloc_max": round(max(realloc), 6) if realloc else None,
        "n_new_tasks": len(new),
        "T_response_new": round(max(resp), 6) if resp else None,
        "T_newtask_done": round(max(done_new), 6) if len(done_new) == len(new) and new else None,
    }
    if fails and recovered < len(fails):
        out["T_recovery_max"] = None                   # есть невосстановленный отказ
    return out
