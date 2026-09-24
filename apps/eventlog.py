"""Человекочитаемая хронология events.jsonl.

    swarmcore log runs/baseline_static_roundrobin_s1
    swarmcore log runs/... --agent 3 --type task_done,target_detected --from 100 --to 300
"""
from __future__ import annotations

import json
from pathlib import Path


def _describe(e: dict) -> str:
    k, a = e["type"], e.get("agent")
    tasks = lambda xs: ", ".join(map(str, xs)) if xs else "—"  # noqa: E731
    match k:
        case "allocation":
            n = sum(len(v) for v in e["assignment"].values())
            return f"распределение ({e['reason']}, {e['allocator']}): {n} задач на {len(e['assignment'])} БПЛА"
        case "plan":
            return f"БПЛА {a}: план [{tasks(e['tasks'])}]" + (
                f", текущая {e['current']}" if e.get("current") is not None else "")
        case "takeoff":
            return f"БПЛА {a}: взлёт"
        case "task_started":
            return f"БПЛА {a}: начал сектор {e['task']}"
        case "task_done":
            return f"БПЛА {a}: закончил сектор {e['task']}"
        case "task_released":
            return f"БПЛА {a}: освободил сектор {e['task']} ({e['reason']})"
        case "target_detected":
            return f"БПЛА {a}: обнаружил цель {e['target']}"
        case "new_targets":
            return f"появились цели [{tasks(e['targets'])}]"
        case "plan_complete_rtl":
            return f"БПЛА {a}: план выполнен, возврат домой"
        case "low_battery_rtl":
            return f"БПЛА {a}: низкий заряд, возврат домой, освобождены [{tasks(e.get('released'))}]"
        case "battery_depleted":
            return f"БПЛА {a}: батарея разряжена"
        case "agent_failed":
            return f"БПЛА {a}: ОТКАЗ ({e['reason']}), освобождены [{tasks(e.get('released'))}]"
        case "landed":
            return f"БПЛА {a}: посадка, остаток {e['energy_j'] / 3600:.1f} Вт·ч"
        case "collision":
            return f"СТОЛКНОВЕНИЕ БПЛА {e['agents']}"
        case "near_miss":
            return f"опасное сближение БПЛА {e['agents']}"
        case "geofence_violation":
            return f"БПЛА {a}: нарушение геозоны"
    rest = {x: y for x, y in e.items() if x not in ("t", "type")}
    return f"{k} {json.dumps(rest, ensure_ascii=False)}"


def load_events(run_dir: str | Path) -> list[dict]:
    with (Path(run_dir) / "events.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _involves(e: dict, agent: int) -> bool:
    return e.get("agent") == agent or agent in e.get("agents", ()) or str(agent) in e.get("assignment", {})


def log_command(args) -> int:
    events = load_events(args.run_dir)
    types = set(args.type.split(",")) if args.type else None
    shown = 0
    for e in events:
        if types and e["type"] not in types:
            continue
        if args.agent is not None and not _involves(e, args.agent):
            continue
        if not (args.t_from <= e["t"] <= args.t_to):
            continue
        m, s = divmod(e["t"], 60)
        print(f"{int(m):3d}:{s:04.1f}  {e['type']:<18} {_describe(e)}")
        shown += 1
    if args.stats:
        from collections import Counter
        print("\nпо типам:")
        for k, n in Counter(e["type"] for e in events).most_common():
            print(f"  {k:<20}{n:>6}")
    print(f"\nпоказано {shown} из {len(events)} событий")
    return 0


def add_log_args(p) -> None:
    p.add_argument("run_dir")
    p.add_argument("--agent", type=int, default=None, help="только события этого БПЛА")
    p.add_argument("--type", default=None, help="типы через запятую, напр. task_done,target_detected")
    p.add_argument("--from", dest="t_from", type=float, default=float("-inf"), help="с момента t, с")
    p.add_argument("--to", dest="t_to", type=float, default=float("inf"), help="до момента t, с")
    p.add_argument("--stats", action="store_true", help="добавить сводку по типам событий")
