"""Общие типы данных — контракт между модулями (см. docs/06, раздел 3).

Меняются только решением всей команды.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import numpy as np

# Назначение: agent_id -> упорядоченный список task_id (маршрут агента).
Assignment = dict[int, list[int]]


class TaskStatus(str, Enum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class Mode(str, Enum):
    IDLE = "idle"          # на земле, до взлёта
    MISSION = "mission"    # выполняет план
    RTL = "rtl"            # возврат домой
    LANDED = "landed"      # приземлился после миссии
    FAILED = "failed"      # отказ (выведен из строя)


@dataclass
class Task:
    """Задача ST-SR-TA: обойти маршрут `waypoints` (например, «змейка» по сектору)."""

    id: int
    waypoints: np.ndarray            # (K, 2), м
    area: float = 0.0                # м², для задач покрытия
    priority: float = 1.0
    status: TaskStatus = TaskStatus.UNASSIGNED
    owner: int | None = None
    t_start: float | None = None
    t_done: float | None = None
    times_done: int = 0              # > 1 означает дублирование (N_dup)
    progress: int = 0                # индекс следующей путевой точки: переназначенная задача
                                     # продолжается с места остановки, а не с начала
    t_appear: float = 0.0            # когда задача появилась (новые задачи, FR-9)
    kind: str = "sector"             # sector — «змейка» сектора; point — осмотр точки (новая цель)

    @property
    def loc(self) -> np.ndarray:
        return self.waypoints[0]

    @property
    def length(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1).sum())


class TaskSet:
    """Набор задач с доступом по id и в стабильном (детерминированном) порядке."""

    def __init__(self, tasks: list[Task] | None = None):
        self._tasks: dict[int, Task] = {}
        for t in tasks or []:
            self.add(t)

    def add(self, task: Task) -> None:
        if task.id in self._tasks:
            raise ValueError(f"duplicate task id {task.id}")
        self._tasks[task.id] = task

    def __getitem__(self, task_id: int) -> Task:
        return self._tasks[task_id]

    def __contains__(self, task_id: int) -> bool:
        return task_id in self._tasks

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def unfinished(self) -> list[Task]:
        return [t for t in self if t.status not in (TaskStatus.DONE, TaskStatus.FAILED)]

    def with_status(self, *statuses: TaskStatus) -> list[Task]:
        return [t for t in self if t.status in statuses]


@dataclass
class VehicleState:
    """Кинематическое состояние аппарата (выход MotionSource.step)."""

    p: np.ndarray                    # (2,) позиция, м
    v: np.ndarray                    # (2,) скорость, м/с
    psi: float = 0.0                 # курс, рад
    alt: float = 50.0                # высота эшелона, м

    @property
    def speed(self) -> float:
        return float(np.hypot(self.v[0], self.v[1]))


@dataclass(frozen=True)
class AgentState:
    """Снимок агента для распределителя: s_i = (p, v, ψ, e, h, σ) из docs/01, 2.2."""

    id: int
    p: np.ndarray
    v: np.ndarray
    psi: float
    alt: float
    energy_j: float
    capacity_j: float
    healthy: bool
    mode: Mode
    home: np.ndarray
    v_cruise: float
    plan: tuple[int, ...] = ()
    current_task: int | None = None
    wp_idx: int = 0                  # следующая путевая точка текущей задачи
    low_battery: bool = False        # ушёл домой по заряду — задач больше не берёт


@dataclass(frozen=True)
class WorldSnapshot:
    """То, что видит распределитель при вызове allocate()."""

    t: float
    agents: tuple[AgentState, ...]
    bounds: tuple[float, float, float, float]      # xmin, ymin, xmax, ymax
    nofly: tuple[np.ndarray, ...] = field(default_factory=tuple)

    def healthy_agents(self) -> list[AgentState]:
        return [a for a in self.agents if a.healthy]
