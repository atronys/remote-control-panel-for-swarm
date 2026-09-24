"""Интерфейсы подключаемых модулей (docs/06, раздел 3) и реестр реализаций.

Ядро (SimLoop) знает только эти интерфейсы. Команды регистрируют свои реализации:

    from swarmsim.interfaces import Allocator, register_allocator

    @register_allocator("cbba")
    class CBBAAllocator(Allocator):
        def allocate(self, snapshot, tasks): ...

и выбирают их в сценарии: `allocator: {kind: cbba, ...параметры}`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .datatypes import AgentState, Assignment, TaskSet, WorldSnapshot

if TYPE_CHECKING:
    from .agents.base import Agent, MotionCommand
    from .scenario import Scenario
    from .world import World


class Allocator(ABC):
    """Владелец — R2. Стратегия распределения задач с единым интерфейсом."""

    name: str = "custom"

    def __init__(self, scenario: "Scenario", params: dict[str, Any], rng: np.random.Generator):
        self.scenario = scenario
        self.params = params
        self.rng = rng

    @abstractmethod
    def allocate(self, snapshot: WorldSnapshot, tasks: TaskSet) -> Assignment:
        """Вернуть agent_id -> упорядоченный список task_id. Одна задача — не более одному агенту."""

    def should_trigger(self, t: float, tasks: TaskSet, events: list[dict]) -> bool:
        """Нужно ли перераспределить сейчас. Только модельное время и события — не настенное."""
        return False


class CommsBus(ABC):
    """Владелец — R3. Модель канала связи."""

    name: str = "custom"

    def __init__(self, scenario: "Scenario", params: dict[str, Any], rng: np.random.Generator):
        self.scenario = scenario
        self.params = params
        self.rng = rng

    @abstractmethod
    def step(self, t: float, agents: list["Agent"]) -> None: ...

    @abstractmethod
    def send(self, src: int, dst: int, msg: dict) -> None: ...

    @abstractmethod
    def broadcast(self, src: int, msg: dict) -> None: ...

    @abstractmethod
    def neighbors_of(self, agent_id: int) -> list[int]: ...

    @abstractmethod
    def inbox(self, agent_id: int) -> list[dict]: ...

    def gcs_reachable(self) -> bool:
        return True

    def stats(self) -> dict[str, float]:
        return {}


class FaultManager(ABC):
    """Владелец — R3. Инъекция отказов."""

    name: str = "custom"

    def __init__(self, scenario: "Scenario", rng: np.random.Generator):
        self.scenario = scenario
        self.rng = rng

    @abstractmethod
    def inject_due(self, t: float, agents: list["Agent"]) -> list[tuple[int, str]]:
        """Вернуть список (agent_id, причина) отказов, наступивших к моменту t."""


class MissionPlugin(ABC):
    """Декомпозиция миссии в набор задач (домен — плагин)."""

    name: str = "custom"

    def __init__(self, scenario: "Scenario"):
        self.scenario = scenario

    @abstractmethod
    def decompose(self, world: "World") -> TaskSet: ...


class SafetyFilter(ABC):
    """Владелец — R2 (ORCA и т.п.). Корректирует команду движения с учётом соседей."""

    name: str = "custom"

    def __init__(self, scenario: "Scenario"):
        self.scenario = scenario

    @abstractmethod
    def filter(self, me: AgentState, cmd: "MotionCommand",
               neighbors: list[AgentState]) -> "MotionCommand": ...


# ---------------------------------------------------------------- реестр

ALLOCATORS: dict[str, type[Allocator]] = {}
COMMS: dict[str, type[CommsBus]] = {}
FAULTS: dict[str, type[FaultManager]] = {}
MISSIONS: dict[str, type[MissionPlugin]] = {}
SAFETY: dict[str, type[SafetyFilter]] = {}


def _register(table: dict, name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in table and table[name] is not cls:
            raise ValueError(f"имя {name!r} уже зарегистрировано: {table[name]}")
        table[name] = cls
        cls.name = name
        return cls
    return deco


def register_allocator(name: str): return _register(ALLOCATORS, name)
def register_comms(name: str): return _register(COMMS, name)
def register_faults(name: str): return _register(FAULTS, name)
def register_mission(name: str): return _register(MISSIONS, name)
def register_safety(name: str): return _register(SAFETY, name)


def lookup(table: dict[str, type], kind: str, what: str) -> type:
    try:
        return table[kind]
    except KeyError:
        raise KeyError(f"неизвестный {what} {kind!r}; доступны: {sorted(table)}") from None
