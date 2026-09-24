# `swarmsim/interfaces.py` — интерфейсы модулей и реестр ★

**Назначение.** Зафиксировать интерфейсы из `docs/06`, раздел 3, и дать механизм, которым
команды подключают свои реализации **без правки ядра**. Шаблон «Стратегия»: `docs/02`,
раздел 2.1, «Allocator — это стратегия с единым интерфейсом».

**Импорты.** `abc` (абстрактные классы), `typing`, NumPy, типы из `datatypes`.
`TYPE_CHECKING`-импорты (`Agent`, `World`, `Scenario`) нужны только для подсказок типов
и не создают циклических импортов.

## Интерфейсы

Все — абстрактные классы (`ABC`). Методы с `@abstractmethod` обязательны: без них класс
нельзя создать. У каждого есть атрибут `name` (подставляется при регистрации, по
умолчанию `"custom"`).

### `Allocator` — владелец R2
- `__init__(scenario, params, rng)` — `params` из YAML, `rng` — свой поток случайных чисел.
- **`allocate(snapshot, tasks) -> Assignment`** — вернуть `{agent_id: [task_id, ...]}`.
  Одна задача — не более одному агенту.
- `should_trigger(t, tasks, events) -> bool` — нужно ли перераспределить на этом шаге;
  `events` — новые события с прошлого шага. По умолчанию `False` (статическое
  назначение). **Нельзя опираться на настенное время** — сломается детерминизм.

### `CommsBus` — владелец R3
`step(t, agents)`, `send(src, dst, msg)`, `broadcast(src, msg)`, `neighbors_of(id)`,
`inbox(id)`, `gcs_reachable()` (есть ли связь с наземной станцией; по умолчанию `True`),
`stats()` (например, число сообщений; уходит в summary).

### `FaultManager` — владелец R3
`inject_due(t, agents) -> [(agent_id, причина), ...]` — какие отказы наступили к `t`.

### `MissionPlugin`
`decompose(world) -> TaskSet` — разбить миссию на задачи («домен — плагин», антипаттерн 7
из `docs/02`).

### `SafetyFilter` — владелец R2
`filter(me, cmd, neighbors) -> MotionCommand` — скорректировать команду движения с учётом
соседей (ORCA и т. п.).

## Реестр

Словари `ALLOCATORS`, `COMMS`, `FAULTS`, `MISSIONS`, `SAFETY`: имя → класс.

### `_register(table, name)` и декораторы `register_allocator(name)`, `register_comms`, `register_faults`, `register_mission`, `register_safety`
Декоратор кладёт класс в словарь под именем и записывает `cls.name = name`. Повторная
регистрация **другого** класса под тем же именем — ошибка (защита от конфликтов между
командами).

### `lookup(table, kind, what) -> type`
Найти класс по имени из конфига. Если имени нет — `KeyError` со списком доступных имён.

## Как подключить свой модуль
```python
# swarmsim/alloc/cbba.py
from ..interfaces import Allocator, register_allocator

@register_allocator("cbba")
class CBBA(Allocator):
    def allocate(self, snapshot, tasks):
        ...
    def should_trigger(self, t, tasks, events):
        return any(e["type"] in ("task_released", "new_targets") for e in events)
```
Затем добавить `from . import cbba` в `swarmsim/alloc/__init__.py` и указать в сценарии
`allocator: {kind: cbba}`.
