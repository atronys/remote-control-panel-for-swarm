# `swarmsim/datatypes.py` — общие типы данных ★

**Назначение.** Типы, которыми обмениваются модули. Это **контракт между командами**
(`docs/06`, раздел 3): меняется только решением всей команды.

Файл называется `datatypes.py`, а не `types.py`: имя `types` совпадает со стандартным
модулем Python и ломает импорт, если запускать скрипты из папки пакета.

**Импорты.** `dataclasses`, `enum`, `typing.Iterator`, NumPy.

## `Assignment = dict[int, list[int]]`
Назначение: `agent_id → упорядоченный список task_id`. Порядок — это маршрут агента.

## `class TaskStatus(str, Enum)`
`UNASSIGNED` → `ASSIGNED` → `IN_PROGRESS` → `DONE`; также `FAILED`. Наследование от
`str` делает значения удобными для JSON/CSV.

## `class Mode(str, Enum)`
Режим агента: `IDLE`, `MISSION`, `RTL`, `LANDED`, `FAILED` (схема переходов — в
[agents_base.md](agents_base.md)).

## `@dataclass class Task`
Задача класса ST-SR-TA (`docs/01`, раздел 2.1).

| Поле | Смысл |
|---|---|
| `id` | номер |
| `waypoints (K, 2)` | маршрут внутри задачи (например, «змейка» по сектору) |
| `area` | площадь для задач покрытия, м² |
| `priority` | приоритет (для `U(M)` из `docs/01`) |
| `status`, `owner` | состояние и исполнитель |
| `t_start`, `t_done` | время начала и окончания |
| `times_done` | сколько раз выполнена; > 1 — это дубль (`N_dup`, жёсткое ограничение) |

Свойства: `loc` — первая точка маршрута; `length` — длина маршрута, м.

## `class TaskSet`
Набор задач с доступом по id и **стабильным порядком** (словарь в порядке добавления):
- `add(task)` — дубликат id даёт ошибку;
- `ts[id]`, `id in ts`, `for t in ts`, `len(ts)`;
- `unfinished()` — не `DONE` и не `FAILED`;
- `with_status(*statuses)` — фильтр по статусам.

## `@dataclass class VehicleState`
Кинематическое состояние — выход `MotionSource.step`: `p (2,)`, `v (2,)`, `psi`, `alt`;
свойство `speed = |v|`.

## `@dataclass(frozen=True) class AgentState`
Неизменяемый снимок агента для распределителя: `id, p, v, psi, alt, energy_j, capacity_j,
healthy, mode, home, v_cruise, plan, current_task` — вектор `s_i = (p, v, ψ, e, h, σ)` из
`docs/01`, раздел 2.2 (сенсоры `σ` пока не нужны).

## `@dataclass(frozen=True) class WorldSnapshot`
Что видит распределитель: `t`, кортеж `agents`, `bounds`, `nofly`; метод
`healthy_agents()`.
