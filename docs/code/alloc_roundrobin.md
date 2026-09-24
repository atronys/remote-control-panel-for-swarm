# `swarmsim/alloc/roundrobin.py` — распределитель `static_roundrobin` ○ (владелец R2)

**Назначение.** Наивный базовый метод B0 из `docs/03`: задачи раздаются по кругу **один
раз** в начале миссии, без учёта расстояний, энергии и отказов. Нужен как (1) эталонная
реализация интерфейса `Allocator` и (2) нижняя планка для сравнения.

## `@register_allocator("static_roundrobin") class StaticRoundRobin(Allocator)`
### `allocate(snapshot, tasks) -> Assignment`
1. Исправные агенты, отсортированные по id.
2. Невыполненные задачи в порядке id.
3. Задача `k` → агенту `k mod N`.

`should_trigger` не переопределён, то есть всегда `False`: после отказа задачи агента
остаются ничьими (`tasks_orphaned`), и CR падает. В `scenarios/failures.yaml` это даёт
CR = 0.84 — базовая линия, которую должен побить динамический распределитель.

## Что сделает R2
`static_oracle` (OR-Tools CP-SAT), `centralized`, `cbba`, `hybrid` — по `docs/04`.
Пример динамического распределителя, с которым ядро уже проверено, есть в тесте
`tests/test_sim.py::_Reallocating`.
