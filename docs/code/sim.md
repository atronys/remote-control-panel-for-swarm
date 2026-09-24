# `swarmsim/sim.py` — ядро: детерминированный цикл симуляции ★

**Назначение.** Собрать из сценария все части симуляции (мир, агентов, задачи, плагины),
прогнать модельное время по шагам в строго фиксированном порядке и вернуть результат с
метриками, журналом и манифестом. Это `SimLoop` из таблицы модулей `docs/02`, раздел 2.1,
а цикл — прямая реализация псевдокода из `docs/02`, раздел 4.

**Импорты.** NumPy (генераторы случайных чисел); из стандартной библиотеки — `time`
(замер скорости), `platform`, `subprocess`, `datetime` (для манифеста), `dataclasses`.
Внутренние модули: `world`, `agents.*`, `interfaces`, `metrics`, `scenario`, `sensing`.
Строка `from . import alloc, comms, faults, missions` нужна, чтобы встроенные плагины
**зарегистрировались** в реестре при импорте (у них декораторы `@register_...`).

---

## Константы

### `RNG_STREAMS = ("world", "sensing", "comms", "faults", "alloc")`
Имена независимых потоков случайных чисел, по одному на подсистему.
**Порядок менять нельзя, новые потоки — только в конец.** Иначе для старых зёрен
изменятся миры, и сравнение со старыми результатами сломается.

## Исключения

### `class CollisionError(RuntimeError)`
Бросается, если произошло столкновение и в сценарии `safety.strict: true`. По умолчанию
`strict: false`: прогон не падает, а помечается `valid = false`.

### `class AssignmentError(ValueError)`
Бросается, если распределитель вернул недопустимое назначение (см. `_check_assignment`).
Это «валидатор уровня ядра»: ошибка алгоритма R2 ловится сразу, а не превращается в
странные метрики.

## `class EventLog` — журнал событий («чёрный ящик», SR-4 / FR-10)

| Метод | Что делает |
|---|---|
| `log(t, kind, **data)` | добавляет запись `{"t": t, "type": kind, ...данные}`; время округляется до 6 знаков, чтобы в JSON не было «шумных» хвостов |
| `__len__()` | число записей |
| `since(idx)` | записи начиная с номера `idx`. Ядро так передаёт распределителю «новые события с прошлого шага» |

Журнал передаётся всем частям как функция `log`: агентам, миру. Поэтому все события
(взлёт, отказ, обнаружение, столкновение…) лежат в одном упорядоченном списке и пишутся в
`events.jsonl`.

## `@dataclass class RunResult`
Результат прогона: `summary` (итоговые метрики, словарь), `timeseries` (ряды раз в
`record_every` секунд), `trajectories` (положения агентов), `events` (журнал),
`manifest` (для воспроизводимости).

## `make_rngs(seed) -> dict[str, np.random.Generator]`
```python
children = np.random.SeedSequence(seed).spawn(len(RNG_STREAMS))
return {name: np.random.default_rng(s) for name, s in zip(RNG_STREAMS, children)}
```
Из одного зерна получается 5 **статистически независимых** генераторов (рекомендованный
в документации NumPy способ, см. `00_libraries_and_sources.md`, п. 3.7).

Зачем это нужно. Если бы все подсистемы брали числа из одного генератора, то изменение,
скажем, числа агентов меняло бы, сколько чисел «съел» сенсор. Тогда сдвинулись бы и отказы,
и мир. С отдельными потоками **мир при одном зерне одинаков при любом N и любом
распределителе** — это парный дизайн из `docs/03`.

---

## `class SimLoop`

### `__init__(scenario, seed, allocator=None)`
Построение прогона. По шагам:
1. `make_rngs(seed)`, `EventLog()`.
2. `World(scenario, rngs["world"], rngs["sensing"], log)` — мир со своими потоками.
3. `SensorModel(scenario.sensor)`, `RotaryWingEnergy(scenario.energy)`.
4. Фильтр безопасности по имени `scenario.safety.filter` из реестра `SAFETY`.
5. `_build_agents()` — агенты.
6. Миссия по `scenario.mission.kind` → `self.tasks = mission.decompose(world)`.
7. Распределитель: либо переданный объект `allocator` (удобно в тестах и для отладки
   своего алгоритма), либо класс из реестра по `scenario.allocator.kind`. Все
   дополнительные ключи из YAML (`allocator: {kind: cbba, max_bundle: 3}`) приходят ему в
   `params` — это `model_extra` pydantic.
8. Связь и отказы — аналогично, из реестров `COMMS`, `FAULTS`.
9. `MetricsRecorder`.

`lookup(...)` выдаёт понятную ошибку со списком доступных имён, если в конфиге опечатка.

### `_build_agents() -> list[Agent]`
Создаёт `agents.count` агентов:
- **Дом**: в ряд вдоль оси x вокруг точки `agents.home`, шаг `home_spacing`:
  `x_i = home_x + (i − (N−1)/2)·spacing`. Результат обрезается границами мира.
- **Высота (эшелонирование)**: `alt_i = altitude + (i mod altitude_layers)·altitude_step`.
  По умолчанию 10 эшелонов по 3 м, то есть 50, 53, …, 77 м. Это реализация
  «эшелонирование по высоте ±3 м» из NFR-9.
- **Модель движения**: `KinematicMotion` или `DoubleIntegratorMotion` по `agents.motion`;
  предельная скорость поворота переводится из градусов в радианы.
- **Энергия**: ёмкость `battery_wh · 3600` Дж, резерв, стоимость взлёта и посадки.
- Фильтр безопасности передаётся агенту, только если он не «пустой» (`PassthroughSafety`):
  так не тратится время на лишние снимки соседей.

### `snapshot(t) -> WorldSnapshot`
Неизменяемый снимок для распределителя: время, `AgentState` каждого агента, границы,
бесполётные зоны.

### `_allocate(t, reason)`
1. `allocator.allocate(snapshot, tasks)` → назначение `{agent_id: [task_id, ...]}`.
2. `_check_assignment(...)` — проверка.
3. Каждому агенту `set_plan(...)`. **Назначение полное**: агент, которого нет в ответе,
   получает пустой план.
4. Событие `allocation` с причиной (`initial` / `trigger`) и самим назначением. Так в
   журнале видно, почему и как переназначали (FR-11 «причины решений»).

### `_check_assignment(assignment)`
Бросает `AssignmentError`, если:
- указан несуществующий агент;
- задачи назначены отказавшему агенту;
- указана несуществующая задача;
- **одна задача назначена дважды** — прямая защита от `N_dup > 0` (жёсткое ограничение);
- назначена уже выполненная задача.

### `run() -> RunResult` — главный цикл
```
n_steps = round(t_end / dt)
_allocate(0, "initial");  все агенты takeoff(0);  metrics.record(0)
for k in 0 .. n_steps-1:            t = k·dt
    1) отказы:   for (aid, reason) in faults.inject_due(t): agent.fail(t, reason, tasks)
    2) связь:    comms.step(t, agents)
    3) перераспределение: new_events = события с прошлого шага
                 if allocator.should_trigger(t, tasks, new_events) and comms.gcs_reachable():
                     _allocate(t, "trigger")
    4) агенты:   для каждого по порядку id:
                     agent.step(t, dt, соседи, tasks)
                     world.observe(agent, sensor)
                     если агент разрядился насмерть на этом шаге — освободить его задачи
    5) мир:      world.step(dt); check_separation; check_geofence
                 если столкновение и strict — CollisionError
    k += 1;  metrics.record(k·dt)
    если никто не в воздухе — стоп
```
Подробности, важные для корректности:
- `t = k·dt`, а не `t += dt` — точное время событий.
- Соседи (`AgentState` соседей по связи) собираются только при активном фильтре
  безопасности — экономия времени.
- Перераспределение не каждый шаг, а **по событию** (антипаттерн 4 из `docs/02`: «не
  делать централизованный распределитель синхронным for-циклом на каждом шаге»).
- Разряд батареи насмерть (`battery_depleted`) обрабатывается здесь, потому что внутри
  `Agent._consume` нет доступа к списку задач.

В конце:
- `extra` — служебные поля: сценарий, зерно, распределитель, модель движения, N, первые
  16 символов хэша конфига, число шагов и перераспределений, статистика связи;
- `metrics.finalize(...)` — итоговые метрики;
- `wall_time_s` и `realtime_factor` (во сколько раз быстрее реального времени) — для
  проверки NFR-1. **В детерминизм не входят.**

### `_manifest() -> dict`
Всё, что нужно, чтобы воспроизвести прогон: сценарий, зерно, распределитель, **полный
SHA-256 хэш конфига**, список потоков RNG, версия пакета, **коммит git**, версии Python,
NumPy и pydantic, время создания, **весь конфиг целиком**. Пишется в `run_manifest.json`.

## `_git_commit() -> str | None`
Вызывает `git rev-parse HEAD` в папке пакета. Если git недоступен — `None`, прогон не
падает.

## `run(scenario, seed, allocator=None) -> RunResult`
Короткая точка входа: `SimLoop(...).run()`. Именно её используют CLI, стенд и тесты.

## Пример
```python
from swarmsim import load_scenario, apply_overrides, run
scn = apply_overrides(load_scenario("scenarios/baseline.yaml"), {"agents.count": 20})
res = run(scn, seed=1)
print(res.summary["T_detect_90"], res.summary["valid"])
```
