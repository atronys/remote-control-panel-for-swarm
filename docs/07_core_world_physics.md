# Ядро симуляции, мир и физика БПЛА — что сделано и как пользоваться

Владелец: R1 (ядро и стенд) + физика аппарата. Статус: MVP-0 готов, двойной интегратор
из MVP-2 тоже готов (переключается в конфиге).

## Быстрый старт

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest                            # 31 тест, ~10 с

swarmcore run --scenario scenarios/baseline.yaml --seed 1 --plot
swarmcore matrix --scenario scenarios/baseline.yaml --vary agents.count=5,10,20 \
    --set agents.altitude_layers=20 --seeds 1-20 --jobs 6 --plot agents.count:T_detect_90
swarmcore compare --scenario scenarios/failures.yaml --allocators static_roundrobin,<ваш> --seeds 1-20
```

Один прогон пишет в `runs/<имя>/`: `summary.csv/json`, `timeseries.csv`, `trajectories.csv`,
`events.jsonl` (журнал событий, «чёрный ящик»), `run_manifest.json` (хэш конфига, зерно,
коммит, версии). Любой параметр переопределяется без правки YAML: `--set agents.count=20`.

## Что где

| Файл | Что | Владелец |
|---|---|---|
| `swarmsim/sim.py` | детерминированный цикл `SimLoop.run()`, журнал событий, проверка назначений | R1 |
| `swarmsim/world.py` | границы, бесполётные зоны, карта вероятности целей, цели (в т.ч. появляющиеся), `observe()`, сепарация, геозоны | R1 |
| `swarmsim/scenario.py` | pydantic-схема сценария, YAML, переопределения, хэш конфига | R1 |
| `swarmsim/agents/base.py` | `Agent` (следование плану, энергия, резерв, возврат домой) и интерфейс `MotionSource` | R1 / физика |
| `swarmsim/agents/kinematics.py` | `KinematicMotion` (MVP-0) и `DoubleIntegratorMotion` (MVP-2) | физика |
| `swarmsim/agents/energy.py` | мощность мультиротора по Zeng et al. | физика |
| `experiments/run_matrix.py`, `apps/cli.py` | стенд прогонов, `run` / `compare` / `matrix` | R1 |
| `swarmsim/interfaces.py`, `datatypes.py` | интерфейсы и типы — **контракт между командами** | все |
| `alloc/roundrobin.py`, `missions/search.py`, `sensing.py`, `comms.py`, `faults.py`, `metrics.py`, `agents/safety.py` | **базовые заглушки**, чтобы ядро работало; владельцы их заменяют | R2 / R3 / R4 |

## Модели

**Движение.** `agents.motion: kinematic` — `p += v_c·normalize(wp − p)·dt` без перелёта.
`double_integrator` — `ṗ = v, v̇ = a, |a| ≤ a_max, |v| ≤ v_max, |ψ̇| ≤ ψ̇_max`, наведение с
профилем торможения `sqrt(2·a_max·d)`. Ниже 0.5 м/с аппарат может развернуться на месте
(мультиротор). Ограничения проверяются тестом на 3000 случайных командах.

**Энергия.** `P(v)` — формула Zeng et al. из `02`, раздел 3.2, параметры из статьи. Взлёт и посадка стоят
фиксированную энергию. Агент уходит домой, если `e − E_домой < reserve·ёмкость`. Батарея
55 Вт·ч при 12 м/с даёт ~20 мин полёта до резерва 20 % (NFR-11 требует ≥ 15).

**Мир.** 2D, сетка ячеек. Цели генерируются из априорной карты (`uniform` / `hotspots`)
отдельным потоком RNG, поэтому **при одном зерне мир одинаков для любых N и любых
распределителей** (парный дизайн из `03`). Обнаружение по `02`, раздел 3.3:
`Pd = 1 − exp(−W·L/A_cell)`, `W` от GSD и высоты, накопление по проходам.

**Жёсткие ограничения.** Столкновение — горизонтально < 5 м **и** по высоте < 3 м;
опасное сближение — < 10 м; геозона — отрезок шага пересёк бесполётную зону или границу.
Нарушение → `valid = false` в summary (или исключение при `safety.strict: true`).

## Как подключить свой модуль (R2 / R3)

```python
from swarmsim.interfaces import Allocator, register_allocator

@register_allocator("cbba")
class CBBA(Allocator):
    def allocate(self, snapshot, tasks):          # -> {agent_id: [task_id, ...]}
        ...
    def should_trigger(self, t, tasks, events):   # например, при "task_released"/"new_targets"
        return any(e["type"] in ("task_released", "new_targets") for e in events)
```

Импортировать модуль в `swarmsim/alloc/__init__.py` и указать `allocator: {kind: cbba, ...}`
в сценарии (дополнительные ключи придут в `self.params`). Так же регистрируются
`register_comms`, `register_faults`, `register_mission`, `register_safety`.

Правила ядра для назначения: одна задача — не более одному агенту, не назначать выполненные
задачи и отказавшим агентам (иначе `AssignmentError`). Назначение **полное**: агент, которого
нет в ответе, получает пустой план. Задачи, выпавшие из плана агента, освобождаются
(`task_released`). События для `should_trigger`: `agent_failed`, `task_released`,
`low_battery_rtl`, `battery_depleted`, `new_targets`, `target_detected`, `task_done`, …

Детерминизм: используйте только переданный `rng` и модельное время. Новые потоки RNG
добавлять только в конец `RNG_STREAMS` в `sim.py`.

## Первые результаты (static_roundrobin, 1000×1000 м, 20 зёрен)

| N | T_detect(90 %), с | CR | E_total, Вт·ч |
|---|---|---|---|
| 3 | не достигнуто (не хватает батареи) | 0.60 | 132 |
| 5 | 797 ± 14 | 1.00 | 169 |
| 10 | 491 ± 13 | 1.00 | 199 |
| 20 | 338 ± 13 | 1.00 | 241 |

Выигрыш суб-линеен, как и предсказывает `T(N) ≈ A/(N·v·w) + c(N)` из `01`.
Отказ 2 из 10 агентов (`scenarios/failures.yaml`) при статическом распределении даёт
CR = 0.84 — это и есть базовая линия для динамического распределителя.

## Найденные проблемы (для обсуждения на синке)

1. **Числа в `02`, раздел 3.2, не совпадают с формулой.** Для параметров Zeng et al. минимум мощности
   (макс. продолжительность) при **~10.2 м/с**, а не 1–3 м/с; максимум дальности при
   **~18.3 м/с**, а не 10–15. Сама формула верна, неверны только комментарии к ней.
   Кривая U-образная, висение — не самый экономный режим.
2. **Без разведения транзитов столкновения при N ≥ 20.** По умолчанию 10 эшелонов по 3 м;
   агенты i и i+10 на одном эшелоне пересекаются на переходах к секторам. Ядро это ловит
   (`N_col`, прогон invalid). Лечится стратегическим разведением / ORCA (R2, FR-5);
   временно — `--set agents.altitude_layers=20`.
3. **Переходы через бесполётные зоны** (`scenarios/nofly.yaml`, `N_geo > 0`). «Змейка»
   выкидывает точки внутри зоны, но маршрут между секторами прокладывается по прямой.
   Нужен планировщик/валидатор R2. Для него есть `World.segment_is_free(a, b)`.

## Что дальше по моей зоне

- `MavsdkMotion` поверх того же `MotionSource` (фаза P6, PX4 SITL).
- Ветер (воздушная ≠ путевая скорость в энергии), если команда решит, что он нужен.
- Проигрыватель `trajectories.csv` для демо (можно в Unity — формат уже подходит).

Подробное описание каждого файла кода, библиотек и источников — в [`docs/code/`](code/README.md).
