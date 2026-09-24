# Документация кода SwarmCore

Здесь описан **каждый файл кода** репозитория: зачем он нужен, какие в нём классы и функции,
что они принимают и возвращают, какие формулы и методики за ними стоят.

Обозначения:
- ★ — зона «ядро + мир + физика БПЛА» (R1 + физика аппарата). Описано максимально подробно.
- ○ — базовая заглушка для чужой зоны (R2 / R3 / R4). Работает, но предназначена для замены
  владельцем модуля. Описано короче.

Сначала стоит прочитать [`00_libraries_and_sources.md`](00_libraries_and_sources.md): какие
библиотеки подключены, какие статьи и методики использованы и откуда взяты формулы.

## Порядок чтения

1. [`00_libraries_and_sources.md`](00_libraries_and_sources.md) — библиотеки, статьи, методики.
2. [`01_architecture.md`](01_architecture.md) — как всё связано: поток данных, цикл, детерминизм.
3. Файлы ядра, мира и физики (★) — ниже по списку.
4. Заглушки (○), CLI, тесты, конфиги.

## Список файлов

### Пакет `swarmsim/` — симулятор

| Файл кода | Документ | Зона |
|---|---|---|
| `swarmsim/sim.py` | [sim.md](sim.md) | ★ ядро: детерминированный цикл |
| `swarmsim/world.py` | [world.md](world.md) | ★ мир |
| `swarmsim/scenario.py` | [scenario.md](scenario.md) | ★ конфиги сценариев |
| `swarmsim/datatypes.py` | [datatypes.md](datatypes.md) | ★ общие типы данных |
| `swarmsim/interfaces.py` | [interfaces.md](interfaces.md) | ★ интерфейсы модулей и реестр |
| `swarmsim/geometry.py` | [geometry.md](geometry.md) | ★ 2D-геометрия |
| `swarmsim/io.py` | [io.md](io.md) | ★ запись результатов |
| `swarmsim/__init__.py` | [package_init.md](package_init.md) | ★ точка входа пакета |
| `swarmsim/agents/base.py` | [agents_base.md](agents_base.md) | ★ агент и интерфейс движения |
| `swarmsim/agents/kinematics.py` | [agents_kinematics.md](agents_kinematics.md) | ★ физика движения |
| `swarmsim/agents/energy.py` | [agents_energy.md](agents_energy.md) | ★ физика энергопотребления |
| `swarmsim/agents/safety.py` | [agents_safety.md](agents_safety.md) | ORCA, тактическое избегание (P5) |
| `swarmsim/alloc/roundrobin.py` | [alloc_roundrobin.md](alloc_roundrobin.md) | ○ R2 |
| `swarmsim/alloc/central.py`, `cpsat.py`, `cbba.py` | [alloc_central_cbba.md](alloc_central_cbba.md) | B0–B5 (P2–P4) |
| `swarmsim/planning/` | [planning.md](planning.md) | общая целевая функция, маршруты, проверка плана (P2) |
| `swarmsim/network.py`, `knowledge.py`, `decentral.py` | [network_knowledge.md](network_knowledge.md) | канал, реплики, CBBA/гибрид (P4) |
| `experiments/gates.py`, `campaign.py`, `stats.py` | [experiments_gates.md](experiments_gates.md) | контрольные точки P2–P5 |
| `swarmsim/missions/search.py` | [missions_search.md](missions_search.md) | ○ миссия |
| `swarmsim/sensing.py` | [sensing.md](sensing.md) | ○ R3 |
| `swarmsim/comms.py` | [comms.md](comms.md) | ○ R3 |
| `swarmsim/faults.py` | [faults.md](faults.md) | ○ R3 |
| `swarmsim/metrics.py` | [metrics.md](metrics.md) | ○ R4 |

### Запуск, эксперименты, тесты, конфиги

| Файл кода | Документ | Зона |
|---|---|---|
| `apps/cli.py` | [apps_cli.md](apps_cli.md) | ★ командный интерфейс |
| `apps/viz.py` | [apps_viz.md](apps_viz.md) | ★ картинка траекторий |
| `experiments/run_matrix.py` | [experiments_run_matrix.md](experiments_run_matrix.md) | ★ стенд прогонов |
| `tests/*.py` | [tests.md](tests.md) | ★ автотесты |
| `pyproject.toml`, `.gitignore`, `.vscode/*`, `scenarios/*.yaml` | [project_files.md](project_files.md) | ★ служебные файлы |

## Глоссарий

| Термин | Значение |
|---|---|
| Агент | один БПЛА в симуляции |
| Задача (Task) | единица работы; сейчас — «облететь сектор змейкой» |
| План | упорядоченный список задач агента |
| Распределитель (Allocator) | алгоритм, который раздаёт задачи агентам |
| Зерно (seed) | число, от которого строятся все случайные величины прогона |
| Прогон (run) | одна симуляция с одним сценарием и одним зерном |
| Детерминизм | то же зерно + тот же конфиг → побитово те же результаты |
| Парный дизайн | сравнение методов на одинаковых мирах (одни и те же зёрна) |
| `dt` | шаг модельного времени (по умолчанию 0.1 с) |
| RTL | Return To Launch — возврат домой |
| GSD | Ground Sample Distance — сколько метров земли приходится на один пиксель камеры |
| `Pd` | вероятность обнаружения цели |
| CR | Completion Rate — доля выполненных задач |
| `T_detect(90%)` | модельное время, когда обнаружено 90 % целей |
| invalid | прогон, нарушивший жёсткое ограничение (столкновение, геозона, дубль задачи) |
