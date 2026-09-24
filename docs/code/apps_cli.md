# `apps/cli.py` — командный интерфейс `swarmcore` ★

**Назначение.** Запуск без написания кода (P1, задача 9: «Командный интерфейс: run,
compare»). Команда `swarmcore` регистрируется в `pyproject.toml`
(`[project.scripts] swarmcore = "apps.cli:main"`) и появляется после `pip install -e .`.

**Импорты.** `argparse`, `sys`, `pathlib`; из проекта — `run_matrix` (серии и агрегация),
`write_run`, `load_scenario`, `apply_overrides`, `parse_override`, `run`.

## `KEY_METRICS`
Метрики, которые `run` печатает в консоль.

## `cmd_run(args)` — `swarmcore run`
```
swarmcore run --scenario scenarios/baseline.yaml [--allocator NAME] [--seed 1]
              [--set key=value ...] [--out DIR] [--plot]
```
Загружает сценарий, применяет `--set` и `--allocator`, делает прогон, пишет файлы
(`write_run`), печатает ключевые метрики. По умолчанию папка —
`runs/<сценарий>_<распределитель>_s<зерно>`. С `--plot` рисует `trajectories.png`.
**Код возврата 2, если прогон невалиден** — удобно для CI.

Это ровно контрольная точка MVP-0 из `docs/02`: «`swarmcore run --scenario X --allocator …
--seed 1` даёт CSV с метриками; повтор даёт те же числа».

## `cmd_compare(args)` — `swarmcore compare`
```
swarmcore compare --scenario S --allocators a,b [--seeds 1-20] [--metrics CR,T_detect_90,E_total_Wh] [--jobs N]
```
Прогоняет каждый распределитель **на одних и тех же зёрнах** (парный дизайн) и печатает
таблицу: среднее, ±95 % ДИ, число валидных прогонов.

## `swarmcore matrix`
Аргументы и логика — в [experiments_run_matrix.md](experiments_run_matrix.md).

## `main(argv=None)`
Настраивает stdout/stderr на замену непечатаемых символов (консоль Windows в cp1251 не
может вывести, например, «→»), строит подкоманды `argparse` и вызывает нужную.
