# `experiments/run_matrix.py` — испытательный стенд прогонов ★

**Назначение.** «Главный артефакт проекта, а не визуализация» (`docs/02`, раздел 1.1, п. 3):
матрица «варианты конфига × распределители × зёрна» → CSV + манифест → график. Закрывает
контрольную точку P1: «20 прогонов выполняются одной командой и дают готовый график».

**Импорты.** `argparse`, `itertools`, `json`, `math`, `statistics`, `concurrent.futures`,
`datetime`, `pathlib`, `yaml`; из проекта — `write_csv`, `apply_overrides`,
`load_scenario`, `parse_override`, `run`; matplotlib — только внутри `plot`.

## Функции

### `parse_seeds(text) -> list[int]`
`"1-20"` → 1…20; `"1,2,5"`; `"1-5,10"`.

### `parse_vary(items) -> dict`
`["agents.count=5,10,20"]` → `{"agents.count": [5, 10, 20]}`; значения разбираются как
YAML.

### `_one(job) -> dict`
Один прогон: загрузить сценарий, применить переопределения, `run`, вернуть summary с
добавленными колонками `cfg:<ключ>`. Функция верхнего уровня — так её можно передавать в
другие процессы.

### `run_matrix(scenario_path, seeds, vary, allocators, fixed, jobs) -> list[dict]`
1. Распределители добавляются как ещё одно измерение `allocator.kind`.
2. Все комбинации строятся через `itertools.product`.
3. Для каждой комбинации — все зёрна. **Одни и те же зёрна во всех ячейках**, поэтому
   сравнение идёт на одинаковых мирах (парный дизайн).
4. `jobs > 1` — параллельно в `ProcessPoolExecutor` (отдельные процессы обходят GIL).
   Результаты не зависят от числа процессов: каждый прогон детерминирован сам по себе.

### `aggregate(rows, group_by, metric) -> list[dict]`
Группирует строки и считает по **валидным** прогонам: `mean`,
`ci95 = 1.96·stdev/√n` (нормальное приближение), `n_valid`, `n_runs`. Невалидные в
среднее не входят, но видны по разнице `n_runs − n_valid`.

### `plot(rows, x, y, out_png)`
Линия на каждый распределитель: среднее `y` от `x` с «усами» ±95 % ДИ.

### `add_matrix_args(p)`, `matrix_command(args)`, `main(argv)`
Аргументы: `--scenario`, `--seeds`, `--vary` (повторяемый), `--set` (повторяемый,
фиксированные значения), `--allocators`, `--jobs`, `--out`, `--plot x:y`.
`matrix_command` пишет `results.csv` и `matrix_manifest.json` (сценарий, зёрна, варианты,
хэши конфигов, время) и печатает, сколько прогонов невалидны.

## Пример
```
swarmcore matrix --scenario scenarios/baseline.yaml --vary agents.count=5,10,20 \
  --set agents.altitude_layers=20 --seeds 1-20 --jobs 6 --plot agents.count:T_detect_90
```
60 прогонов примерно за минуту. Результат — таблица в `docs/07_core_world_physics.md`.
