# `swarmsim/io.py` — запись результатов ★

**Назначение.** Сохранить результат прогона (`RunResult`) в папку в форматах, удобных для
анализа (CSV — таблицы, pandas; JSON — машинное чтение; JSONL — журнал построчно).

**Импорты.** `csv`, `json`, `pathlib`, `typing.Iterable`, `RunResult`.

## `write_csv(path, rows)`
Пишет список словарей в CSV (UTF-8). Заголовок — **объединение ключей всех строк** в
порядке первого появления, поэтому строки с разным набором полей (например, из разных
распределителей) не теряются. `None` пишется пустой ячейкой. Папка создаётся при
необходимости.

## `write_run(result, out_dir) -> Path`
Создаёт в `out_dir`:

| Файл | Содержимое |
|---|---|
| `summary.csv` | одна строка итоговых метрик |
| `summary.json` | то же в JSON (с отступами, кириллица без экранирования) |
| `timeseries.csv` | ряды: `t, detected_frac, expected_detect_frac, tasks_done, airborne, energy_used_wh` |
| `trajectories.csv` | `t, agent, x, y, alt, vx, vy, mode, task, energy_frac` — если включено `output.trajectories` |
| `events.jsonl` | журнал: одно событие JSON на строку |
| `run_manifest.json` | всё для воспроизведения (см. [sim.md](sim.md), `_manifest`) |

`trajectories.csv` подходит для внешнего проигрывателя, например 3D-сцены в Unity: в нём
есть время, координаты, высота, скорость и режим каждого аппарата.
