# Служебные файлы: `pyproject.toml`, `.gitignore`, `.vscode/`, `scenarios/` ★

## `pyproject.toml` — описание пакета
- `[build-system]` — сборка через setuptools.
- `[project]` — имя `swarmcore`, версия, Python ≥ 3.10, зависимости `numpy`, `pydantic>=2`,
  `pyyaml`.
- `[project.optional-dependencies] dev` — `pytest`, `matplotlib` (ставятся через
  `pip install -e ".[dev]"`).
- `[project.scripts]` — команда `swarmcore` → `apps.cli:main`.
- `[tool.setuptools.packages.find]` — в пакет входят `swarmsim`, `apps`, `experiments`.
- `[tool.pytest.ini_options]` — тесты ищутся в `tests/`.

Установка `-e` (editable) означает, что правки кода сразу видны без переустановки.

## `.gitignore`
Не коммитятся: `.venv/` (виртуальное окружение, у каждого своё), `__pycache__/`,
`*.egg-info/`, `.pytest_cache/`, `runs/` (результаты прогонов), `.DS_Store`.

## `.vscode/settings.json`
Интерпретатор — `.venv\Scripts\python.exe`; автоактивация окружения в терминале; тесты
через pytest в панели Testing; кодировка UTF-8.

## `.vscode/launch.json`
Четыре конфигурации для «Run and Debug» (F5): прогон `baseline`, прогон `failures`,
матрица `T_detect_90` от N, pytest. Все запускаются как модули (`apps.cli`, `pytest`) из
корня проекта, поэтому работают точки останова.

## `scenarios/*.yaml`
| Файл | Что проверяет |
|---|---|
| `baseline.yaml` | базовая линия из `docs/03`: 1000×1000 м, равномерные цели, 10 агентов, без отказов, идеальный канал. Все параметры выписаны явно с комментариями |
| `failures.yaml` | MVP-1: отказ 2 из 10 агентов (t = 120 и 240 с), цели в 3 горячих зонах, 5 новых целей на 300 с, двойной интегратор |
| `nofly.yaml` | бесполётная зона 200×200 м в центре; показывает, что переходы нужно планировать (`N_geo > 0`) |

Сценарий можно не копировать, а переопределять: `--set agents.count=20`.
