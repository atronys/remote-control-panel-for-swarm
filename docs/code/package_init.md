# `swarmsim/__init__.py`, `swarmsim/agents/__init__.py`, `alloc/__init__.py`, `missions/__init__.py` ★

## `swarmsim/__init__.py`
Точка входа пакета. Реэкспортирует главное, чтобы писать коротко:
```python
from swarmsim import Scenario, load_scenario, apply_overrides, SimLoop, RunResult, run
```
Здесь же `__version__ = "0.1.0"`, она попадает в манифест.

## `swarmsim/agents/__init__.py`
Реэкспорт `Agent`, `MotionCommand`, `MotionSource`, `RotaryWingEnergy`, `KinematicMotion`,
`DoubleIntegratorMotion`.

## `swarmsim/alloc/__init__.py`, `swarmsim/missions/__init__.py`
Импортируют модули с реализациями (`roundrobin`, `search`), чтобы сработали декораторы
регистрации. **Новый распределитель или миссию нужно добавить сюда же**, иначе ядро его не
увидит.

## `apps/__init__.py`, `experiments/__init__.py`
Пустые: делают папки пакетами, чтобы работали `python -m apps.cli` и
`python -m experiments.run_matrix`.
