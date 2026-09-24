"""SwarmCore simulator: детерминированное ядро, мир и физика БПЛА."""
from .scenario import Scenario, apply_overrides, load_scenario
from .sim import RunResult, SimLoop, run

__all__ = ["Scenario", "load_scenario", "apply_overrides", "SimLoop", "RunResult", "run"]
__version__ = "0.1.0"
