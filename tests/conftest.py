import pytest

from swarmsim.scenario import Scenario


@pytest.fixture
def small():
    """Маленький быстрый сценарий: 400×400 м, 4 агента."""
    return Scenario.model_validate({
        "name": "small", "t_end": 900, "dt": 0.1,
        "world": {"size": [400, 400], "cell_size": 25, "targets": 10},
        "agents": {"count": 4, "home": [200, 0]},
        "mission": {"sector_size": 200, "track_spacing": 25},
    })
