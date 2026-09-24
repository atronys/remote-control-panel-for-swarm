from pathlib import Path

import pytest
from pydantic import ValidationError

from swarmsim.scenario import Scenario, apply_overrides, load_scenario, parse_override

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "scenarios").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_scenarios_are_valid(path):
    load_scenario(path)


def test_validation_errors():
    with pytest.raises(ValidationError):
        Scenario.model_validate({"agents": {"v_cruise": 20, "v_max": 15}})
    with pytest.raises(ValidationError):
        Scenario.model_validate({"agents": {"cont": 5}})       # опечатка в ключе
    with pytest.raises(ValidationError):
        Scenario.model_validate({"dt": 10, "t_end": 5})


def test_overrides_and_hash():
    base = Scenario()
    other = apply_overrides(base, dict([parse_override("agents.count=20")]))
    assert other.agents.count == 20
    assert other.config_hash() != base.config_hash()
    assert apply_overrides(base, {}).config_hash() == base.config_hash()
    with pytest.raises(KeyError):
        apply_overrides(base, {"nope.x": 1})
