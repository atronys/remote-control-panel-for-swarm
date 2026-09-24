"""Пульт в браузере: параметры формы → переопределения сценария."""
from apps.serve import build_overrides


def test_form_to_overrides():
    scen, o, seed = build_overrides({"scenario": "demo", "allocator": "cbba", "agents": "12", "fail_rate": "0.2",
                                     "silent": True, "range": "700", "loss": "0.1", "outage": True,
                                     "outage_from": "200", "outage_to": "300", "wind": True, "orca": True,
                                     "one_layer": False, "hotspots": True, "seed": "42"})
    assert scen == "demo" and seed == 42
    assert o["allocator"] == {"kind": "cbba"} and o["agents.count"] == 12
    assert o["comms"]["kind"] == "network" and o["comms"]["gcs_outages"] == [[200.0, 300.0]]
    assert o["faults.detect"] == "silent" and o["realism.enabled"] and o["safety.filter"] == "orca"


def test_empty_seed_is_random_and_bad_scenario_falls_back():
    s1 = build_overrides({"scenario": "../../etc", "seed": ""})
    s2 = build_overrides({"scenario": "demo", "seed": ""})
    assert s1[0] == "demo" and isinstance(s1[2], int) and 0 <= s2[2] < 1_000_000
