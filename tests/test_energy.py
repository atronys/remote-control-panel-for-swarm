import numpy as np
import pytest

from swarmsim.agents.energy import RotaryWingEnergy
from swarmsim.scenario import EnergyConfig


@pytest.fixture
def e():
    return RotaryWingEnergy(EnergyConfig())


def test_hover_power(e):
    assert e.power(0.0) == pytest.approx(e.cfg.P0 + e.cfg.Pi)


def test_power_curve_shape(e):
    """U-образная кривая: минимум мощности не в висении; скорость макс. дальности выше."""
    v_mp, v_mr = e.v_min_power(), e.v_max_range()
    assert 0 < v_mp < v_mr
    assert e.power(v_mp) < e.hover_power
    assert e.power(30) > e.hover_power
    # для параметров Zeng et al. 2019: ~10.2 и ~18.3 м/с (а не 1–3 и 10–15, как в docs/02)
    assert v_mp == pytest.approx(10.2, abs=0.2)
    assert v_mr == pytest.approx(18.3, abs=0.3)


def test_energy_per_meter_minimum_at_v_max_range(e):
    v = np.linspace(1, 30, 2901)
    assert v[np.argmin(e.energy_per_meter(v))] == pytest.approx(e.v_max_range(), abs=0.05)


def test_endurance_matches_nfr11(e):
    """NFR-11: ≥ 15 мин на аккумулятор 55 Вт·ч при крейсерской 12 м/с с резервом 20 %."""
    minutes = 55 * 3600 * 0.8 / e.power(12) / 60
    assert 15 <= minutes <= 25
