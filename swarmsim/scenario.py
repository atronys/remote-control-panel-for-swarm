"""Схема сценария (pydantic) + загрузка YAML + хэш конфига для run_manifest.json."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PriorConfig(_Model):
    """Априорная карта вероятности целей."""

    kind: Literal["uniform", "hotspots"] = "uniform"
    n: int = Field(3, ge=1, description="число «горячих» зон")
    sigma: float = Field(100.0, gt=0, description="размер зоны, м")
    background: float = Field(0.1, ge=0, le=1, description="доля равномерного фона")


class NewTargetsEvent(_Model):
    t: float = Field(ge=0)
    count: int = Field(ge=1)


class NewTaskEvent(_Model):
    """FR-9: новая цель → задача осмотра точки с приоритетом (может прервать текущую задачу)."""

    t: float = Field(ge=0)
    x: float
    y: float
    priority: float = Field(5.0, gt=0)
    size: float = Field(40.0, gt=0, description="сторона квадрата осмотра, м")


class WorldConfig(_Model):
    size: tuple[float, float] = (1000.0, 1000.0)
    cell_size: float = Field(25.0, gt=0)
    nofly: list[list[tuple[float, float]]] = Field(default_factory=list,
                                                   description="многоугольники бесполётных зон")
    prior: PriorConfig = PriorConfig()
    targets: int = Field(30, ge=0)
    new_targets: list[NewTargetsEvent] = Field(default_factory=list)
    new_tasks: list[NewTaskEvent] = Field(default_factory=list)

    @field_validator("nofly")
    @classmethod
    def _polygons(cls, v):
        for poly in v:
            if len(poly) < 3:
                raise ValueError("бесполётная зона должна иметь >= 3 вершин")
        return v


class AgentsConfig(_Model):
    count: int = Field(10, ge=1)
    motion: Literal["kinematic", "double_integrator"] = "kinematic"
    v_cruise: float = Field(12.0, gt=0)
    v_max: float = Field(15.0, gt=0)
    a_max: float = Field(3.0, gt=0)
    yaw_rate_max_deg: float = Field(90.0, gt=0)
    altitude: float = Field(50.0, gt=0)
    altitude_step: float = Field(3.0, ge=0, description="эшелонирование по высоте")
    altitude_layers: int = Field(10, ge=1)
    home: tuple[float, float] = (500.0, 0.0)
    home_spacing: float = Field(15.0, gt=0)
    battery_wh: float = Field(55.0, gt=0)
    reserve: float = Field(0.2, ge=0, lt=1)
    arrive_radius: float = Field(2.0, gt=0)
    # локальный ремонт на борту (docs/05 P3): при нехватке энергии сбросить хвост плана,
    # а не всё сразу; RTL — только если не хватает на текущую задачу
    local_repair: bool = True
    repair_period: float = Field(5.0, gt=0)
    deviation_threshold: float = Field(30.0, gt=0, description="SR-5: отставание от плана, с")

    @model_validator(mode="after")
    def _speeds(self):
        if self.v_cruise > self.v_max:
            raise ValueError("v_cruise не может превышать v_max")
        return self


class EnergyConfig(_Model):
    """Модель мощности мультиротора (Zeng et al., 2019). Значения — из статьи."""

    P0: float = 79.86        # Вт, профильная мощность в висении
    Pi: float = 88.63        # Вт, индукционная мощность в висении
    U_tip: float = 120.0     # м/с, скорость кончика лопасти
    v0: float = 4.03         # м/с, индуцированная скорость в висении
    d0: float = 0.6          # коэффициент сопротивления фюзеляжа
    rho: float = 1.225       # кг/м³
    s: float = 0.05          # заполнение ротора
    A: float = 0.503         # м², площадь ротора
    E_takeoff: float = 2000.0  # Дж
    E_land: float = 1000.0     # Дж


class SensorConfig(_Model):
    fov_deg: float = Field(60.0, gt=0, lt=180)
    pixel_pitch_m: float = Field(3.0e-6, gt=0)
    focal_length_m: float = Field(8.0e-3, gt=0)
    gsd_target_m: float = Field(0.02, gt=0, description="нужное разрешение для распознавания")
    gsd_max_m: float = Field(0.05, gt=0, description="хуже этого — цель не видна")


class MissionConfig(_Model):
    kind: str = "lawnmower_grid"
    sector_size: float = Field(200.0, gt=0)
    track_spacing: float = Field(20.0, gt=0)


class ComponentConfig(_Model):
    """Подключаемый компонент: kind — имя в реестре, остальное — его параметры."""

    model_config = ConfigDict(extra="allow")
    kind: str


class FaultEvent(_Model):
    t: float = Field(ge=0)
    agent: int = Field(ge=0)
    type: Literal["fail", "degrade"] = "fail"       # отказ | деградация (отклонение от плана, SR-5)
    detect: Literal["reported", "silent"] = "reported"   # борт успел сообщить | заметят по heartbeat
    factor: float = Field(0.5, gt=0, le=1)          # degrade: доля крейсерской скорости


class FaultsConfig(_Model):
    kind: str = "scheduled"                          # scheduled — по списку; random — с интенсивностью
    scheduled: list[FaultEvent] = Field(default_factory=list)
    rate: float = Field(0.0, ge=0, le=1, description="λ: доля агентов, отказывающих за миссию (random)")
    t_range: tuple[float, float] = (60.0, 400.0)     # окно случайных отказов, с
    detect: Literal["reported", "silent"] = "reported"   # для random
    detect_timeout: float = Field(1.5, gt=0, description="тихий отказ замечают по пропаже heartbeat, с")


class SafetyConfig(_Model):
    filter: str = "none"                              # локальное избегание (R2): имя в реестре
    min_separation: float = Field(5.0, gt=0)          # NFR-9
    vertical_separation: float = Field(3.0, ge=0)
    # тактическое избегание (filter: orca): соседи — по бортовому обзору, не по каналу
    sense_range: float = Field(60.0, gt=0, description="радиус обзора соседей, м")
    orca_tau: float = Field(5.0, gt=0, description="горизонт ORCA, с")
    orca_radius_factor: float = Field(2.2, gt=0, description="радиус агента = factor·min_separation/2")
    strict: bool = False    # True — исключение при столкновении; False — прогон помечается invalid
    # SR-6: repair — недопустимый по энергии хвост плана отрезается; log — только записать
    # нарушение (остаётся бортовая защита по заряду); off — не проверять
    plan_validation: Literal["repair", "log", "off"] = "repair"


class PlanningConfig(_Model):
    """Общая для всех распределителей целевая функция планирования (docs/03 §8: одинаковые условия)."""

    discount: float = Field(0.998, gt=0, le=1, description="λ: ценность задачи ∝ λ^(время завершения)")
    value: Literal["prior", "uniform"] = Field(
        "prior", description="ценность сектора: доля вероятности целей (Pd-планирование) или 1 (покрытие)")


class ObjectiveConfig(_Model):
    """U(M) = Σ priority·1[выполнено до deadline] − λ_E·E_total − λ_T·makespan (docs/03, раздел 4)."""

    deadline: float = Field(900.0, gt=0, description="«вовремя», с (NFR-11: миссия ≤ 15 мин)")
    lambda_E: float = Field(0.01, ge=0, description="штраф за Вт·ч")
    lambda_T: float = Field(0.005, ge=0, description="штраф за секунду makespan")


class OutputConfig(_Model):
    record_every: float = Field(1.0, gt=0)
    trajectories: bool = True


class RealismConfig(_Model):
    """Возмущения реального мира для демонстраций (в экспериментах P2–P5 выключены).

    Ветер: постоянный на прогон (сила 0..wind_max, случайное направление) + порывы у каждого
    дрона (процесс Орнштейна–Уленбека, σ = gust_sigma, τ = gust_tau). Сносят дрон, автопилот
    доворачивает на точку — траектории «живые» и разные от прогона к прогону.
    speed_spread — разброс крейсерской скорости дронов (доля, нормальное распределение)."""

    enabled: bool = False
    wind_max: float = Field(3.0, ge=0, description="м/с")
    gust_sigma: float = Field(0.8, ge=0, description="м/с")
    gust_tau: float = Field(8.0, gt=0, description="с")
    speed_spread: float = Field(0.05, ge=0, le=0.3)


class Scenario(_Model):
    name: str = "unnamed"
    t_end: float = Field(1800.0, gt=0)
    dt: float = Field(0.1, gt=0)
    world: WorldConfig = WorldConfig()
    agents: AgentsConfig = AgentsConfig()
    energy: EnergyConfig = EnergyConfig()
    sensor: SensorConfig = SensorConfig()
    mission: MissionConfig = MissionConfig()
    allocator: ComponentConfig = ComponentConfig(kind="static_roundrobin")
    comms: ComponentConfig = ComponentConfig(kind="ideal")
    faults: FaultsConfig = FaultsConfig()
    safety: SafetyConfig = SafetyConfig()
    output: OutputConfig = OutputConfig()
    planning: PlanningConfig = PlanningConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    realism: RealismConfig = RealismConfig()

    @model_validator(mode="after")
    def _time(self):
        if self.dt >= self.t_end:
            raise ValueError("dt должен быть меньше t_end")
        return self

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()


def load_scenario(path: str | Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Scenario.model_validate(data)


def apply_overrides(scenario: Scenario, overrides: dict[str, Any]) -> Scenario:
    """Переопределение полей по точечному пути: {"agents.count": 20, "allocator.kind": "cbba"}."""
    data = copy.deepcopy(scenario.model_dump(mode="json"))
    for dotted, value in overrides.items():
        node = data
        keys = dotted.split(".")
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise KeyError(f"нет такого раздела конфига: {dotted}")
            node = node[k]
        node[keys[-1]] = value
    return Scenario.model_validate(data)


def parse_override(text: str) -> tuple[str, Any]:
    """'agents.count=20' -> ('agents.count', 20); значение разбирается как YAML."""
    if "=" not in text:
        raise ValueError(f"ожидается key=value, получено {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)
