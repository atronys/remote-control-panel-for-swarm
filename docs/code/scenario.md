# `swarmsim/scenario.py` — схема и загрузка сценария ★

**Назначение.** Описать **всё, что задаёт прогон**, одной проверяемой структурой: мир,
агентов, энергию, сенсор, миссию, плагины, ограничения, вывод. Загрузить её из YAML,
переопределить отдельные поля из командной строки и посчитать хэш для манифеста.
Реализует решение `docs/02`, раздел 5: «Конфиги — pydantic + YAML: валидация сценариев и
воспроизводимость».

**Импорты.** `pydantic` (`BaseModel`, `ConfigDict`, `Field`, `field_validator`,
`model_validator`), `yaml` (PyYAML), `hashlib`, `json`, `copy`, `pathlib`, `typing.Literal`.

## Как работает pydantic здесь
Каждый раздел конфига — класс-наследник `BaseModel`. У полей есть тип, значение по
умолчанию и ограничения (`Field(gt=0)` — строго больше нуля, `ge`, `lt`, `le`). При
загрузке pydantic:
- приводит типы (`"10"` → `10`, список → кортеж);
- проверяет ограничения и пишет понятную ошибку с путём до поля;
- благодаря `extra="forbid"` **запрещает неизвестные ключи**, так что опечатка `cont: 5`
  вместо `count: 5` не пройдёт молча.

## Классы конфига

### `_Model`
Базовый класс с `extra="forbid"` для всех разделов.

### `PriorConfig` — карта вероятности целей
`kind`: `uniform` | `hotspots`; `n` — число горячих зон; `sigma` — их размер, м;
`background` — доля равномерного фона `[0, 1]`.

### `NewTargetsEvent` — `t` (≥ 0), `count` (≥ 1): сколько целей появится в момент `t`.

### `WorldConfig`
`size` `(W, H)`, м; `cell_size`, м; `nofly` — список многоугольников `[[x, y], ...]`;
`prior`; `targets` — число начальных целей; `new_targets` — список событий.
Валидатор `_polygons`: у каждого многоугольника ≥ 3 вершин.

### `AgentsConfig`
| Поле | По умолчанию | Смысл |
|---|---|---|
| `count` | 10 | число БПЛА |
| `motion` | `kinematic` | `kinematic` / `double_integrator` |
| `v_cruise` | 12 м/с | крейсерская скорость |
| `v_max`, `a_max`, `yaw_rate_max_deg` | 15, 3, 90 | ограничения двойного интегратора |
| `altitude`, `altitude_step`, `altitude_layers` | 50 м, 3 м, 10 | эшелонирование |
| `home`, `home_spacing` | (500, 0), 15 м | где стоят дома |
| `battery_wh`, `reserve` | 55 Вт·ч, 0.2 | батарея и резерв |
| `arrive_radius` | 2 м | «точка достигнута» |

Валидатор `_speeds`: `v_cruise ≤ v_max`.

### `EnergyConfig` — параметры модели Zeng и `E_takeoff`, `E_land` ([agents_energy.md](agents_energy.md)).

### `SensorConfig` — `fov_deg`, `pixel_pitch_m`, `focal_length_m`, `gsd_target_m`, `gsd_max_m` ([sensing.md](sensing.md)).

### `MissionConfig` — `kind`, `sector_size` (200 м), `track_spacing` (20 м, расстояние между галсами).

### `ComponentConfig` — подключаемый компонент
`kind` — имя в реестре. Здесь `extra="allow"`: **любые дополнительные ключи разрешены** и
передаются компоненту как параметры. Используется для `allocator` и `comms`:
```yaml
allocator: {kind: cbba, max_bundle: 3}   # max_bundle придёт в Allocator.params
```

### `FaultEvent`, `FaultsConfig` — `kind` и список `scheduled: [{t, agent}]`.

### `SafetyConfig`
`filter` (имя фильтра избегания, `none`), `min_separation` 5 м, `vertical_separation`
3 м (NFR-9), `strict` — падать ли при столкновении.

### `OutputConfig` — `record_every` (с), `trajectories` (писать ли траектории).

### `Scenario` — корень
`name`, `t_end` (с), `dt` (с) и все разделы выше. Валидатор `_time`: `dt < t_end`.

#### `config_hash() -> str`
SHA-256 от JSON всего конфига с отсортированными ключами. Одинаковый конфиг даёт
одинаковый хэш независимо от порядка ключей в YAML. Пишется в манифест и summary.

## Функции

### `load_scenario(path) -> Scenario`
Читает YAML в UTF-8 (`yaml.safe_load` — безопасная загрузка без выполнения кода) и
валидирует. Пустой файл даёт сценарий по умолчанию.

### `apply_overrides(scenario, overrides) -> Scenario`
Переопределение по точечным путям: `{"agents.count": 20, "world.size": [2000, 2000]}`.
Конфиг превращается в словарь (глубокая копия), значения подменяются, затем **заново
проходит валидацию**, поэтому недопустимое значение не проскочит. Несуществующий раздел
даёт `KeyError`.

### `parse_override(text) -> (key, value)`
`"agents.count=20"` → `("agents.count", 20)`. Значение разбирается как YAML, поэтому
типы получаются естественно: `20` — число, `true` — логическое, `[1,2]` — список,
`cbba` — строка.
