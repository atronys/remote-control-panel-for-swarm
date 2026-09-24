# `tests/` — автотесты ★

Запуск: `pytest` (31 тест, около 10 с). Фреймворк — pytest: функции `test_*`, фикстуры,
`@pytest.mark.parametrize`, `pytest.approx` для сравнения чисел с плавающей точкой,
`pytest.raises` для ожидаемых ошибок. По `docs/06`, раздел 7, модуль «готов», когда у него
есть хотя бы один тест на ключевой инвариант.

## `conftest.py`
Фикстура `small` — быстрый сценарий: 400×400 м, 4 агента, 10 целей, секторы 200 м.

## `test_kinematics.py` — физика движения
| Тест | Проверяет |
|---|---|
| `test_kinematic_constant_speed_no_overshoot` | шаг ровно `v·dt`; точку не перелетает |
| `test_kinematic_velocity_is_clipped` | `|v| ≤ v_max` |
| `test_double_integrator_respects_limits` | 3000 случайных команд: `|a| ≤ a_max`, `|v| ≤ v_max`, `|Δψ| ≤ ψ̇_max·dt` |
| `test_double_integrator_reaches_waypoint_and_stops` | долетает и останавливается |
| `test_double_integrator_is_slower_than_kinematic` | инерция увеличивает время перехода |

## `test_energy.py` — энергия
| Тест | Проверяет |
|---|---|
| `test_hover_power` | `P(0) = P0 + Pi` |
| `test_power_curve_shape` | U-образность; `v_min_power ≈ 10.2`, `v_max_range ≈ 18.3` м/с |
| `test_energy_per_meter_minimum_at_v_max_range` | минимум Дж/м совпадает с `v_max_range` |
| `test_endurance_matches_nfr11` | ≥ 15 мин полёта (NFR-11) |

## `test_world.py` — мир
| Тест | Проверяет |
|---|---|
| `test_geometry` | точка в многоугольнике, отрезок насквозь |
| `test_nofly_and_bounds` | зоны, границы, цели не в зонах |
| `test_targets_inside_bounds_and_prior_normalized` | сумма карты = 1, цели в пределах |
| `test_world_is_paired_across_agent_count` | **парный дизайн**: тот же мир при 3 и 12 агентах; другой при другом зерне |
| `test_new_targets_appear_on_schedule` | новые цели появляются ровно вовремя |
| `test_swath_from_gsd` | GSD, ширина полосы, слепота выше `GSD_max` |
| `test_observe_accumulates_pd_as_in_formula` | накопление `Pd = 1 − exp(−W·ΣL/A)` |
| `test_separation_counts_entries_not_steps` | столкновения считаются по входу пары; разные эшелоны не сталкиваются |

## `test_sim.py` — ядро
| Тест | Проверяет |
|---|---|
| `test_determinism_same_seed_same_numbers` ×2 модели | **детерминизм**: совпадают summary, события, ряды, траектории |
| `test_different_seed_different_world` | другое зерно — другой прогон |
| `test_baseline_completes_all_tasks_safely` | CR = 1, 0 нарушений, все сели |
| `test_failure_releases_tasks_and_static_baseline_loses_them` | отказ освобождает задачи; статический метод их теряет |
| `test_low_battery_triggers_rtl` | с малой батареей — возврат **до** разряда |
| `test_core_supports_dynamic_reallocation` | тестовый динамический распределитель после отказа выполняет 100 % без дублей — **ядро готово к CBBA** |
| `test_duplicate_assignment_is_rejected` | дубль задачи → `AssignmentError` |
| `test_collisions_invalidate_run` | столкновение → `valid = False` |

## `test_scenario.py` — конфиги
| Тест | Проверяет |
|---|---|
| `test_shipped_scenarios_are_valid` | все `scenarios/*.yaml` загружаются |
| `test_validation_errors` | `v_cruise > v_max`, опечатка в ключе, `dt > t_end` отвергаются |
| `test_overrides_and_hash` | `--set` работает; хэш меняется от изменений и стабилен без них |
