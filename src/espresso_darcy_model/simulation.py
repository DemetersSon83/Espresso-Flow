from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .config import SimulationConfig
from .medium import MediumState, build_medium
from .plotting import (
    plot_brew_timeseries,
    plot_extraction_map,
    plot_monte_carlo,
    plot_permeability_map,
    plot_pressure_flow_map,
    plot_psd,
)
from .solver import PressureSolution, outlet_flow_rate_m3_s, solve_normalized_pressure, velocity_field


@dataclass
class SimulationResult:
    config: SimulationConfig
    medium: MediumState
    pressure_solution: PressureSolution
    time_s: np.ndarray
    pressure_bar: np.ndarray
    flow_ml_s: np.ndarray
    beverage_mass_g: np.ndarray
    solids_g: np.ndarray
    instant_tds_percent: np.ndarray
    cumulative_tds_percent: np.ndarray
    extraction_yield_percent: np.ndarray
    mean_wetness: np.ndarray
    final_u_r: np.ndarray
    final_u_z: np.ndarray
    final_speed: np.ndarray
    local_extraction_percent: np.ndarray
    summary: dict[str, Any]


def pressure_schedule_bar(config: SimulationConfig, t_s: float) -> float:
    return config.flow.preinfusion_bar if t_s < config.flow.preinfusion_time_s else config.flow.brew_bar


def _safe_percent(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return 100.0 * numerator / denominator


def run_simulation(
    config: SimulationConfig,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    save_outputs: bool = True,
) -> SimulationResult:
    medium = build_medium(config, seed=seed)
    pressure = solve_normalized_pressure(medium)

    times = np.arange(0.0, config.flow.total_time_s + 0.5 * config.flow.dt_s, config.flow.dt_s)

    flow_ml_s = np.zeros_like(times)
    beverage_mass_g = np.zeros_like(times)
    solids_g = np.zeros_like(times)
    instant_tds_percent = np.zeros_like(times)
    cumulative_tds_percent = np.zeros_like(times)
    extraction_yield_percent = np.zeros_like(times)
    mean_wetness = np.zeros_like(times)
    pressure_bar_arr = np.zeros_like(times)

    fast_remaining = medium.soluble_fast_g.copy()
    slow_remaining = medium.soluble_slow_g.copy()
    wetness = np.zeros_like(medium.permeability_m2)

    initial_soluble = medium.soluble_fast_g + medium.soluble_slow_g
    final_u_r = np.zeros_like(medium.permeability_m2)
    final_u_z = np.zeros_like(medium.permeability_m2)
    final_speed = np.zeros_like(medium.permeability_m2)

    cumulative_beverage_g = 0.0
    cumulative_solids_g = 0.0

    temp_factor = np.exp(
        config.extraction.temperature_sensitivity_per_c
        * (config.extraction.temperature_c - config.extraction.temperature_reference_c)
    )

    for idx, t in enumerate(times):
        pressure_bar = pressure_schedule_bar(config, float(t))
        delta_p_pa = pressure_bar * 1e5
        pressure_bar_arr[idx] = pressure_bar

        u_r, u_z, speed = velocity_field(
            medium,
            pressure,
            delta_p_pa=delta_p_pa,
            viscosity_pa_s=config.flow.viscosity_pa_s,
            density_kg_m3=config.flow.density_kg_m3,
            use_forchheimer=config.flow.use_forchheimer,
            forchheimer_beta=config.flow.forchheimer_beta,
        )
        q_out_m3_s = outlet_flow_rate_m3_s(medium, u_z)
        beverage_increment_g = config.flow.density_kg_m3 * q_out_m3_s * config.flow.dt_s * 1000.0

        wetness = np.clip(
            wetness + config.flow.dt_s * speed / max(config.extraction.wetting_velocity_reference_m_s, 1e-8),
            0.0,
            1.0,
        )

        u_ratio = np.maximum(speed / max(config.extraction.velocity_reference_m_s, 1e-9), 1e-6)
        k_fast = config.extraction.k_fast_s * temp_factor * wetness * (u_ratio ** config.extraction.fast_velocity_exponent)
        k_slow = config.extraction.k_slow_s * temp_factor * wetness * (u_ratio ** config.extraction.slow_velocity_exponent)

        release_fast = fast_remaining * (1.0 - np.exp(-k_fast * config.flow.dt_s))
        release_slow = slow_remaining * (1.0 - np.exp(-k_slow * config.flow.dt_s))

        fast_remaining -= release_fast
        slow_remaining -= release_slow

        solids_increment_g = float(np.sum(release_fast + release_slow))
        cumulative_beverage_g += beverage_increment_g
        cumulative_solids_g += solids_increment_g

        flow_ml_s[idx] = q_out_m3_s * 1e6
        beverage_mass_g[idx] = cumulative_beverage_g
        solids_g[idx] = cumulative_solids_g
        instant_tds_percent[idx] = _safe_percent(solids_increment_g, beverage_increment_g)
        cumulative_tds_percent[idx] = _safe_percent(cumulative_solids_g, cumulative_beverage_g)
        extraction_yield_percent[idx] = _safe_percent(cumulative_solids_g, config.extraction.dose_g)
        mean_wetness[idx] = float(np.mean(wetness))

        final_u_r = u_r
        final_u_z = u_z
        final_speed = speed

    local_extraction_percent = 100.0 * (initial_soluble - (fast_remaining + slow_remaining)) / np.maximum(
        medium.dry_mass_g, 1e-12
    )

    target_mass = 2.0 * config.extraction.dose_g
    target_index = np.argmax(beverage_mass_g >= target_mass)
    time_to_target = float(times[target_index]) if beverage_mass_g[target_index] >= target_mass else float("nan")

    summary = {
        "model_name": config.model_name,
        "random_seed": int(config.random_seed if seed is None else seed),
        "dose_g": float(config.extraction.dose_g),
        "target_beverage_mass_g": float(target_mass),
        "time_to_target_mass_s": time_to_target,
        "final_beverage_mass_g": float(beverage_mass_g[-1]),
        "final_solids_g": float(solids_g[-1]),
        "final_extraction_yield_percent": float(extraction_yield_percent[-1]),
        "final_cumulative_tds_percent": float(cumulative_tds_percent[-1]),
        "mean_flow_ml_s": float(np.mean(flow_ml_s)),
        "median_permeability_m2": float(np.median(medium.permeability_m2)),
        "mean_porosity": float(np.mean(medium.porosity)),
        "mean_particle_um": float(np.mean(medium.mean_particle_um)),
        "mean_fines_fraction": float(np.mean(medium.fines_fraction)),
    }

    result = SimulationResult(
        config=config,
        medium=medium,
        pressure_solution=pressure,
        time_s=times,
        pressure_bar=pressure_bar_arr,
        flow_ml_s=flow_ml_s,
        beverage_mass_g=beverage_mass_g,
        solids_g=solids_g,
        instant_tds_percent=instant_tds_percent,
        cumulative_tds_percent=cumulative_tds_percent,
        extraction_yield_percent=extraction_yield_percent,
        mean_wetness=mean_wetness,
        final_u_r=final_u_r,
        final_u_z=final_u_z,
        final_speed=final_speed,
        local_extraction_percent=local_extraction_percent,
        summary=summary,
    )

    if save_outputs and output_dir is not None:
        save_simulation_outputs(result, Path(output_dir))

    return result


def save_simulation_outputs(result: SimulationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    if result.config.output.save_timeseries_csv:
        csv_path = output_dir / "timeseries.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time_s",
                    "pressure_bar",
                    "flow_ml_s",
                    "beverage_mass_g",
                    "solids_g",
                    "instant_tds_percent",
                    "cumulative_tds_percent",
                    "extraction_yield_percent",
                    "mean_wetness",
                ]
            )
            for row in zip(
                result.time_s,
                result.pressure_bar,
                result.flow_ml_s,
                result.beverage_mass_g,
                result.solids_g,
                result.instant_tds_percent,
                result.cumulative_tds_percent,
                result.extraction_yield_percent,
                result.mean_wetness,
            ):
                writer.writerow([float(v) for v in row])

    if result.config.output.save_summary_json:
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(result.summary, f, indent=2)

    if result.config.output.save_fields_npz:
        np.savez_compressed(
            output_dir / "fields.npz",
            radius_m=result.medium.radius_m,
            depth_m=result.medium.depth_m,
            permeability_m2=result.medium.permeability_m2,
            porosity=result.medium.porosity,
            fines_fraction=result.medium.fines_fraction,
            coarse_median_um=result.medium.coarse_median_um,
            mean_particle_um=result.medium.mean_particle_um,
            pressure_norm=result.pressure_solution.p_norm,
            final_u_r=result.final_u_r,
            final_u_z=result.final_u_z,
            local_extraction_percent=result.local_extraction_percent,
        )

    if result.config.output.save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_psd(result.medium, plots_dir)
        plot_permeability_map(result.medium, plots_dir)
        plot_pressure_flow_map(
            result.medium,
            result.pressure_solution,
            result.final_u_r,
            result.final_u_z,
            result.config.flow.brew_bar,
            plots_dir,
        )
        plot_brew_timeseries(result, plots_dir)
        plot_extraction_map(result.medium, result.local_extraction_percent, plots_dir)


def _aggregate_percentiles(stack: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(stack, axis=0).tolist(),
        "p10": np.percentile(stack, 10.0, axis=0).tolist(),
        "p90": np.percentile(stack, 90.0, axis=0).tolist(),
    }


def run_monte_carlo(
    config: SimulationConfig,
    runs: int,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    seeds = [config.random_seed + i for i in range(runs)]
    results = [run_simulation(config, output_dir=None, seed=seed, save_outputs=False) for seed in seeds]

    flow_stack = np.vstack([r.flow_ml_s for r in results])
    mass_stack = np.vstack([r.beverage_mass_g for r in results])
    yield_stack = np.vstack([r.extraction_yield_percent for r in results])

    summary_table = {
        "seed": seeds,
        "time_to_target_mass_s": [r.summary["time_to_target_mass_s"] for r in results],
        "final_beverage_mass_g": [r.summary["final_beverage_mass_g"] for r in results],
        "final_extraction_yield_percent": [r.summary["final_extraction_yield_percent"] for r in results],
        "final_cumulative_tds_percent": [r.summary["final_cumulative_tds_percent"] for r in results],
        "median_permeability_m2": [r.summary["median_permeability_m2"] for r in results],
    }

    mc_result = {
        "time_s": results[0].time_s.tolist(),
        "flow_ml_s_mean": np.mean(flow_stack, axis=0).tolist(),
        "flow_ml_s_p10": np.percentile(flow_stack, 10.0, axis=0).tolist(),
        "flow_ml_s_p90": np.percentile(flow_stack, 90.0, axis=0).tolist(),
        "beverage_mass_g": _aggregate_percentiles(mass_stack),
        "extraction_yield_percent": _aggregate_percentiles(yield_stack),
        "summary_table": summary_table,
        "overall_summary": {
            "runs": runs,
            "mean_time_to_target_mass_s": float(np.nanmean(summary_table["time_to_target_mass_s"])),
            "std_time_to_target_mass_s": float(np.nanstd(summary_table["time_to_target_mass_s"])),
            "mean_final_extraction_yield_percent": float(np.mean(summary_table["final_extraction_yield_percent"])),
            "std_final_extraction_yield_percent": float(np.std(summary_table["final_extraction_yield_percent"])),
            "mean_final_cumulative_tds_percent": float(np.mean(summary_table["final_cumulative_tds_percent"])),
            "target_mass_g": float(results[0].summary["target_beverage_mass_g"]),
        },
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "mc_summary.json", "w", encoding="utf-8") as f:
            json.dump(mc_result, f, indent=2)
        with open(out / "mc_runs.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(summary_table.keys())
            for row in zip(*summary_table.values()):
                writer.writerow(row)
        plot_monte_carlo(mc_result, out / "plots")

    return mc_result
