from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict


@dataclass
class GeometryConfig:
    basket_radius_mm: float = 29.0
    puck_depth_mm: float = 9.0
    nr: int = 28
    nz: int = 24


@dataclass
class GrindDistributionConfig:
    coarse_median_um: float = 320.0
    coarse_sigma_ln: float = 0.33
    fines_median_um: float = 60.0
    fines_sigma_ln: float = 0.25
    fines_fraction: float = 0.18
    samples_per_cell: int = 300


@dataclass
class PackingConfig:
    porosity_uncompressed: float = 0.43
    compression_level: float = 0.55
    porosity_heterogeneity_std: float = 0.018
    grind_heterogeneity_std: float = 0.16
    fines_heterogeneity_std: float = 0.045
    correlation_length_r_mm: float = 6.0
    correlation_length_z_mm: float = 2.5
    kozeny_carman_constant: float = 180.0
    permeability_scale: float = 2.5e-5
    fines_penalty: float = 3.2


@dataclass
class FlowConfig:
    viscosity_pa_s: float = 0.00032
    density_kg_m3: float = 970.0
    preinfusion_bar: float = 2.0
    preinfusion_time_s: float = 6.0
    brew_bar: float = 9.0
    total_time_s: float = 32.0
    dt_s: float = 0.2
    use_forchheimer: bool = True
    forchheimer_beta: float = 0.55


@dataclass
class ExtractionConfig:
    dose_g: float = 18.0
    soluble_fraction: float = 0.30
    fast_pool_fraction: float = 0.62
    k_fast_s: float = 0.18
    k_slow_s: float = 0.022
    velocity_reference_m_s: float = 0.0028
    fast_velocity_exponent: float = 0.32
    slow_velocity_exponent: float = 0.18
    temperature_c: float = 93.0
    temperature_reference_c: float = 93.0
    temperature_sensitivity_per_c: float = 0.018
    wetting_velocity_reference_m_s: float = 0.0018


@dataclass
class OutputConfig:
    save_fields_npz: bool = True
    save_timeseries_csv: bool = True
    save_summary_json: bool = True
    save_plots: bool = True


@dataclass
class SimulationConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    grind: GrindDistributionConfig = field(default_factory=GrindDistributionConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    random_seed: int = 7
    model_name: str = "espresso_darcy"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        def merge(dc_cls, values):
            if values is None:
                return dc_cls()
            kwargs = {}
            for field_name in dc_cls.__dataclass_fields__:
                if field_name in values:
                    kwargs[field_name] = values[field_name]
            return dc_cls(**kwargs)

        return cls(
            geometry=merge(GeometryConfig, data.get("geometry")),
            grind=merge(GrindDistributionConfig, data.get("grind")),
            packing=merge(PackingConfig, data.get("packing")),
            flow=merge(FlowConfig, data.get("flow")),
            extraction=merge(ExtractionConfig, data.get("extraction")),
            output=merge(OutputConfig, data.get("output")),
            random_seed=data.get("random_seed", 7),
            model_name=data.get("model_name", "espresso_darcy"),
        )


def load_config(path: str | Path) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SimulationConfig.from_dict(data)


def save_config(config: SimulationConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
