from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig
from .random_fields import gaussian_random_field


UM_TO_M = 1e-6
MM_TO_M = 1e-3


@dataclass
class MediumState:
    radius_m: np.ndarray
    depth_m: np.ndarray
    r2d: np.ndarray
    z2d: np.ndarray
    dr: float
    dz: float
    coarse_median_um: np.ndarray
    fines_fraction: np.ndarray
    porosity: np.ndarray
    d32_m: np.ndarray
    permeability_m2: np.ndarray
    mean_particle_um: np.ndarray
    dry_mass_g: np.ndarray
    soluble_fast_g: np.ndarray
    soluble_slow_g: np.ndarray
    psd_sample_um: np.ndarray
    psd_coarse_sample_um: np.ndarray
    psd_fines_sample_um: np.ndarray


def sample_bimodal_diameters_um(
    n: int,
    coarse_median_um: float,
    coarse_sigma_ln: float,
    fines_median_um: float,
    fines_sigma_ln: float,
    fines_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_fines = int(round(n * fines_fraction))
    n_fines = min(max(n_fines, 1), n - 1)
    n_coarse = n - n_fines
    coarse = np.exp(np.log(coarse_median_um) + coarse_sigma_ln * rng.standard_normal(n_coarse))
    fines = np.exp(np.log(fines_median_um) + fines_sigma_ln * rng.standard_normal(n_fines))
    mixture = np.concatenate([coarse, fines])
    rng.shuffle(mixture)
    return mixture, coarse, fines


def sauter_mean_diameter_m(diameters_um: np.ndarray) -> float:
    d = diameters_um * UM_TO_M
    numerator = np.sum(d**3)
    denominator = np.sum(d**2)
    if denominator <= 0:
        raise ValueError("Diameter sample produced zero second moment.")
    return float(numerator / denominator)


def _base_porosity(config: SimulationConfig) -> float:
    phi0 = config.packing.porosity_uncompressed
    compressed = phi0 - 0.10 * config.packing.compression_level
    return float(np.clip(compressed, 0.26, 0.50))


def _cell_volumes_m3(radius_m: np.ndarray, dr: float, dz: float) -> np.ndarray:
    r_inner = np.maximum(radius_m - 0.5 * dr, 0.0)
    r_outer = radius_m + 0.5 * dr
    annular_area = np.pi * (r_outer**2 - r_inner**2)
    return annular_area[:, None] * dz


def build_medium(config: SimulationConfig, seed: int | None = None) -> MediumState:
    rng = np.random.default_rng(config.random_seed if seed is None else seed)

    nr = config.geometry.nr
    nz = config.geometry.nz
    radius_total_m = config.geometry.basket_radius_mm * MM_TO_M
    depth_total_m = config.geometry.puck_depth_mm * MM_TO_M

    dr = radius_total_m / nr
    dz = depth_total_m / nz
    radius_m = (np.arange(nr) + 0.5) * dr
    depth_m = (np.arange(nz) + 0.5) * dz
    r2d, z2d = np.meshgrid(radius_m, depth_m, indexing="ij")

    sigma_r_cells = max(
        config.packing.correlation_length_r_mm / config.geometry.basket_radius_mm * nr, 0.5
    )
    sigma_z_cells = max(
        config.packing.correlation_length_z_mm / config.geometry.puck_depth_mm * nz, 0.5
    )

    grind_field = gaussian_random_field((nr, nz), sigma_r_cells, sigma_z_cells, rng)
    fines_field = gaussian_random_field((nr, nz), sigma_r_cells * 0.8, sigma_z_cells * 0.8, rng)
    porosity_field = gaussian_random_field((nr, nz), sigma_r_cells * 1.2, sigma_z_cells, rng)

    coarse_median_um = config.grind.coarse_median_um * np.exp(
        config.packing.grind_heterogeneity_std * grind_field
    )
    local_fines_fraction = np.clip(
        config.grind.fines_fraction + config.packing.fines_heterogeneity_std * fines_field,
        0.02,
        0.55,
    )
    porosity = np.clip(
        _base_porosity(config) + config.packing.porosity_heterogeneity_std * porosity_field,
        0.24,
        0.50,
    )

    d32_m = np.zeros((nr, nz), dtype=float)
    mean_particle_um = np.zeros((nr, nz), dtype=float)

    psd_sample_um = []
    psd_coarse_sample_um = []
    psd_fines_sample_um = []
    sample_stride = max((nr * nz) // 50, 1)

    for i in range(nr):
        for j in range(nz):
            mix, coarse, fines = sample_bimodal_diameters_um(
                n=config.grind.samples_per_cell,
                coarse_median_um=float(coarse_median_um[i, j]),
                coarse_sigma_ln=config.grind.coarse_sigma_ln,
                fines_median_um=config.grind.fines_median_um,
                fines_sigma_ln=config.grind.fines_sigma_ln,
                fines_fraction=float(local_fines_fraction[i, j]),
                rng=rng,
            )
            d32_m[i, j] = sauter_mean_diameter_m(mix)
            mean_particle_um[i, j] = float(np.mean(mix))
            if (i * nz + j) % sample_stride == 0:
                psd_sample_um.append(mix)
                psd_coarse_sample_um.append(coarse)
                psd_fines_sample_um.append(fines)

    psd_sample_um = np.concatenate(psd_sample_um)
    psd_coarse_sample_um = np.concatenate(psd_coarse_sample_um)
    psd_fines_sample_um = np.concatenate(psd_fines_sample_um)

    phi = porosity
    kc = config.packing.kozeny_carman_constant
    base_perm = (d32_m**2) * (phi**3) / (kc * (1.0 - phi) ** 2)
    fines_penalty = np.exp(-config.packing.fines_penalty * local_fines_fraction)
    permeability_m2 = config.packing.permeability_scale * base_perm * fines_penalty
    permeability_m2 = np.clip(permeability_m2, 3.0e-16, 2.0e-13)

    volumes_m3 = _cell_volumes_m3(radius_m, dr, dz)
    solid_weights = volumes_m3 * (1.0 - porosity)
    solid_weights /= solid_weights.sum()

    dry_mass_g = config.extraction.dose_g * solid_weights
    soluble_total_g = config.extraction.soluble_fraction * dry_mass_g
    soluble_fast_g = config.extraction.fast_pool_fraction * soluble_total_g
    soluble_slow_g = (1.0 - config.extraction.fast_pool_fraction) * soluble_total_g

    return MediumState(
        radius_m=radius_m,
        depth_m=depth_m,
        r2d=r2d,
        z2d=z2d,
        dr=dr,
        dz=dz,
        coarse_median_um=coarse_median_um,
        fines_fraction=local_fines_fraction,
        porosity=porosity,
        d32_m=d32_m,
        permeability_m2=permeability_m2,
        mean_particle_um=mean_particle_um,
        dry_mass_g=dry_mass_g,
        soluble_fast_g=soluble_fast_g,
        soluble_slow_g=soluble_slow_g,
        psd_sample_um=psd_sample_um,
        psd_coarse_sample_um=psd_coarse_sample_um,
        psd_fines_sample_um=psd_fines_sample_um,
    )
