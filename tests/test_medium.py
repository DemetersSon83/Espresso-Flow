import numpy as np

from espresso_darcy_model.config import SimulationConfig
from espresso_darcy_model.medium import build_medium, sample_bimodal_diameters_um


def test_bimodal_sampler_respects_positive_sizes():
    rng = np.random.default_rng(0)
    mix, coarse, fines = sample_bimodal_diameters_um(
        n=500,
        coarse_median_um=300.0,
        coarse_sigma_ln=0.3,
        fines_median_um=60.0,
        fines_sigma_ln=0.2,
        fines_fraction=0.2,
        rng=rng,
    )
    assert len(mix) == 500
    assert np.all(mix > 0)
    assert np.mean(coarse) > np.mean(fines)


def test_medium_builder_outputs_physical_fields():
    cfg = SimulationConfig()
    cfg.geometry.nr = 10
    cfg.geometry.nz = 8
    cfg.grind.samples_per_cell = 40
    medium = build_medium(cfg, seed=11)
    assert medium.permeability_m2.shape == (10, 8)
    assert medium.porosity.min() > 0.2
    assert medium.permeability_m2.min() > 0.0
    assert np.isclose(medium.dry_mass_g.sum(), cfg.extraction.dose_g)
