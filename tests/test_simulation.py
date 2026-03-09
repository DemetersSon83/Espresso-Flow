import numpy as np

from espresso_darcy_model.config import SimulationConfig
from espresso_darcy_model.simulation import run_simulation


def test_simulation_runs_and_is_monotonic():
    cfg = SimulationConfig()
    cfg.geometry.nr = 10
    cfg.geometry.nz = 8
    cfg.grind.samples_per_cell = 40
    cfg.flow.total_time_s = 5.0
    cfg.flow.dt_s = 0.5
    result = run_simulation(cfg, output_dir=None, save_outputs=False, seed=5)
    assert result.beverage_mass_g.shape == result.time_s.shape
    assert np.all(np.diff(result.beverage_mass_g) >= -1e-9)
    assert np.all(np.diff(result.solids_g) >= -1e-9)
    assert result.summary["final_beverage_mass_g"] >= 0.0
    assert result.summary["final_extraction_yield_percent"] >= 0.0
