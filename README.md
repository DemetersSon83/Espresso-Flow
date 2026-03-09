# Espresso Flow

A reduced-order Python repository for simulating espresso extraction as flow through a compressed porous puck. The repository couples:

- stochastic **bimodal grind-size sampling** for each cell in the puck,
- **2D heterogeneous** radial-axial permeability and porosity fields,
- a Darcy pressure solve with an optional **Forchheimer-style inertial correction**,
- a simple **two-pool extraction model** with wetting/preinfusion effects,
- black-and-white plotting for publication-friendly figures.

This is not a CFD-grade or experimentally validated digital twin. It is a compact, modifiable research codebase intended for rapid hypothesis testing and visualization.

## What the model includes

- **User-controlled grind settings**
  - coarse median particle size
  - fines median particle size
  - fines mass/number fraction
  - lognormal spreads for each mode
- **Compression / packing controls**
  - uncompressed porosity
  - compression level
  - heterogeneous porosity fluctuations
- **2D heterogeneity**
  - correlated radial-axial random fields in grind size, fines fraction, and porosity
- **Flow model**
  - Darcy pressure field solve in an axisymmetric radial-axial slice
  - optional local Forchheimer correction for inertial loss at higher gradients
- **Extraction model**
  - wetting state that ramps during preinfusion
  - fast and slow soluble pools
  - flow-dependent release kinetics
- **Outputs**
  - timeseries CSV
  - summary JSON
  - compressed field file (`fields.npz`)
  - black-and-white PNG plots

## Repository layout

```text
espresso-flow/
├── CITATION.cff
├── LICENSE
├── README.md
├── pyproject.toml
├── docs/
│   ├── demo_baseline/
│   └── demo_montecarlo/
├── examples/
│   ├── baseline_config.json
│   └── fast_flow_config.json
├── src/
│   └── espresso_darcy_model/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── medium.py
│       ├── plotting.py
│       ├── random_fields.py
│       ├── simulation.py
│       └── solver.py
└── tests/
    ├── test_medium.py
    └── test_simulation.py
```

## Quick start

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Write a default configuration:

```bash
espresso-darcy template --output examples/default_config.json
```

Run a single simulation:

```bash
espresso-darcy run   --config examples/baseline_config.json   --output outputs/baseline
```

Run a Monte Carlo ensemble:

```bash
espresso-darcy montecarlo   --config examples/baseline_config.json   --runs 16   --output outputs/montecarlo
```

Override key parameters from the command line:

```bash
espresso-darcy run   --config examples/baseline_config.json   --output outputs/finer_grind   --coarse-median-um 280   --fines-fraction 0.24   --compression-level 0.62   --brew-bar 8.5
```

## Model sketch

### 1. Stochastic grind distribution

Each cell samples particles from a **bimodal lognormal mixture**:

- coarse mode with median `coarse_median_um`
- fines mode with median `fines_median_um`
- mixture weight `fines_fraction`

An effective Sauter mean diameter is computed from the sampled particles and used in a Kozeny–Carman-style permeability relation.

### 2. Packing and permeability

The local permeability is built from:

- sampled effective particle size,
- compressed porosity,
- a fines penalty term,
- a user-facing empirical calibration scale.

The permeability field is therefore emergent rather than directly prescribed.

### 3. 2D heterogeneous flow

The puck is represented as an **axisymmetric radial-axial slice**. Smooth Gaussian random fields perturb:

- local coarse median size,
- local fines fraction,
- local porosity.

A normalized pressure field is solved from

```text
∇·(K ∇p) = 0
```

with fixed pressure on the top and bottom boundaries and no-flux on the side wall.

### 4. Optional inertial correction

The main pressure solve is Darcy-based. If enabled, the local velocity magnitude is corrected with a quadratic Darcy–Forchheimer relation during post-processing and extraction-rate evaluation.

### 5. Extraction kinetics

Each cell contains a fast and a slow soluble pool. Local release rates depend on:

- wetting state,
- local flow magnitude,
- brew temperature.

This creates channel-sensitive extraction maps without requiring a full advection-diffusion PDE.

## Generated plots

A typical single run produces:

- `psd.png` — bimodal particle-size histograms
- `permeability_map.png` — grayscale permeability map
- `pressure_flow_map.png` — pressure contours with streamlines
- `brew_timeseries.png` — pressure, flow, beverage mass, TDS, yield, wetness
- `extraction_map.png` — final local extraction pattern

All plots use a black/white or grayscale palette.

Sample output folders generated from the included example configurations are provided in `docs/demo_baseline/` and `docs/demo_montecarlo/`.

## Configuration notes

Important configuration groups:

- `geometry`: basket radius, puck depth, grid resolution
- `grind`: average size, fines fraction, distribution spreads, samples per cell
- `packing`: porosity model, heterogeneity scales, permeability calibration
- `flow`: viscosity, pressure schedule, Forchheimer switch
- `extraction`: dose, soluble fraction, kinetic coefficients, temperature

The default configuration is intentionally moderate rather than “correct” for every machine or coffee. You will usually want to calibrate:

- `packing.permeability_scale`
- `grind.coarse_median_um`
- `grind.fines_fraction`
- `packing.compression_level`
- `extraction.k_fast_s`
- `extraction.k_slow_s`

against your own shot time and target beverage mass.

## Example workflow

1. Start from `examples/baseline_config.json`.
2. Adjust average grind size and fines fraction until the flow trajectory is plausible.
3. Tune `packing.permeability_scale` if the shot is globally too fast or too slow.
4. Tune `k_fast_s` and `k_slow_s` to reach a desired extraction yield or TDS level.
5. Run `montecarlo` to inspect shot-to-shot variability from stochastic heterogeneity.

## Testing

Install the dev extras and run:

```bash
pip install -e .[dev]
pytest
```

## Limitations

- The extraction chemistry is reduced-order.
- The Forchheimer correction is local rather than part of a fully coupled nonlinear pressure solve.
- The wetting model is heuristic.
- Basket hardware, shower-screen details, puck fracture, fines migration, and temperature gradients are not explicitly resolved.
- Parameters are best treated as calibration knobs unless you have your own experimental data.

## Citation

Please cite the software using the metadata in `CITATION.cff`.

## Suggested scientific background

This repository was designed to be compatible with common ideas from the espresso / porous-media literature:

- Darcy-style flow through compressed coffee beds
- Kozeny–Carman-inspired permeability scaling
- bimodal grind distributions with a fines population
- spatial heterogeneity and channel-sensitive extraction
- preinfusion / wetting effects
- optional inertial corrections at high gradients

## Future work

- explicit fines migration,
- a coupled nonlinear Forchheimer pressure solve,
- unsaturated infiltration before full saturation,
- temperature transport,
- measured grind histograms instead of synthetic mixtures,
- calibration against shot-weight and refractometer data.
