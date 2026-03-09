from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import SimulationConfig, load_config, save_config
from .simulation import run_monte_carlo, run_simulation


def _apply_overrides(config: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    if args.coarse_median_um is not None:
        config.grind.coarse_median_um = args.coarse_median_um
    if args.fines_fraction is not None:
        config.grind.fines_fraction = args.fines_fraction
    if args.brew_bar is not None:
        config.flow.brew_bar = args.brew_bar
    if args.preinfusion_bar is not None:
        config.flow.preinfusion_bar = args.preinfusion_bar
    if args.temperature_c is not None:
        config.extraction.temperature_c = args.temperature_c
    if args.compression_level is not None:
        config.packing.compression_level = args.compression_level
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="espresso-darcy",
        description="Reduced-order axisymmetric espresso extraction model with 2D heterogeneity.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    template = sub.add_parser("template", help="Write a default JSON configuration.")
    template.add_argument("--output", required=True, help="Path to write the default configuration JSON.")

    run = sub.add_parser("run", help="Run a single simulation.")
    run.add_argument("--config", required=True, help="JSON configuration path.")
    run.add_argument("--output", required=True, help="Directory for outputs.")
    run.add_argument("--coarse-median-um", dest="coarse_median_um", type=float)
    run.add_argument("--fines-fraction", dest="fines_fraction", type=float)
    run.add_argument("--brew-bar", dest="brew_bar", type=float)
    run.add_argument("--preinfusion-bar", dest="preinfusion_bar", type=float)
    run.add_argument("--temperature-c", dest="temperature_c", type=float)
    run.add_argument("--compression-level", dest="compression_level", type=float)
    run.add_argument("--random-seed", dest="random_seed", type=int)

    mc = sub.add_parser("montecarlo", help="Run a Monte Carlo ensemble.")
    mc.add_argument("--config", required=True, help="JSON configuration path.")
    mc.add_argument("--output", required=True, help="Directory for outputs.")
    mc.add_argument("--runs", type=int, default=12)
    mc.add_argument("--coarse-median-um", dest="coarse_median_um", type=float)
    mc.add_argument("--fines-fraction", dest="fines_fraction", type=float)
    mc.add_argument("--brew-bar", dest="brew_bar", type=float)
    mc.add_argument("--preinfusion-bar", dest="preinfusion_bar", type=float)
    mc.add_argument("--temperature-c", dest="temperature_c", type=float)
    mc.add_argument("--compression-level", dest="compression_level", type=float)
    mc.add_argument("--random-seed", dest="random_seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "template":
        config = SimulationConfig()
        save_config(config, args.output)
        print(f"Wrote default config to {args.output}")
        return 0

    config = load_config(args.config)
    config = _apply_overrides(config, args)

    if args.command == "run":
        result = run_simulation(config, output_dir=Path(args.output), save_outputs=True)
        print("Run complete.")
        for key, value in result.summary.items():
            print(f"{key}: {value}")
        return 0

    if args.command == "montecarlo":
        mc_result = run_monte_carlo(config, runs=args.runs, output_dir=Path(args.output))
        print("Monte Carlo run complete.")
        for key, value in mc_result["overall_summary"].items():
            print(f"{key}: {value}")
        return 0

    parser.error("Unknown command.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
