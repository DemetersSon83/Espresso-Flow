from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .medium import MediumState
from .solver import PressureSolution


def apply_bw_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "0.7",
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "image.cmap": "Greys",
        "font.size": 10,
    })


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_psd(medium: MediumState, output_dir: Path) -> Path:
    apply_bw_style()
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bins = np.geomspace(20, max(float(np.percentile(medium.psd_sample_um, 99.5)), 500), 48)
    ax.hist(
        medium.psd_sample_um,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color="black",
        label="Combined sample",
    )
    ax.hist(
        medium.psd_coarse_sample_um,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        linestyle="--",
        color="black",
        label="Coarse mode",
    )
    ax.hist(
        medium.psd_fines_sample_um,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        linestyle=":",
        color="black",
        label="Fines mode",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Particle diameter [µm]")
    ax.set_ylabel("Density [1/µm]")
    ax.set_title("Bimodal particle-size distribution")
    ax.legend(frameon=False)
    out = output_dir / "psd.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_permeability_map(medium: MediumState, output_dir: Path) -> Path:
    apply_bw_style()
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    field = np.log10(medium.permeability_m2).T
    extent = [0.0, medium.radius_m[-1] * 1e3 + 0.5 * medium.dr * 1e3, medium.depth_m[-1] * 1e3 + 0.5 * medium.dz * 1e3, 0.0]
    im = ax.imshow(field, extent=extent, aspect="auto", cmap="Greys")
    cs = ax.contour(
        medium.radius_m * 1e3,
        medium.depth_m * 1e3,
        field,
        colors="black",
        linewidths=0.5,
        levels=6,
    )
    ax.clabel(cs, fmt="%.1f", fontsize=8)
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Depth [mm]")
    ax.set_title("log10 permeability map [m²]")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("log10(K)")
    out = output_dir / "permeability_map.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_pressure_flow_map(
    medium: MediumState,
    pressure: PressureSolution,
    u_r: np.ndarray,
    u_z: np.ndarray,
    brew_bar: float,
    output_dir: Path,
) -> Path:
    apply_bw_style()
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    p_bar = pressure.p_norm.T * brew_bar
    im = ax.contourf(
        medium.radius_m * 1e3,
        medium.depth_m * 1e3,
        p_bar,
        levels=14,
        cmap="Greys",
    )
    x = medium.radius_m * 1e3
    y = medium.depth_m * 1e3
    ax.streamplot(
        x,
        y,
        u_r.T * 1e3,
        u_z.T * 1e3,
        color="black",
        density=1.0,
        linewidth=0.6,
        arrowsize=0.7,
    )
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Depth [mm]")
    ax.invert_yaxis()
    ax.set_title("Pressure contours and flow field at brew pressure")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Pressure [bar]")
    out = output_dir / "pressure_flow_map.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_brew_timeseries(result, output_dir: Path) -> Path:
    apply_bw_style()
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 9.0), sharex=True)

    axes[0].plot(result.time_s, result.pressure_bar, color="black", linewidth=1.6)
    axes[0].set_ylabel("Pressure [bar]")
    axes[0].set_title("Brew trajectory")

    axes[1].plot(result.time_s, result.flow_ml_s, color="black", linewidth=1.5, label="Flow")
    axes[1].plot(result.time_s, result.beverage_mass_g, color="black", linewidth=1.2, linestyle="--", label="Beverage mass")
    axes[1].set_ylabel("Flow [mL/s] / mass [g]")
    axes[1].legend(frameon=False, ncol=2)

    axes[2].plot(result.time_s, result.instant_tds_percent, color="black", linewidth=1.4, label="Instantaneous TDS")
    axes[2].plot(result.time_s, result.cumulative_tds_percent, color="black", linewidth=1.2, linestyle="--", label="Cumulative TDS")
    axes[2].set_ylabel("TDS [%]")
    axes[2].legend(frameon=False)

    axes[3].plot(result.time_s, result.extraction_yield_percent, color="black", linewidth=1.4, label="Extraction yield")
    axes[3].plot(result.time_s, 100.0 * result.mean_wetness, color="black", linewidth=1.2, linestyle=":", label="Mean wetness")
    axes[3].set_ylabel("Yield / wetness [%]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(frameon=False, ncol=2)

    out = output_dir / "brew_timeseries.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_extraction_map(medium: MediumState, local_extraction_percent: np.ndarray, output_dir: Path) -> Path:
    apply_bw_style()
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    extent = [0.0, medium.radius_m[-1] * 1e3 + 0.5 * medium.dr * 1e3, medium.depth_m[-1] * 1e3 + 0.5 * medium.dz * 1e3, 0.0]
    lo = float(np.nanpercentile(local_extraction_percent, 5.0))
    hi = float(np.nanpercentile(local_extraction_percent, 95.0))
    if hi - lo < 1.0:
        lo = float(np.nanmin(local_extraction_percent))
        hi = float(np.nanmax(local_extraction_percent))
    if hi - lo < 1.0:
        hi = lo + 1.0
    im = ax.imshow(
        local_extraction_percent.T,
        extent=extent,
        aspect="auto",
        cmap="Greys",
        vmin=lo,
        vmax=hi,
    )
    cs = ax.contour(
        medium.radius_m * 1e3,
        medium.depth_m * 1e3,
        local_extraction_percent.T,
        colors="black",
        linewidths=0.5,
        levels=np.linspace(lo, hi, 7),
    )
    ax.clabel(cs, fmt="%.1f", fontsize=8)
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Depth [mm]")
    ax.set_title("Local extraction yield map [% dry mass]")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Extraction yield [%]")
    out = output_dir / "extraction_map.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_monte_carlo(mc_result: dict, output_dir: Path) -> tuple[Path, Path]:
    apply_bw_style()
    _ensure_dir(output_dir)

    t = mc_result["time_s"]
    mass = mc_result["beverage_mass_g"]
    yield_pct = mc_result["extraction_yield_percent"]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.4), sharex=True)
    axes[0].plot(t, mc_result["flow_ml_s_mean"], color="black", linewidth=1.4)
    axes[0].fill_between(t, mc_result["flow_ml_s_p10"], mc_result["flow_ml_s_p90"], color="0.85")
    axes[0].set_ylabel("Flow [mL/s]")
    axes[0].set_title("Monte Carlo envelope")

    axes[1].plot(t, mass["mean"], color="black", linewidth=1.5, label="Beverage mass")
    axes[1].fill_between(t, mass["p10"], mass["p90"], color="0.85")
    axes[1].plot(t, yield_pct["mean"], color="black", linewidth=1.2, linestyle="--", label="Extraction yield")
    axes[1].fill_between(t, yield_pct["p10"], yield_pct["p90"], color="0.92")
    axes[1].set_ylabel("Mass [g] / yield [%]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(frameon=False, ncol=2)

    envelope_out = output_dir / "mc_envelope.png"
    fig.tight_layout()
    fig.savefig(envelope_out, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    finite_times = np.asarray(mc_result["summary_table"]["time_to_target_mass_s"], dtype=float)
    finite_times = finite_times[np.isfinite(finite_times)]
    if finite_times.size == 0:
        finite_times = np.array([0.0, 1.0])
    bins = np.linspace(float(finite_times.min()) - 1.0, float(finite_times.max()) + 1.0, 10)
    ax.hist(finite_times, bins=bins, histtype="step", color="black", linewidth=1.5, label="time to target mass")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Count")
    ax.set_title("Monte Carlo shot-time spread")
    ax.legend(frameon=False)
    hist_out = output_dir / "mc_histograms.png"
    fig.tight_layout()
    fig.savefig(hist_out, dpi=180)
    plt.close(fig)

    return envelope_out, hist_out
