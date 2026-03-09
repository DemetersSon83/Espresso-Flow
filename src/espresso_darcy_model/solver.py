from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .medium import MediumState


@dataclass
class PressureSolution:
    p_norm: np.ndarray
    dpdr_norm: np.ndarray
    dpdz_norm: np.ndarray


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def solve_normalized_pressure(medium: MediumState) -> PressureSolution:
    """
    Solve ∇·(K ∇p) = 0 on an axisymmetric (r, z) grid with
    p(top)=1, p(bottom)=0, and no-flux side boundaries.
    """
    K = medium.permeability_m2
    nr, nz = K.shape
    n = nr * nz
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    def idx(i: int, j: int) -> int:
        return i * nz + j

    dr = medium.dr
    dz = medium.dz
    r = medium.radius_m

    for i in range(nr):
        for j in range(nz):
            k = idx(i, j)
            if j == 0:
                A[k, k] = 1.0
                b[k] = 1.0
                continue
            if j == nz - 1:
                A[k, k] = 1.0
                b[k] = 0.0
                continue

            diag = 0.0
            ri = r[i]

            if i > 0:
                rw = max(ri - 0.5 * dr, 0.0)
                kw = _harmonic_mean(K[i, j], K[i - 1, j])
                coeff_w = rw * kw / (ri * dr * dr)
                A[k, idx(i - 1, j)] = coeff_w
                diag -= coeff_w
            if i < nr - 1:
                re = ri + 0.5 * dr
                ke = _harmonic_mean(K[i, j], K[i + 1, j])
                coeff_e = re * ke / (ri * dr * dr)
                A[k, idx(i + 1, j)] = coeff_e
                diag -= coeff_e

            ks = _harmonic_mean(K[i, j], K[i, j - 1])
            kn = _harmonic_mean(K[i, j], K[i, j + 1])
            coeff_s = ks / (dz * dz)
            coeff_n = kn / (dz * dz)

            A[k, idx(i, j - 1)] = coeff_s
            A[k, idx(i, j + 1)] = coeff_n
            diag -= (coeff_s + coeff_n)

            A[k, k] = diag

    p_flat = np.linalg.solve(A, b)
    p = p_flat.reshape((nr, nz))

    dpdr = np.zeros_like(p)
    dpdz = np.zeros_like(p)

    dpdr[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dr)
    dpdr[0, :] = (p[1, :] - p[0, :]) / dr
    dpdr[-1, :] = (p[-1, :] - p[-2, :]) / dr

    dpdz[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dz)
    dpdz[:, 0] = (p[:, 1] - p[:, 0]) / dz
    dpdz[:, -1] = (p[:, -1] - p[:, -2]) / dz

    return PressureSolution(p_norm=p, dpdr_norm=dpdr, dpdz_norm=dpdz)


def velocity_field(
    medium: MediumState,
    pressure: PressureSolution,
    delta_p_pa: float,
    viscosity_pa_s: float,
    density_kg_m3: float,
    use_forchheimer: bool,
    forchheimer_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (u_r, u_z, |u|) in m/s. The pressure solve is Darcy-based; an
    optional Forchheimer correction rescales the velocity magnitude locally.
    """
    K = medium.permeability_m2
    g_r = pressure.dpdr_norm * delta_p_pa
    g_z = pressure.dpdz_norm * delta_p_pa

    u_r_d = -(K / viscosity_pa_s) * g_r
    u_z_d = -(K / viscosity_pa_s) * g_z
    speed_d = np.sqrt(u_r_d**2 + u_z_d**2)

    if (not use_forchheimer) or forchheimer_beta <= 0.0:
        return u_r_d, u_z_d, speed_d

    grad_mag = np.sqrt(g_r**2 + g_z**2)
    a = forchheimer_beta * density_kg_m3 / np.sqrt(K)
    b = viscosity_pa_s / K
    disc = np.maximum(b**2 + 4.0 * a * grad_mag, 0.0)
    speed_f = (-b + np.sqrt(disc)) / (2.0 * a)
    speed = np.where(np.isfinite(speed_f), speed_f, speed_d)

    direction_r = np.divide(u_r_d, speed_d, out=np.zeros_like(u_r_d), where=speed_d > 1e-15)
    direction_z = np.divide(u_z_d, speed_d, out=np.zeros_like(u_z_d), where=speed_d > 1e-15)

    u_r = direction_r * speed
    u_z = direction_z * speed
    return u_r, u_z, speed


def outlet_flow_rate_m3_s(medium: MediumState, u_z: np.ndarray) -> float:
    """
    Approximate the outlet flow using the bottom-row axial velocity and
    axisymmetric annular weights.
    """
    r = medium.radius_m
    dr = medium.dr
    bottom_u = np.maximum(u_z[:, -1], 0.0)
    r_inner = np.maximum(r - 0.5 * dr, 0.0)
    r_outer = r + 0.5 * dr
    annular_area = np.pi * (r_outer**2 - r_inner**2)
    return float(np.sum(bottom_u * annular_area))
