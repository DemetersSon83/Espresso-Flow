from __future__ import annotations

import numpy as np


def gaussian_random_field(
    shape: tuple[int, int],
    sigma_r_cells: float,
    sigma_z_cells: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a smooth Gaussian random field with mean ≈ 0 and std ≈ 1."""
    nr, nz = shape
    noise = rng.standard_normal((nr, nz))
    fr = np.fft.fftfreq(nr)
    fz = np.fft.fftfreq(nz)
    kr, kz = np.meshgrid(fr, fz, indexing="ij")
    filt = np.exp(
        -0.5
        * ((2.0 * np.pi * sigma_r_cells * kr) ** 2 + (2.0 * np.pi * sigma_z_cells * kz) ** 2)
    )
    field = np.fft.ifft2(np.fft.fft2(noise) * filt).real
    field -= field.mean()
    std = field.std()
    if std < 1e-12:
        return np.zeros_like(field)
    return field / std
