"""Radiometric model: SNR, NEDL, ARA, RXRA striping, noise (§3.1-§3.4, §6.5).
"""

from __future__ import annotations

import numpy as np


def snr(radiance: np.ndarray, lref: np.ndarray, g: np.ndarray, n0: np.ndarray) -> np.ndarray:
    """SNR vs at-sensor radiance L in W·sr⁻¹·m⁻²·µm⁻¹ (Eq. 1, §3.1)."""
    x = radiance/lref
    return x * g / np.sqrt(x + n0)


def nedl(radiance: np.ndarray, snr_value: np.ndarray) -> np.ndarray:
    """Noise-equivalent delta radiance, L / SNR, in W·sr⁻¹·m⁻²·µm⁻¹ (Eq. 2, §3.1)."""
    return radiance / snr_value


def ara(
    radiance: np.ndarray,
    lref: np.ndarray,
    ara_a: np.ndarray,
    ara_b: np.ndarray,
    k: float,
    s0: float,
) -> np.ndarray:
    """ARA for uniform scenes, as a fraction; k fractional, s0 in electrons (Eq. 4, §3.2).

    Linear sum, not RSS (§9.4); ara_a is kept though negligible (§9.5).
    """
    return k + s0 / (ara_a + ara_b * radiance/lref)


def rxra_gain(n_columns: int, sigma_eq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Equalization residual gain 1 + N(0, sigma^2), [I, K] (Eq. 7, §3.4).

    One draw per (column, channel): that structure is the striping (§9.2).
    """
    if n_columns < 1:
        raise ValueError(f"n_columns must be >= 1, got {n_columns}")
    sigma_eq = np.asarray(sigma_eq, dtype=float)
    if np.any(sigma_eq < 0.0):
        raise ValueError("sigma_eq must be non-negative")
    return 1.0 + rng.normal(0.0, 1.0, size=(n_columns, sigma_eq.size)) * sigma_eq


def apply_rxra(cube: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Apply the Eq. 7 gain [I, K] to a cube [I, J, K], broadcast along-track (§3.4)."""
    cube = np.asarray(cube, dtype=float)
    gain = np.asarray(gain, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"cube must be [I, J, K], got shape {cube.shape}")
    if gain.shape != (cube.shape[0], cube.shape[2]):
        raise ValueError(
            f"gain must be [I, K] = {(cube.shape[0], cube.shape[2])}, got {gain.shape}"
        )
    return cube * gain[:, np.newaxis, :]
