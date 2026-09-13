"""Spatial model: Gaussian PSF from the Nyquist MTF
"""

from __future__ import annotations
import warnings
import numpy as np


def psf_sigma_px(mtf_nyquist: float, OF :int) -> float:
    """PSF sigma in pixels, (p/pi)*sqrt(-2*ln(MTF)) (Eq. 10, §5)."""
    if not 0.0 < mtf_nyquist < 1.0:
        raise ValueError(f"mtf_nyquist must be in (0, 1), got {mtf_nyquist}")
    return OF*float(np.sqrt(-2.0 * np.log(mtf_nyquist)) / np.pi)


def psf_kernel(
    sigma_act_px: float, sigma_alt_px: float, size: int | None = None,
) -> np.ndarray:
    """Separable 2-D Gaussian PSF kernel (ACT, ALT), normalised to 1 (Eq. 11, §5).

    Sigmas are in pixels of the grid it will be applied to: the oversampled grid
    ``size`` defaults to the smallest odd value covering +/-4 sigma.
    """
    if sigma_act_px <= 0.0 or sigma_alt_px <= 0.0:
        raise ValueError("PSF sigmas must be strictly positive")

    if size is None:
        size = 2 * int(np.ceil(4.0 * max(sigma_act_px, sigma_alt_px))) + 1
    if size < 1 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd number of samples, got {size}")

    offsets = np.arange(size, dtype=float) - (size - 1) / 2.0
    kernel = np.outer(
        np.exp(-0.5 * (offsets / sigma_act_px) ** 2),
        np.exp(-0.5 * (offsets / sigma_alt_px) ** 2),
    )
    return kernel / kernel.sum()

