"""Spectral model: Gaussian ISRF
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np


def isrf_sigma(fwhm_nm: np.ndarray) -> np.ndarray:
    """ISRF standard deviation in nm, FWHM / (2*sqrt(2*ln 2)) (Eq. 8, §4.1)."""
    if np.any(np.asarray(fwhm_nm) <= 0.0):
        raise ValueError("fwhm_nm must be strictly positive")
    return fwhm_nm * 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def isrf(wavelength_nm: np.ndarray, centre_nm: np.ndarray, sigma_nm: np.ndarray) -> np.ndarray:
    """Gaussian ISRF in nm⁻¹, normalised to integrate to 1 (Eq. 9, §4.1)."""
    if np.any(np.asarray(sigma_nm) <= 0.0):
        raise ValueError("sigma_nm must be strictly positive")
    z = (wavelength_nm - centre_nm) / sigma_nm
    return np.exp(-0.5 * z**2) / (sigma_nm * np.sqrt(2.0 * np.pi))