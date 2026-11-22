"""Spatial interpolation, spectral interpolation and spectral-library reconstruction."""

import dask.array as da
import numpy as np
import xarray as xr
from eopl import image_t
from numba import njit
from scipy.interpolate import interp1d
from scipy.optimize import nnls

from .algorithms import grid, reflectance_image
from .backends import cpp_pixels, native_module, select_backend, tile_call, tiles


def spatial_grid(coordinates, detector_spacing_m, oversampling="auto"):
    """Place oversampled samples at detector spacing divided by an integer factor."""
    coordinates = grid(coordinates)

    if not np.isfinite(detector_spacing_m) or detector_spacing_m <= 0:
        raise ValueError("Detector spacing must be finite and positive.")

    if oversampling == "auto":
        source_spacing_m = np.median(np.diff(coordinates))

        # Round to the nearest integer; an exact half chooses the larger oversampling factor.
        oversampling = max(1, int(np.floor(detector_spacing_m / source_spacing_m + 0.5)))

    if (
        isinstance(oversampling, (bool, np.bool_))
        or not isinstance(oversampling, (int, np.integer))
        or oversampling < 1
    ):
        raise ValueError("Spatial oversampling must be a positive integer or 'auto'.")

    step_m = detector_spacing_m / oversampling

    # Nearest-factor selection may slightly coarsen the scene (5.5 m -> 6 m).
    # The first input centre anchors both the oversampled grid and nominal detector centres.
    target = grid(coordinates, step_m, allow_coarsening=True)

    return target, int(oversampling)


def validate_spatial_sampling(data, detector_spacing_m):
    """Require regular ACT and ALT grids with integer detector-to-scene spacing ratios."""
    image = data.image()

    for dimension in ("ACT", "ALT"):
        coordinates = image[dimension].values
        _, factor = spatial_grid(coordinates, detector_spacing_m)
        step_m = detector_spacing_m / factor

        if not np.allclose(np.diff(coordinates), step_m, rtol=1e-7, atol=1e-9):
            raise ValueError(
                f"{dimension} has non-integer or irregular spatial oversampling; "
                "enable scene.spatial_interpolation to resample it."
            )


@njit(cache=True, nogil=True)
def interpolate_spatial_axis(values, source_m, target_m, axis):
    """Apply linear interpolation along ACT or ALT without full-image temporaries."""
    shape = list(values.shape)
    shape[axis] = len(target_m)
    result = np.empty((shape[0], shape[1], shape[2]))

    # Each requested position lies between two neighbouring source samples.
    upper = np.searchsorted(source_m, target_m)

    for i in range(len(target_m)):
        hi = min(max(upper[i], 1), len(source_m) - 1)
        lo = hi - 1
        distance = target_m[i] - source_m[lo]
        spacing = source_m[hi] - source_m[lo]

        if axis == 0:
            for j in range(values.shape[1]):
                for k in range(values.shape[2]):
                    slope = (values[hi, j, k] - values[lo, j, k]) / spacing
                    result[i, j, k] = values[lo, j, k] + slope * distance
        else:
            for j in range(values.shape[0]):
                for k in range(values.shape[2]):
                    slope = (values[j, hi, k] - values[j, lo, k]) / spacing
                    result[j, i, k] = values[j, lo, k] + slope * distance

    return result


def interpolate_spatial(
    data, detector_spacing_m=30.0, oversampling="auto", backend=None, threads=1
):
    """Interpolate ACT and ALT onto grids with integer spatial oversampling."""
    backend = select_backend(data, backend)
    source = data.image().transpose("ACT", "ALT", "SPE")
    if backend == "dask" and not data.is_dask:
        source = source.chunk({"ACT": 128, "ALT": 128, "SPE": -1})

    act, factor_act = spatial_grid(source.ACT.values, detector_spacing_m, oversampling)
    alt, factor_alt = spatial_grid(source.ALT.values, detector_spacing_m, oversampling)

    # Applying linear interpolation along ACT and ALT gives bilinear interpolation.
    # Existing missing samples remain NaN; no points outside the scene are requested.
    result = source.copy(deep=False)

    for axis, target in enumerate((act, alt)):
        dimension = source.dims[axis]
        original = result[dimension].values

        if np.array_equal(target, original):
            continue

        if backend == "dask":
            values = interpolate_spatial_dask(result.data, original, target, axis)
        elif backend == "cpp":
            values = native_module().interpolate_spatial(
                cpp_pixels(result.values),
                np.ascontiguousarray(original, dtype=float),
                np.ascontiguousarray(target),
                axis,
                threads,
            )
        else:
            values = interpolate_spatial_axis(result.values, original, target, axis)

        # Xarray handles coordinate attributes and auxiliary coordinates only;
        # The selected kernel fills the new pixel array.
        coordinates = result.coords.to_dataset().interp({dimension: target}).coords
        result = xr.DataArray(
            values,
            dims=result.dims,
            coords=coordinates,
            name=source.name,
            attrs=source.attrs.copy(),
        )

    result.attrs = {
        **source.attrs,
        "detector_spacing_m": float(detector_spacing_m),
        "spatial_oversampling_ACT": factor_act,
        "spatial_oversampling_ALT": factor_alt,
        "detector_origin_ACT_m": float(act[0]),
        "detector_origin_ALT_m": float(alt[0]),
    }

    return image_t(result)


interpolate_spatial.cuda_support = False
interpolate_spatial.dask_support = True
interpolate_spatial.input_access = ("read",)


def interpolate_spatial_dask(values, original, target, axis):
    values = da.asarray(values)
    shape = list(values.shape)
    shape[axis] = len(target)

    def make_tile(xs, ys):
        selection = [xs, ys]
        points = target[selection[axis]]
        # Use the same brackets as the reference kernel, across chunk edges too.
        upper = np.searchsorted(original, points).clip(1, len(original) - 1)
        neighbours = slice(upper.min() - 1, upper.max() + 1)
        selection[axis] = neighbours
        pixels = values[tuple(selection)]
        return tile_call(interpolate_spatial_axis, pixels, original[neighbours], points, axis)

    return tiles(values, shape, make_tile)


@njit(cache=True, nogil=True)
def interpolate_spectra(spectra, wavelength_nm, target_nm):
    """Evaluate the same straight-line spectral interpolant at every pixel."""
    result = np.empty((len(spectra), len(target_nm)))

    for pixel in range(len(spectra)):
        result[pixel] = np.interp(target_nm, wavelength_nm, spectra[pixel])

    return result


def interpolate_spectral(data, step_nm=None, backend=None, threads=1):
    """Interpolate each pixel spectrum onto the requested SPE grid."""
    source = data.image().transpose("ACT", "ALT", "SPE")

    source_nm = grid(source.SPE.values)
    target_nm = grid(source_nm, step_nm)

    backend = select_backend(data, backend)
    if backend == "dask":
        result = interpolate_spectral_dask(source.data, source_nm, target_nm)
    elif backend == "cpp":
        result = native_module().interpolate_spectral(
            cpp_pixels(source.values),
            np.ascontiguousarray(source_nm),
            np.ascontiguousarray(target_nm),
            threads,
        )
    else:
        result = interpolate_spectral_python(source.values, source_nm, target_nm)

    image = xr.DataArray(
        result,
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": source.ACT, "ALT": source.ALT, "SPE": target_nm},
        name=source.name,
        attrs={**source.attrs, "spectral_method": "linear"},
    )
    image.SPE.attrs = source.SPE.attrs.copy()

    return image_t(image)


interpolate_spectral.cuda_support = False
interpolate_spectral.dask_support = True
interpolate_spectral.input_access = ("read",)


def interpolate_spectral_python(values, source_nm, target_nm):
    # Each row contains one pixel spectrum.
    spectra = values.reshape(-1, len(source_nm))
    result = interpolate_spectra(spectra, source_nm, target_nm)
    return result.reshape(*values.shape[:2], len(target_nm))


def interpolate_spectral_dask(values, source_nm, target_nm):
    values = da.asarray(values).rechunk({2: -1})
    return values.map_blocks(
        interpolate_spectral_python,
        source_nm=source_nm,
        target_nm=target_nm,
        chunks=(*values.chunks[:2], (len(target_nm),)),
        dtype=float,
        meta=np.empty((0, 0, 0)),
    )


def fit_library(data, library, backend=None, threads=1):
    """Fit non-negative library weights independently at each spatial pixel."""
    reflectance = reflectance_image(data)
    measured = library["measured"].transpose("SPE", "material")

    # These library values must include the INPUT sensor's band-response averages.
    if not np.array_equal(measured.SPE.values, reflectance.SPE.values):
        raise ValueError("Library measured bands must match the input SPE coordinates.")

    if not np.isfinite(measured.values).all():
        raise ValueError("Measured-band library values must be finite.")

    backend = select_backend(data, backend)
    if backend == "dask":
        weights = fit_library_dask(reflectance.data, measured.values)
    elif backend == "cpp":
        weights = native_module().fit_library(
            cpp_pixels(reflectance.values),
            np.ascontiguousarray(measured.values, dtype=float),
            threads,
        )
    else:
        weights = fit_library_python(reflectance.values, measured.values)

    image = xr.DataArray(
        weights,
        dims=("ACT", "ALT", "material"),
        coords={"ACT": reflectance.ACT, "ALT": reflectance.ALT, "material": measured.material},
        name=reflectance.name,
        attrs={**reflectance.attrs, "units": "1"},
    )

    return image_t(image)


fit_library.cuda_support = False
fit_library.dask_support = True
fit_library.input_access = ("read",)


def fit_library_python(values, measured):
    weights = np.full((*values.shape[:2], measured.shape[1]), np.nan)
    for act in range(values.shape[0]):
        for alt in range(values.shape[1]):
            spectrum = values[act, alt, :]
            if np.isfinite(spectrum).all():
                weights[act, alt, :] = nnls(measured, spectrum)[0]
    return weights


def fit_library_dask(values, measured):
    values = da.asarray(values).rechunk({2: -1})
    return values.map_blocks(
        fit_library_python,
        measured=measured,
        chunks=(*values.chunks[:2], (measured.shape[1],)),
        dtype=float,
        meta=np.empty((0, 0, 0)),
    )


def reconstruct_library(data, library, target_nm):
    """Reconstruct rho(lambda) as the weighted sum of the library spectra."""
    weights = data.image()
    endmembers = library["reflectance"].transpose("library_SPE", "material")
    source_nm = grid(endmembers.library_SPE.values)

    if not np.array_equal(weights.material.values, endmembers.material.values):
        raise ValueError("Material order must match between fitted weights and library.")

    if not np.isfinite(endmembers.values).all():
        raise ValueError("Library spectra must be finite.")

    if target_nm[0] < source_nm[0] or target_nm[-1] > source_nm[-1]:
        raise ValueError("Library must cover the complete target SPE grid.")

    # Interpolate the library once, then apply rho_m = sum_r c_r e_rm to every pixel.
    spectra = interp1d(source_nm, endmembers.values, axis=0)(target_nm)
    reflectance = weights.data @ spectra.T

    image = xr.DataArray(
        reflectance,
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": weights.ACT, "ALT": weights.ALT, "SPE": target_nm},
        name=weights.name,
        attrs={**weights.attrs, "units": "1", "spectral_method": "NNLS"},
    )
    image.SPE.attrs["units"] = "nm"

    return image_t(image)


reconstruct_library.cuda_support = False
reconstruct_library.dask_support = True
reconstruct_library.input_access = ("read",)
