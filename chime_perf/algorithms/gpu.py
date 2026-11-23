"""CuPy helpers and CUDA kernels for the CHIME image operations."""

from functools import lru_cache
from math import prod
from pathlib import Path

import numpy as np
import xarray as xr
from eopl.data import require_cupy


def on_device(value):
    """Move a small coefficient to the device and retain any named axes."""
    if np.isscalar(value):
        return value
    cp = require_cupy()
    if isinstance(value, xr.DataArray):
        return value.copy(deep=False, data=cp.asarray(value.data))
    return cp.asarray(value)


@lru_cache
def sampling_module():
    cp = require_cupy()
    source = Path(__file__).with_name("cuda") / "sampling.cu"
    names = [
        f"{kernel}<{source_type}, {result_type}>"
        for kernel in (
            "interpolate_spectra", "interpolate_axis", "integrate_spectra", "convolve_centres"
        )
        for source_type in ("float", "double")
        for result_type in ("float", "double")
    ]
    return cp.RawModule(
        code=source.read_text(), options=("--std=c++11", "--fmad=false"),
        name_expressions=names,
    )


def launch_sampling(name, source, shape, *arguments, output_float32=False):
    cp = require_cupy()
    input_type = "float" if source.dtype == np.float32 else "double"
    output_type = "float" if output_float32 else "double"
    dtype = np.float32 if input_type == "float" else np.float64
    source = cp.ascontiguousarray(source, dtype=dtype)
    result = cp.empty(shape, dtype=np.float32 if output_float32 else np.float64)
    size = prod(shape)

    if size:
        kernel = sampling_module().get_function(f"{name}<{input_type}, {output_type}>")
        block_size = 256
        kernel(
            ((size + block_size - 1) // block_size,), (block_size,),
            (source, result, np.int64(size), *arguments),
        )
    return result


def interpolate_spectral_cuda(source, original, target, output_float32):
    cp = require_cupy()
    lower = np.clip(np.searchsorted(original, target, side="right") - 1, 0, original.size - 2)
    left = target - original[lower]
    right = target - original[lower + 1]
    upper_edge = right >= 0
    endpoint = (left <= 0) | upper_edge

    # In exact-band cases the kernel reads only the chosen source sample.
    return launch_sampling(
        "interpolate_spectra", source, (*source.shape[:2], target.size),
        np.int64(original.size), np.int64(target.size),
        cp.asarray(lower + upper_edge, dtype=np.int64),
        cp.asarray(original[lower + 1] - original[lower]),
        cp.asarray(np.where(endpoint, 0, left)), cp.asarray(right),
        output_float32=output_float32,
    )


def interpolate_spatial_cuda(source, original, target, axis):
    cp = require_cupy()
    lower = np.clip(np.searchsorted(original, target), 1, original.size - 1) - 1
    shape = list(source.shape)
    shape[axis] = target.size
    return launch_sampling(
        "interpolate_axis", source, shape,
        np.int64(source.shape[1]), np.int64(source.shape[2]),
        np.int64(shape[1]), np.int32(axis),
        cp.asarray(lower, dtype=np.int64),
        cp.asarray(original[lower + 1] - original[lower], dtype=float),
        cp.asarray(target - original[lower], dtype=float),
    )


def integrate_spectral_cuda(source, columns, weights, column_ids, output_float32):
    cp = require_cupy()
    if (
        weights.ndim != 3
        or weights.shape[0] != column_ids.size
        or weights.shape[-1] != source.shape[-1]
        or columns.shape != source.shape[:2]
    ):
        raise ValueError("Spectral response and detector columns must match the image.")
    order = np.argsort(column_ids)
    sorted_ids = column_ids[order]
    rows = np.searchsorted(sorted_ids, columns)
    if np.any(rows == sorted_ids.size) or not np.array_equal(sorted_ids[rows], columns):
        raise ValueError("A detector column is missing its spectral response.")
    weights = cp.ascontiguousarray(weights, dtype=float)
    support = weights > 0
    bands = source.shape[-1]
    channels = weights.shape[1]
    first = cp.argmax(support, axis=-1).astype(np.int64)
    stop = cp.where(support.any(axis=-1), bands - cp.argmax(support[..., ::-1], axis=-1), 0)
    return launch_sampling(
        "integrate_spectra", source, (*source.shape[:2], channels),
        np.int64(bands), np.int64(channels), cp.asarray(order[rows], dtype=np.int64),
        weights, first, stop.astype(np.int64), output_float32=output_float32,
    )


def convolve_downsample_cuda(source, kernel, shape, factor_act, factor_alt):
    cp = require_cupy()
    return launch_sampling(
        "convolve_centres", source, shape, *(np.int64(n) for n in source.shape),
        np.int64(shape[1]), np.int64(factor_act), np.int64(factor_alt),
        cp.asarray(np.ascontiguousarray(kernel)), *(np.int64(n) for n in kernel.shape),
    )


def rotate_to_sensor_cuda(source, act, alt, x, y):
    cp = require_cupy()
    from cupyx.scipy.interpolate import RegularGridInterpolator

    interpolate = RegularGridInterpolator((cp.asarray(act), cp.asarray(alt)), source)
    return interpolate(cp.asarray(np.stack((x, y), axis=-1)))
