"""Backend selection and a shared helper for independent spatial tiles."""

import dask.array as da
import numpy as np
from dask import delayed


def validate_backend(backend):
    if backend in ("python", "dask", "cpp"):
        return backend
    if backend == "cuda":
        raise NotImplementedError(f"The CHIME {backend!r} backend is not implemented yet.")
    raise ValueError("backend must be python, dask, cpp, or cuda.")


def select_backend(data, backend):
    if backend is None:
        backend = "dask" if data.is_dask else "python"
    return validate_backend(backend)


def native_module():
    """Load the optional extension only when the C++ backend is selected."""
    try:
        from chime_perf import _cpp
    except ImportError as error:
        raise RuntimeError(
            "Build CHIME with OpenMP first: pip install -e '.[pipeline]' -Cwheel.cmake=true"
        ) from error
    return _cpp


def cpp_pixels(values):
    """Keep the reference input precision without copying contiguous float cubes."""
    dtype = np.float32 if values.dtype == np.float32 else np.float64
    return np.ascontiguousarray(values, dtype=dtype)


def cpp_spectral_formula(name, image, threads, *coefficients):
    """Pass aligned spectra and coefficients to an OpenMP pixel formula."""
    types = [
        value if np.isscalar(value) else np.asarray(value).dtype for value in coefficients
    ]
    dtype = np.result_type(image.dtype, *types)
    dtype = np.float32 if dtype == np.float32 else np.float64
    pixels = np.ascontiguousarray(image.transpose("ACT", "ALT", "SPE").values, dtype=dtype)
    vectors = [
        np.ascontiguousarray(
            np.broadcast_to(np.asarray(coefficient), (image.sizes["SPE"],)), dtype=dtype
        )
        for coefficient in coefficients
    ]
    values = getattr(native_module(), name)(pixels, *vectors, threads)
    return (
        image.transpose("ACT", "ALT", "SPE")
        .copy(deep=False, data=values)
        .transpose(*image.dims)
    )


def tiles(values, shape, make_tile, tile_sizes=None):
    """Assemble a lazy output from tiles whose input region is chosen by the caller."""
    if tile_sizes is None:
        tile_sizes = tuple(max(chunks) for chunks in values.chunks[:2])
    act_size, alt_size = tile_sizes
    rows = []
    for act in range(0, shape[0], act_size):
        columns = []
        for alt in range(0, shape[1], alt_size):
            xs = slice(act, min(act + act_size, shape[0]))
            ys = slice(alt, min(alt + alt_size, shape[1]))
            shape_tile = (xs.stop - xs.start, ys.stop - ys.start, shape[2])
            task = make_tile(xs, ys)
            columns.append(da.from_delayed(task, shape=shape_tile, dtype=float))
        rows.append(da.concatenate(columns, axis=1))
    return da.concatenate(rows, axis=0)


def tile_call(kernel, *args):
    # A delayed tile receives only the sliced input blocks it needs.
    return delayed(kernel, pure=True)(*args)
