"""Backend selection and a shared helper for independent spatial tiles."""

import dask.array as da
from dask import delayed


def validate_backend(backend):
    if backend in ("python", "dask"):
        return backend
    if backend in ("cpp", "cuda"):
        raise NotImplementedError(f"The CHIME {backend!r} backend is not implemented yet.")
    raise ValueError("backend must be python, dask, cpp, or cuda.")


def select_backend(data, backend):
    if backend is None:
        backend = "dask" if data.is_dask else "python"
    return validate_backend(backend)


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
