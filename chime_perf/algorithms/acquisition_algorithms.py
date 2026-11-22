"""Image operations for spectral response, spatial response and radiometric errors."""

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from eopl import image_t
from scipy.interpolate import RegularGridInterpolator

from chime_perf.algorithms.algorithms import require_units
from chime_perf.algorithms.backends import (
    cpp_pixels,
    cpp_spectral_formula,
    native_module,
    select_backend,
    tile_call,
    tiles,
)


def spectral_response(
    centres, wavelength_nm, sigma_nm, support_sigma=3, backend=None, threads=1
):
    """Normalise Gaussian channel responses, repeating edge samples for missing tails."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if (
        wavelength_nm.size < 2
        or not np.isfinite(wavelength_nm).all()
        or np.any(np.diff(wavelength_nm) <= 0)
    ):
        raise ValueError("Input wavelengths must be finite and strictly increasing.")

    centre = centres.image()
    require_units(centre, "nm")
    sigma_nm = np.asarray(sigma_nm)
    if np.any(sigma_nm <= 0) or not np.isfinite(sigma_nm).all():
        raise ValueError("Spectral response widths must be finite and positive.")
    centre_values = centre.values
    if not np.isfinite(centre_values).all():
        raise ValueError("Channel centres must be finite.")
    if not np.isfinite(support_sigma) or support_sigma <= 0:
        raise ValueError("Kernel support must be finite and positive.")

    # Extend wavelengths at each edge's sampling interval to cover every response tail.
    radius = support_sigma * sigma_nm
    lower = float(np.min(centre_values - radius))
    upper = float(np.max(centre_values + radius))
    step_left = wavelength_nm[1] - wavelength_nm[0]
    step_right = wavelength_nm[-1] - wavelength_nm[-2]
    left = max(0, int(np.ceil((wavelength_nm[0] - lower) / step_left)))
    right = max(0, int(np.ceil((upper - wavelength_nm[-1]) / step_right)))
    padded_nm = np.concatenate(
        (
            wavelength_nm[0] - step_left * np.arange(left, 0, -1),
            wavelength_nm,
            wavelength_nm[-1] + step_right * np.arange(1, right + 1),
        )
    )

    count = wavelength_nm.size
    backend = select_backend(centres, backend)
    if backend == "dask" and "detector_column" in centre.dims:
        # Each task builds a few detector columns using the same global padding.
        blocks = da.from_array(centre_values, chunks=(8, -1))
        weights = blocks.map_blocks(
            spectral_response_python,
            sigma_nm=sigma_nm,
            radius=radius,
            padded_nm=padded_nm,
            left=left,
            count=count,
            new_axis=2,
            chunks=(*blocks.chunks, (count,)),
            dtype=float,
            meta=np.empty((0, 0, 0)),
        )
    elif backend == "cpp":
        shape = centre_values.shape
        weights = native_module().spectral_response(
            np.ascontiguousarray(centre_values, dtype=float),
            np.ascontiguousarray(np.broadcast_to(sigma_nm, shape), dtype=float).ravel(),
            np.ascontiguousarray(np.broadcast_to(radius, shape), dtype=float).ravel(),
            padded_nm,
            np.gradient(padded_nm),
            left,
            count,
            threads,
        )
    else:
        weights = spectral_response_python(
            centre_values, sigma_nm, radius, padded_nm, left, count
        )

    coords = {name: centre[name] for name in centre.dims}
    coords["sample_SPE"] = wavelength_nm
    image = xr.DataArray(
        weights,
        dims=(*centre.dims, "sample_SPE"),
        coords=coords,
        name=centre.name,
        attrs={
            "units": "1",
            "spectral_edge_mode": "edge",
            "padding_samples_left": left,
            "padding_samples_right": right,
        },
    )
    return image_t(image)


spectral_response.cuda_support = False
spectral_response.dask_support = True
spectral_response.input_access = ("read",)


def spectral_response_python(centre_values, sigma_nm, radius, padded_nm, left, count):
    # The extra axis compares every channel centre with every input wavelength.
    distance = padded_nm - centre_values[..., None]
    response = np.exp(-0.5 * (distance / sigma_nm[..., None]) ** 2)
    response *= np.abs(distance) <= radius[..., None]
    response *= np.gradient(padded_nm)

    # Padded values equal the nearest edge sample. Add their weights to that sample
    # instead of allocating a padded copy of every image spectrum.
    weights = response[..., left : left + count].copy()
    weights[..., 0] += response[..., :left].sum(axis=-1)
    weights[..., -1] += response[..., left + count :].sum(axis=-1)
    total = weights.sum(axis=-1, keepdims=True)
    if np.any(total == 0):
        raise ValueError(
            "Input spectral sampling is too sparse to resolve a channel response."
        )
    weights /= total
    return weights


def rotation_parameters(image, heading_deg):
    """Use the same scene centre for all tiles."""
    angle = np.deg2rad(heading_deg)
    xc = float((image.ACT[0] + image.ACT[-1]) / 2)
    yc = float((image.ALT[0] + image.ALT[-1]) / 2)
    return xc, yc, np.cos(angle), np.sin(angle)


def sensor_coordinates(image, heading_deg):
    """Map a sensor grid to the source map, rotating about the scene centre."""
    return rotated_coordinates(
        image.ACT.values, image.ALT.values, rotation_parameters(image, heading_deg)
    )


def rotated_coordinates(act, alt, rotation):
    xc, yc, cosine, sine = rotation
    a, b = (act - xc)[:, None], (alt - yc)[None, :]
    return xc + a * cosine - b * sine, yc + a * sine + b * cosine


def column_coordinates(image, heading_deg, detector_spacing_m, xs=slice(None), ys=slice(None)):
    """Find the detector ACT column covering each requested source-map pixel."""
    xc, yc, cosine, sine = rotation_parameters(image, heading_deg)
    x = (image.ACT.values[xs] - xc)[:, None]
    y = (image.ALT.values[ys] - yc)[None, :]

    # Inverse of the rotation used when evaluating the sensor-grid image.
    sensor_act = xc + x * cosine + y * sine
    column = np.floor((sensor_act - float(image.ACT[0])) / detector_spacing_m + 0.5)
    return column.astype(int)


def channel_centres(
    data, nominal_nm, spacing_m, heading_deg, sigma_error_nm, common_fraction, rng
):
    """Draw a fixed wavelength shift for each detector column and channel."""
    if not np.isfinite(sigma_error_nm) or sigma_error_nm < 0:
        raise ValueError("spectral_error_sigma_nm must be finite and non-negative.")
    if not np.isfinite(common_fraction) or not 0 <= common_fraction <= 1:
        raise ValueError("spectral_error_common_fraction must be between zero and one.")

    # The extrema of the affine detector coordinate occur at the image corners.
    corners = slice(None, None, max(1, data.image().sizes["ACT"] - 1))
    ends = slice(None, None, max(1, data.image().sizes["ALT"] - 1))
    columns = column_coordinates(data.image(), heading_deg, spacing_m, corners, ends)
    indices = np.arange(columns.min(), columns.max() + 1)
    shifts = np.zeros((indices.size, len(nominal_nm)))

    if sigma_error_nm > 0:
        common = rng.normal()
        independent = rng.normal(size=shifts.shape)
        shifts = sigma_error_nm * (
            np.sqrt(common_fraction) * common + np.sqrt(1 - common_fraction) * independent
        )

    centres = xr.DataArray(
        nominal_nm + shifts,
        dims=("detector_column", "SPE"),
        coords={"detector_column": indices, "SPE": nominal_nm},
        name=data.image().name,
        attrs={"units": "nm"},
    )
    return image_t(centres)


channel_centres.cuda_support = False
channel_centres.dask_support = True
channel_centres.input_access = ("read",)


def integrate_spectral(data, response, spacing_m, heading_deg, backend=None, threads=1):
    """Sum the input radiance with the response belonging to each detector column."""
    source = data.image().transpose("ACT", "ALT", "SPE")
    weights = response.image()
    if not np.array_equal(source.SPE.values, weights.sample_SPE.values):
        raise ValueError("Spectral response wavelengths must match the input image.")
    backend = select_backend(data, backend)
    if backend == "dask":
        values = integrate_spectral_dask(source, weights, spacing_m, heading_deg)
    elif backend == "cpp":
        columns = column_coordinates(source, heading_deg, spacing_m)
        values = native_module().integrate(
            cpp_pixels(source.values),
            np.ascontiguousarray(columns, dtype=np.intp),
            np.ascontiguousarray(weights.values, dtype=float),
            np.ascontiguousarray(weights.detector_column.values, dtype=np.intp),
            threads,
        )
    else:
        columns = column_coordinates(source, heading_deg, spacing_m)
        values = integrate_spectral_python(
            source.values, columns, weights.values, weights.detector_column.values
        )

    result = xr.DataArray(
        values,
        dims=source.dims,
        coords={"ACT": source.ACT, "ALT": source.ALT, "SPE": weights.SPE},
        name=source.name,
        attrs=source.attrs.copy(),
    )
    result.SPE.attrs["units"] = "nm"
    return image_t(result)


integrate_spectral.cuda_support = False
integrate_spectral.dask_support = True
integrate_spectral.input_access = ("read", "read")


def integrate_spectral_python(pixels, columns, weights, column_ids):
    values = np.empty((*pixels.shape[:2], weights.shape[1]))

    # Only non-zero weights contribute; missing distant bands do not mask a channel.
    for index, column in enumerate(column_ids):
        mask = columns == column
        samples = pixels[mask]
        if not samples.size:
            continue
        for channel, weight in enumerate(weights[index]):
            support = weight > 0
            values[mask, channel] = samples[:, support] @ weight[support]
    return values


def integrate_spectral_dask(source, weights, spacing_m, heading_deg):
    pixels = da.asarray(source.data).rechunk({2: -1})
    responses = weights.data
    column_ids = weights.detector_column.values
    shape = (*pixels.shape[:2], weights.sizes["SPE"])

    def make_tile(xs, ys):
        tile_columns = column_coordinates(source, heading_deg, spacing_m, xs, ys)
        first = np.searchsorted(column_ids, tile_columns.min())
        last = np.searchsorted(column_ids, tile_columns.max(), side="right")
        needed = slice(first, last)
        return tile_call(
            integrate_spectral_python,
            pixels[xs, ys],
            tile_columns,
            responses[needed],
            column_ids[needed],
        )

    return tiles(pixels, shape, make_tile)


def rotate_to_sensor(data, heading_deg, edge_mode="edge", backend=None, threads=1):
    """Evaluate the map radiance at the rotated sensor-grid coordinates."""
    if edge_mode != "edge":
        raise ValueError("Only edge padding is supported.")
    source = data.image()
    backend = select_backend(data, backend)
    act, alt = source.ACT.values, source.ALT.values
    rotation = rotation_parameters(source, heading_deg)
    if backend == "dask":
        pixels = da.asarray(source.data)
        x, y = rotated_coordinates(
            da.from_array(act, chunks=pixels.chunks[0]),
            da.from_array(alt, chunks=pixels.chunks[1]),
            rotation,
        )
        values = (
            pixels if heading_deg == 0 else rotate_to_sensor_dask(pixels, act, alt, rotation)
        )
        result = source.copy(data=values)
    else:
        x, y = sensor_coordinates(source, heading_deg)
        if heading_deg == 0:
            result = source.copy(deep=True)
        else:
            lookup_x = np.clip(x, act[0], act[-1])
            lookup_y = np.clip(y, alt[0], alt[-1])
            if backend == "cpp":
                values = native_module().rotate(
                    cpp_pixels(source.values),
                    np.ascontiguousarray(act, dtype=float),
                    np.ascontiguousarray(alt, dtype=float),
                    lookup_x,
                    lookup_y,
                    threads,
                )
            else:
                values = rotate_to_sensor_python(source.values, act, alt, lookup_x, lookup_y)
            result = source.copy(data=values)

    # These coordinates retain the original map position of every output centre.
    result = result.assign_coords(map_ACT=(("ACT", "ALT"), x), map_ALT=(("ACT", "ALT"), y))
    result.map_ACT.attrs["units"] = "m"
    result.map_ALT.attrs["units"] = "m"
    result.attrs["heading_deg"] = float(heading_deg)
    result.attrs["spatial_edge_mode"] = edge_mode
    result.attrs["geometry"] = "sensor"
    return image_t(result)


rotate_to_sensor.cuda_support = False
rotate_to_sensor.dask_support = True
rotate_to_sensor.input_access = ("read",)


def rotate_to_sensor_python(values, act, alt, x, y):
    interpolate = RegularGridInterpolator((act, alt), values, bounds_error=True)
    return interpolate(np.stack((x, y), axis=-1))


def rotate_tile(values, act, alt, target_act, target_alt, rotation):
    x, y = rotated_coordinates(target_act, target_alt, rotation)
    return rotate_to_sensor_python(
        values, act, alt, np.clip(x, act[0], act[-1]), np.clip(y, alt[0], alt[-1])
    )


def rotate_to_sensor_dask(values, act, alt, rotation):
    def make_tile(xs, ys):
        target_act, target_alt = act[xs], alt[ys]
        # The four rotated corners bound all source coordinates for this output tile.
        x, y = rotated_coordinates(target_act[[0, -1]], target_alt[[0, -1]], rotation)
        # SciPy uses the interval to the right at exact nodes, except at the far edge.
        ia = (np.searchsorted(act, x, side="right") - 1).clip(0, len(act) - 2)
        ib = (np.searchsorted(alt, y, side="right") - 1).clip(0, len(alt) - 2)
        ix = slice(ia.min(), ia.max() + 2)
        iy = slice(ib.min(), ib.max() + 2)
        return tile_call(
            rotate_tile, values[ix, iy], act[ix], alt[iy], target_act, target_alt, rotation
        )

    return tiles(values, values.shape, make_tile)


def convolve_downsample(
    data, kernel, factor_act, factor_alt, edge_mode="edge", backend=None, threads=1
):
    """Evaluate spatial convolution only at the nominal centres that will be kept."""
    source = data.image()
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2 or any(size % 2 != 1 for size in kernel.shape):
        raise ValueError("The spatial kernel must have odd ACT and ALT sizes.")
    if not np.isfinite(kernel).all() or not np.isclose(kernel.sum(), 1):
        raise ValueError("The spatial kernel must be finite and sum to one.")
    if any(not isinstance(s, (int, np.integer)) or s < 1 for s in (factor_act, factor_alt)):
        raise ValueError("Spatial oversampling factors must be positive integers.")
    if edge_mode != "edge":
        raise ValueError("Only edge padding is supported.")

    act = np.arange(0, source.sizes["ACT"], factor_act)
    alt = np.arange(0, source.sizes["ALT"], factor_alt)
    backend = select_backend(data, backend)
    if backend == "dask":
        values = convolve_downsample_dask(
            source.data, kernel, act, alt, factor_act, factor_alt
        )
    elif backend == "cpp":
        values = native_module().convolve_downsample(
            cpp_pixels(source.values),
            np.ascontiguousarray(kernel),
            factor_act,
            factor_alt,
            threads,
        )
    else:
        radius_act, radius_alt = np.array(kernel.shape) // 2
        padding = ((radius_act, radius_act), (radius_alt, radius_alt), (0, 0))
        values = convolve_downsample_python(source.values, kernel, act, alt, padding)

    result = source.isel(ACT=act, ALT=alt).copy(data=values)
    result.attrs["spatial_edge_mode"] = edge_mode
    result.attrs["source_spatial_oversampling_ACT"] = int(factor_act)
    result.attrs["source_spatial_oversampling_ALT"] = int(factor_alt)
    result.attrs["spatial_oversampling_ACT"] = 1
    result.attrs["spatial_oversampling_ALT"] = 1
    return image_t(result)


convolve_downsample.cuda_support = False
convolve_downsample.dask_support = True
convolve_downsample.input_access = ("read",)


def convolve_downsample_python(values, kernel, act, alt, padding):
    padded = np.pad(values, padding, mode="edge")
    result = np.empty((act.size, alt.size, values.shape[2]))

    for i, u in enumerate(act):
        for j, v in enumerate(alt):
            patch = padded[u : u + kernel.shape[0], v : v + kernel.shape[1], :]
            # Reverse the kernel for convolution (rather than correlation).
            result[i, j] = np.einsum("abk,ab->k", patch, kernel[::-1, ::-1])
    return result


def convolve_downsample_dask(values, kernel, act, alt, factor_act, factor_alt):
    values = da.asarray(values)
    radius_act, radius_alt = np.array(kernel.shape) // 2
    shape = (len(act), len(alt), values.shape[2])
    tile_sizes = (
        max(1, max(values.chunks[0]) // factor_act),
        max(1, max(values.chunks[1]) // factor_alt),
    )

    def make_tile(xs, ys):
        # Tile on the global detector grid. Read actual neighbours at internal seams.
        centres_act, centres_alt = act[xs], alt[ys]
        act_start, act_stop = centres_act[0] - radius_act, centres_act[-1] + radius_act + 1
        alt_start, alt_stop = centres_alt[0] - radius_alt, centres_alt[-1] + radius_alt + 1
        ix = slice(max(0, act_start), min(values.shape[0], act_stop))
        iy = slice(max(0, alt_start), min(values.shape[1], alt_stop))

        # Repeat edge values only outside the whole image.
        padding = (
            (ix.start - act_start, act_stop - ix.stop),
            (iy.start - alt_start, alt_stop - iy.stop),
            (0, 0),
        )
        return tile_call(
            convolve_downsample_python,
            values[ix, iy],
            kernel,
            centres_act - centres_act[0],
            centres_alt - centres_alt[0],
            padding,
        )

    return tiles(values, shape, make_tile, tile_sizes)


def relative_radiance(data, lref, backend=None, threads=1):
    """Express noise-free radiance as a fraction of the reference radiance."""
    source = data.image()
    require_units(source, "W m-2 sr-1 um-1")
    if select_backend(data, backend) == "cpp":
        image = cpp_spectral_formula("relative_radiance", source, threads, lref)
    else:
        # lref has one value per channel; it repeats across both spatial axes.
        image = source.copy(data=(source / lref).data)
        if data.is_dask:
            image = image.copy(data=image.data.map_blocks(check_radiance, dtype=image.dtype))
        else:
            check_radiance(image.values)
    image.attrs["units"] = "1"
    return image_t(image)


relative_radiance.cuda_support = False
relative_radiance.dask_support = True
relative_radiance.input_access = ("read",)


def check_radiance(values):
    if np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Noise-free radiance must be non-negative.")
    return values


def temporal_snr(data, g, n0, backend=None, threads=1):
    """Calculate SNR = g r / sqrt(r + N0)."""
    r = data.image()
    require_units(r, "1")
    if select_backend(data, backend) == "cpp":
        result = cpp_spectral_formula("temporal_snr", r, threads, g, n0)
    else:
        result = r.copy(data=(g * r / np.sqrt(r + n0)).data)
    result.attrs["units"] = "1"
    return image_t(result)


temporal_snr.cuda_support = False
temporal_snr.dask_support = True
temporal_snr.input_access = ("read",)


def temporal_sigma(data, lref, g, n0, backend=None, threads=1):
    """Evaluate noise amplitude without dividing zero radiance by zero SNR."""
    r = data.image()
    require_units(r, "1")
    if select_backend(data, backend) == "cpp":
        result = cpp_spectral_formula("temporal_sigma", r, threads, lref / g, n0)
    else:
        result = r.copy(data=(lref / g * np.sqrt(r + n0)).data)
    result.attrs["units"] = "W m-2 sr-1 um-1"
    return image_t(result)


temporal_sigma.cuda_support = False
temporal_sigma.dask_support = True
temporal_sigma.input_access = ("read",)


def draw_noise(data, rng, enabled=True, backend=None, threads=1):
    """Draw independent Gaussian errors in the supplied standard deviation's units."""
    sigma = data.image()
    if select_backend(data, backend) == "cpp" and enabled:
        # NumPy draws in the same order as Python. OpenMP applies the amplitudes.
        arranged = sigma.transpose("ACT", "ALT", "SPE")
        draws = xr.DataArray(rng.normal(size=sigma.shape), dims=sigma.dims)
        draws = np.ascontiguousarray(draws.transpose(*arranged.dims).values)
        values = native_module().scale_noise(cpp_pixels(arranged.values), draws, threads)
        return image_t(arranged.copy(deep=False, data=values).transpose(*sigma.dims))
    if data.is_dask:
        if enabled:
            values = standard_noise_dask(rng, sigma.shape, sigma.chunks) * sigma.data
        else:
            values = da.zeros(sigma.shape, chunks=sigma.chunks)
    else:
        values = (
            rng.normal(size=sigma.shape) * sigma.values if enabled else np.zeros(sigma.shape)
        )
    return image_t(sigma.copy(data=values))


draw_noise.cuda_support = False
draw_noise.dask_support = True
draw_noise.input_access = ("read",)


def draw_standard_block(bit_generator, state, count):
    generator = bit_generator()
    generator.state = state
    return np.random.Generator(generator).standard_normal(count)


def standard_noise_dask(rng, shape, chunks):
    """Retain the reference seed sequence without allocating a whole noise cube."""
    size = int(np.prod(shape))
    if not size:
        return da.empty(shape, chunks=chunks)
    buffer = np.empty(min(2**20, size))
    bit_generator = type(rng.bit_generator)
    blocks = []
    for start in range(0, size, buffer.size):
        length = min(buffer.size, size - start)
        task = delayed(draw_standard_block, pure=True)(
            bit_generator, rng.bit_generator.state, length
        )
        blocks.append(da.from_delayed(task, shape=(length,), dtype=float))
        # Advance the shared RNG now so RXRA starts at the same state as in Python.
        # The worker later regenerates that block from its saved state.
        rng.standard_normal(out=buffer[:length])
    return da.concatenate(blocks).reshape(shape).rechunk(chunks)


def electron_signal(data, a, b, backend=None, threads=1):
    """Calculate the paper's equivalent electron signal n = a + b r."""
    r = data.image()
    require_units(r, "1")
    if select_backend(data, backend) == "cpp":
        result = cpp_spectral_formula("electron_signal", r, threads, a, b)
    else:
        result = r.copy(data=(a + b * r).data)
    result.attrs["units"] = "electron"
    return image_t(result)


electron_signal.cuda_support = False
electron_signal.dask_support = True
electron_signal.input_access = ("read",)


def absolute_offset(data, electrons, lref, a, b, coefficient, s0, backend=None, threads=1):
    """Calculate the additive ARA term, with the explicit zero-radiance limit."""
    signal = data.image()
    n = electrons.image()
    require_units(signal, "W m-2 sr-1 um-1")
    require_units(n, "electron")
    if select_backend(data, backend) == "cpp":
        source = signal.transpose("ACT", "ALT", "SPE")
        dtype = np.result_type(source.dtype, n.dtype)
        dtype = np.float32 if dtype == np.float32 else np.float64
        bands = (source.sizes["SPE"],)
        values = native_module().absolute_offset(
            np.ascontiguousarray(source.values, dtype=dtype),
            np.ascontiguousarray(n.transpose(*source.dims).values, dtype=dtype),
            np.ascontiguousarray(np.broadcast_to(lref / b, bands), dtype=float),
            np.ascontiguousarray(np.broadcast_to(np.asarray(a) == 0, bands)),
            coefficient * s0,
            threads,
        )
        return image_t(source.copy(deep=False, data=values).transpose(*signal.dims))
    if data.is_dask or electrons.is_dask:
        nonzero = n.data != 0
        ratio = da.where(nonzero, signal.data / da.where(nonzero, n.data, 1), 0)
    else:
        ratio = np.zeros(signal.shape)
        np.divide(signal.values, n.values, out=ratio, where=n.values != 0)

    # For a = 0, L / (b L / Lref) tends to Lref / b, including at L = 0.
    zero_limit = (signal.data == 0) & (np.asarray(a) == 0)
    ratio = np.where(zero_limit, np.asarray(lref / b), ratio)
    return image_t(signal.copy(data=coefficient * s0 * ratio))


absolute_offset.cuda_support = False
absolute_offset.dask_support = True
absolute_offset.input_access = ("read", "read")


def apply_absolute_error(data, noise, offset, gain, backend=None, threads=1):
    """Multiply only the signal by the ARA gain; retain the original noise draw."""
    source = data.image()
    if select_backend(data, backend) == "cpp":
        source, draws, bias = xr.align(
            source, noise.image(), offset.image(), join="exact", copy=False
        )
        dtype = np.result_type(source.dtype, draws.dtype, bias.dtype, gain)
        dtype = np.float32 if dtype == np.float32 else np.float64
        pixels = np.ascontiguousarray(source.values, dtype=dtype)
        draws = np.ascontiguousarray(draws.transpose(*source.dims).values, dtype=dtype)
        bias = np.ascontiguousarray(bias.transpose(*source.dims).values, dtype=dtype)
        values = native_module().apply_absolute_error(pixels, draws, bias, gain, threads)
        return image_t(source.copy(deep=False, data=values))

    values = gain * source + noise.image() + offset.image()
    return image_t(source.copy(data=values.transpose(*source.dims).data))


apply_absolute_error.cuda_support = False
apply_absolute_error.dask_support = True
apply_absolute_error.input_access = ("read", "read", "read")


def striping_gains(data, sigma, rng, enabled=True):
    """Draw one RXRA gain per ACT column and channel, shared along ALT."""
    source = data.image()
    values = np.ones((source.sizes["ACT"], source.sizes["SPE"]))
    if enabled:
        values += rng.normal(size=values.shape) * sigma

    gains = xr.DataArray(
        values,
        dims=("ACT", "SPE"),
        coords={"ACT": source.ACT, "SPE": source.SPE},
        name=source.name,
        attrs={"units": "1"},
    )
    return image_t(gains)


striping_gains.cuda_support = False
striping_gains.dask_support = True
striping_gains.input_access = ("read",)


def apply_striping(data, gains, backend=None, threads=1):
    """Apply the saved column gain to every ALT pixel."""
    source = data.image()
    if select_backend(data, backend) == "cpp":
        arranged = source.transpose("ACT", "ALT", "SPE")
        arranged, gain = xr.align(arranged, gains.image(), join="exact", copy=False)
        dtype = np.result_type(arranged.dtype, gain.dtype)
        dtype = np.float32 if dtype == np.float32 else np.float64
        values = native_module().apply_striping(
            np.ascontiguousarray(arranged.values, dtype=dtype),
            np.ascontiguousarray(gain.transpose("ACT", "SPE").values, dtype=dtype),
            threads,
        )
        return image_t(arranged.copy(deep=False, data=values).transpose(*source.dims))
    # Xarray matches ACT and SPE by name and repeats each gain along the missing ALT axis.
    values = (source * gains.image()).transpose(*source.dims)
    return image_t(source.copy(data=values.data))


apply_striping.cuda_support = False
apply_striping.dask_support = True
apply_striping.input_access = ("read", "read")


def select_channels(wavelength_nm, parameters):
    """Keep nominal centres inside the input range; response tails use edge padding."""
    centres = parameters.wavelength_nm
    covered = (centres >= wavelength_nm[0]) & (centres <= wavelength_nm[-1])
    if not covered.any():
        raise ValueError(
            "The input wavelength range contains no nominal CHIME channel centres."
        )
    return np.flatnonzero(covered)
