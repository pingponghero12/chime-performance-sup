"""Check execution backends against the Python reference using EOPL."""

import numpy as np
import pytest
import yaml

dask = pytest.importorskip("dask")
xr = pytest.importorskip("xarray")
image_t = pytest.importorskip("eopl").image_t
operation_compare_test = pytest.importorskip("eopl.validation").operation_compare_test
pytest.importorskip("numba")

ssgm = pytest.importorskip("chime_perf.algorithms.ssgm_algorithms")
main = pytest.importorskip("chime_perf.run").main
acquisition = pytest.importorskip("chime_perf.algorithms.acquisition_algorithms")


def require_cuda():
    cupy = pytest.importorskip("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        device_count = 0
    if not device_count:
        pytest.skip("A CUDA device is required.")


@pytest.mark.parametrize("input_dtype", [np.float32, np.float64])
def test_cpp_spectral_interpolation_writes_requested_precision(input_dtype):
    """Round on write while preserving exact nodes, clamped edges and missing bands."""
    native = pytest.importorskip("chime_perf._cpp")
    source = np.array([410.0, 415.0, 421.0, 430.0])
    target = np.array([400.0, 410.0, 412.8, 415.0, 416.2, 421.0, 423.8, 430.0, 440.0])
    pixels = np.array(
        [[[.1, .25, .4, .7], [.15, np.nan, .3, .8], [.2, np.inf, np.inf, .9]]],
        dtype=input_dtype,
    )
    reference = np.array([[np.interp(target, source, spectrum) for spectrum in pixels[0]]])

    full = native.interpolate_spectral(pixels, source, target, 2)
    rounded = native.interpolate_spectral(pixels, source, target, 2, output_float32=True)

    assert full.dtype == np.float64
    assert rounded.dtype == np.float32
    np.testing.assert_array_equal(full, reference)
    np.testing.assert_array_equal(rounded, reference.astype(np.float32))


@pytest.mark.parametrize("input_dtype", [np.float32, np.float64])
def test_cpp_spectral_integration_preserves_positive_support(input_dtype):
    """Use the matching detector row and leave missing values outside support alone."""
    native = pytest.importorskip("chime_perf._cpp")
    pixels = np.array([[[np.nan, 8, 12, np.inf], [4, np.nan, 12, 16]]], dtype=input_dtype)
    columns = np.array([[7, -3]], dtype=np.intp)
    column_ids = np.array([7, -3], dtype=np.intp)
    weights = np.array([
        [[0, .25, .75, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        [[.5, 0, .5, -1], [0, 1, 0, 0], [0, 0, 0, 0]],
    ])
    expected = np.array([[[11, np.inf, 0], [8, np.nan, 0]]])

    full = native.integrate(pixels, columns, weights, column_ids, 2)
    rounded = native.integrate(pixels, columns, weights, column_ids, 2, output_float32=True)

    assert full.dtype == np.float64
    assert rounded.dtype == np.float32
    np.testing.assert_array_equal(full, expected)
    np.testing.assert_array_equal(rounded, expected.astype(np.float32))


@pytest.mark.parametrize("backend", ["cpp", "dask", "cuda"])
def test_nnls_matches_python_with_missing_spectra(backend):
    """Fit mixtures, reject missing spectra and keep the material coordinates."""
    if backend == "cpp":
        pytest.importorskip("chime_perf._cpp")
    if backend == "cuda":
        require_cuda()
    rng = np.random.default_rng(2)
    measured = rng.uniform(0.05, 0.8, (12, 4))
    mixtures = rng.uniform(0, 0.5, (6, 7, 4))
    mixtures[::2, :, 1] = 0
    values = mixtures @ measured.T
    values[0, 0, 2] = np.nan
    wavelengths = 500 + np.arange(12)
    scene = xr.DataArray(
        values,
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": np.arange(6), "ALT": np.arange(7), "SPE": wavelengths},
        attrs={"units": "1"},
    )
    for axis, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
        scene[axis].attrs["units"] = unit
    library = xr.Dataset(
        {
            "measured": (("SPE", "material"), measured),
            "reflectance": (("library_SPE", "material"), measured),
        },
        coords={
            "SPE": wavelengths,
            "library_SPE": wavelengths,
            "material": ["a", "b", "c", "d"],
        },
    )

    def fit(data, library, backend):
        if backend == "dask":
            data = data.chunk({"ACT": 3, "ALT": 4})
        image = image_t(data)
        if backend == "cuda":
            image.to_device()
        result = ssgm.fit_library(image, library, backend=backend, threads=2)
        assert result.is_cuda == (backend == "cuda")
        return result

    assert operation_compare_test(
        fit, fit, scene, library,
        kwargs={"backend": backend}, reference_kwargs={"backend": "python"},
        rtol=1e-11, atol=1e-12,
    )

    weights = fit(scene, library, "python").image()
    target = np.linspace(wavelengths[0], wavelengths[-1], 23)

    def reconstruct(data, library, target, backend):
        if backend == "dask":
            data = data.chunk({"ACT": 3, "ALT": 4})
        image = image_t(data)
        if backend == "cuda":
            image.to_device()
        result = ssgm.reconstruct_library(image, library, target, backend=backend, threads=2)
        assert result.is_cuda == (backend == "cuda")
        return result

    assert operation_compare_test(
        reconstruct, reconstruct, weights, library, target,
        kwargs={"backend": backend}, reference_kwargs={"backend": "python"},
        rtol=1e-11, atol=1e-12,
    )


def test_cuda_nnls_with_dependent_columns():
    """Compare fitted spectra when more than one set of weights is valid."""
    require_cuda()
    from chime_perf.algorithms.gpu import fit_library_cuda

    # The first two endmembers are identical and there are fewer bands than materials.
    measured = np.array([[1., 1., 0.], [0., 0., 1.]])
    source = np.array([[[.25, .75], [-1., 2.], [np.inf, .5], [0., 0.]]])
    weights = fit_library_cuda(source, measured).get()

    assert (weights[:, [0, 1, 3], :] >= 0).all()
    np.testing.assert_allclose(weights[:, [0, 1, 3], :] @ measured.T,
                               [[[.25, .75], [0., 2.], [0., 0.]]])
    assert np.isnan(weights[0, 2]).all()


def test_cpp_spectral_response_accepts_widths_by_detector_column():
    pytest.importorskip("chime_perf._cpp")
    centres = image_t(
        xr.DataArray(
            [[480.0, 504.0], [481.0, 503.0]],
            dims=("detector_column", "SPE"),
            attrs={"units": "nm"},
        )
    )
    wavelengths = np.arange(480.0, 505.0)
    widths = np.array([[2.0, 3.0], [3.0, 2.0]])

    assert operation_compare_test(
        acquisition.spectral_response, acquisition.spectral_response,
        centres, wavelengths, widths,
        kwargs={"backend": "cpp", "threads": 2}, reference_kwargs={"backend": "python"},
        rtol=1e-12, atol=1e-12,
    )


def test_cpp_noise_keeps_the_same_draws_and_rng_state():
    pytest.importorskip("chime_perf._cpp")
    sigma = image_t(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4), dims=("SPE", "ALT", "ACT")
    )
    def noise_and_next_draw(data, rng, backend):
        noise = acquisition.draw_noise(data, rng, backend=backend, threads=2)
        return noise, rng.normal()

    assert operation_compare_test(
        noise_and_next_draw, noise_and_next_draw, sigma, np.random.default_rng(42),
        kwargs={"backend": "cpp"}, reference_kwargs={"backend": "python"},
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("chunks", [{"ACT": 4}, {"ALT": 6}, {"ACT": 4, "ALT": 6}])
def test_dask_convolution_matches_python_across_chunk_edges(chunks):
    """Detector centres on chunk seams still use samples from neighbouring chunks."""
    values = np.random.default_rng(42).uniform(1, 10, (11, 13, 3))
    values[4, 6, 1] = np.nan
    image = xr.DataArray(values, dims=("ACT", "ALT", "SPE"))
    kernel = np.arange(1.0, 10).reshape(3, 3) / 45

    def sample(data, backend):
        if backend == "dask":
            data = data.chunk(chunks)
        result = acquisition.convolve_downsample(image_t(data), kernel, 2, 3, backend=backend)
        assert result.is_dask == (backend == "dask")
        return result

    assert operation_compare_test(
        sample, sample, image, kwargs={"backend": "dask"},
        reference_kwargs={"backend": "python"}, rtol=0, atol=0,
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["cpp", "dask", "cuda"])
def test_full_chain_matches_python(tmp_path, monkeypatch, precision, backend):
    """Compare all four saved products on a rotated scene with noise and missing pixels."""
    if backend == "cpp":
        pytest.importorskip("chime_perf._cpp")
    if backend == "cuda":
        require_cuda()
    rng = np.random.default_rng(42)
    input_wavelengths = np.arange(480.0, 623.0, 2)
    wavelengths = np.arange(480.0, 623.0)
    values = rng.uniform(0.1, 0.5, (19, 23, len(input_wavelengths))).astype(precision)
    values[0, 0, :] = np.nan
    values[9, 10, 15] = np.nan
    scene = xr.DataArray(
        values,
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": 4 * np.arange(19), "ALT": 5 * np.arange(23), "SPE": input_wavelengths},
        name="reflectance",
        attrs={"units": "1", "geometry": "map"},
    )
    for axis, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
        scene[axis].attrs["units"] = unit
    input_file = tmp_path / "scene.nc"
    scene.to_netcdf(input_file)

    atmosphere = xr.Dataset(coords={"SPE": wavelengths})
    atmosphere.SPE.attrs["units"] = "nm"
    for name, value, unit in (
        ("transmission", 0.8, "1"),
        ("spherical_albedo", 0.04, "1"),
        ("path_radiance", 3.0, "W m-2 sr-1 um-1"),
        ("solar_scale", 80.0, "W m-2 sr-1 um-1"),
    ):
        atmosphere[name] = (
            "SPE",
            np.full(wavelengths.size, value, dtype=precision),
            {"units": unit},
        )
    atmosphere_file = tmp_path / "atmosphere.nc"
    atmosphere.to_netcdf(atmosphere_file)

    config = {
        "run_options": {
            "precision": precision,
            "num_cores": 2,
            "dask_chunks": {"ACT": 8, "ALT": 9},
        },
        "scene": {"spatial_interpolation": True, "spectral_step_nm": 1.0},
        "instrument": {
            "heading_deg": 17.0,
            "spectral_error_sigma_nm": 0.25,
            "seed": 42,
            "ara_gain_coefficient": 1,
            "ara_offset_coefficient": 1,
        },
        "breakpoints": True,
    }

    for selected in ("python", backend):
        config["run_options"]["backend"] = selected
        settings_file = tmp_path / f"{selected}.yml"
        settings_file.write_text(yaml.safe_dump(config))
        # Execute the Dask graph without concurrent NetCDF access in this numerical test.
        with dask.config.set(scheduler="synchronous"):
            main(
                [
                    "--all",
                    "--input",
                    str(input_file),
                    "--atmosphere",
                    str(atmosphere_file),
                    "--config",
                    str(settings_file),
                    "--output",
                    str(tmp_path / selected),
                    "--save-intermediates",
                ]
            )

    tolerance = 1e-6 if precision == "float32" else 1e-12
    for name in (
        "ssgm/scene.nc",
        "athmo/toa.nc",
        "instrument/breakpoints/spectral_response.nc",
        "instrument/breakpoints/spatial_response.nc",
        "instrument/instrument.nc",
        "reconstruction/reconstructed.nc",
    ):
        with (
            xr.open_dataset(tmp_path / "python" / name) as expected,
            xr.open_dataset(tmp_path / backend / name) as actual,
        ):
            assert operation_compare_test(
                actual.load, expected.load, rtol=tolerance, atol=tolerance,
            ), f"{backend} differs from Python in {name}"
            xr.testing.assert_identical(
                actual.coords.to_dataset(), expected.coords.to_dataset()
            )
            for variable in expected.data_vars:
                assert actual[variable].attrs == expected[variable].attrs

    if backend == "cuda":
        from chime_perf import run

        # Verify the arrays arriving at each module are still on the GPU.
        handoffs = []

        def record_device(function):
            def wrapped(*args, **kwargs):
                handoffs.append(kwargs["data"].is_cuda)
                return function(*args, **kwargs)

            return wrapped

        for name in ("athmo", "instrument", "reconstruction"):
            monkeypatch.setattr(run, name, record_device(getattr(run, name)))

        output = tmp_path / "cuda-memory"
        main([
            "--all", "--input", str(input_file), "--atmosphere", str(atmosphere_file),
            "--config", str(settings_file), "--output", str(output),
        ])
        assert handoffs == [True, True, True]
        assert {path.name for path in output.iterdir()} == {"reconstruction"}
        name = "reconstruction/reconstructed.nc"
        with xr.open_dataset(output / name) as memory, xr.open_dataset(
            tmp_path / "cuda" / name
        ) as disk:
            xr.testing.assert_identical(memory, disk)
