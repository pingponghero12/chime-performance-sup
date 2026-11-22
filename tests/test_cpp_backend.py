"""Check the optional OpenMP backend against the Python reference."""

import numpy as np
import pytest
import yaml

xr = pytest.importorskip("xarray")
image_t = pytest.importorskip("eopl").image_t
pytest.importorskip("numba")
pytest.importorskip("chime_perf._cpp")

fit_library = pytest.importorskip("chime_perf.algorithms.ssgm_algorithms").fit_library
main = pytest.importorskip("chime_perf.run").main
spectral_response = pytest.importorskip(
    "chime_perf.algorithms.acquisition_algorithms"
).spectral_response


def test_cpp_nnls_matches_python_with_missing_spectra():
    """Fit mixtures, reject missing spectra and keep the material coordinates."""
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
        {"measured": (("SPE", "material"), measured)},
        coords={"SPE": wavelengths, "material": ["a", "b", "c", "d"]},
    )

    expected = fit_library(image_t(scene), library).image()
    actual = fit_library(image_t(scene), library, backend="cpp", threads=2).image()

    xr.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


def test_cpp_spectral_response_accepts_widths_by_detector_column():
    centres = image_t(
        xr.DataArray(
            [[480.0, 504.0], [481.0, 503.0]],
            dims=("detector_column", "SPE"),
            attrs={"units": "nm"},
        )
    )
    wavelengths = np.arange(480.0, 505.0)
    widths = np.array([[2.0, 3.0], [3.0, 2.0]])

    expected = spectral_response(centres, wavelengths, widths).image()
    actual = spectral_response(centres, wavelengths, widths, backend="cpp", threads=2).image()

    xr.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_cpp_full_chain_matches_python(tmp_path, precision):
    """Compare all four saved products on a rotated scene with noise and missing pixels."""
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
        "run_options": {"precision": precision, "num_cores": 2},
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

    for backend in ("python", "cpp"):
        config["run_options"]["backend"] = backend
        settings_file = tmp_path / f"{backend}.yml"
        settings_file.write_text(yaml.safe_dump(config))
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
                str(tmp_path / backend),
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
            xr.open_dataset(tmp_path / "cpp" / name) as actual,
        ):
            xr.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            xr.testing.assert_identical(
                actual.coords.to_dataset(), expected.coords.to_dataset()
            )
            for variable in expected.data_vars:
                assert actual[variable].attrs == expected[variable].attrs
                np.testing.assert_array_equal(
                    np.isnan(actual[variable]), np.isnan(expected[variable])
                )
