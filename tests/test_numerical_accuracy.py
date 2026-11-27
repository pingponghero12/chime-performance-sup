"""Check selected CHIME operations against exact calculations under varied rounding."""

import ctypes
import platform
import random
from fractions import Fraction

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
eopl = pytest.importorskip("eopl")
operation_numerical_test = pytest.importorskip("eopl.validation").operation_numerical_test
pytest.importorskip("significantdigits")
dask = pytest.importorskip("dask")
propagate = pytest.importorskip("chime_perf.algorithms.athmo_algorithms").propagate
acquisition = pytest.importorskip("chime_perf.algorithms.acquisition_algorithms")

pytestmark = pytest.mark.skipif(
    platform.machine() != "x86_64" or platform.libc_ver()[0] != "glibc",
    reason="These tests use the glibc x86-64 floating-point rounding constants.",
)


@pytest.fixture(params=["python", "dask", "cpp"])
def backend(request):
    if request.param == "cpp":
        pytest.importorskip("chime_perf._cpp")
    return request.param


def check_rounding(operation, reference, *args, backend, dtype, min_bits):
    math = ctypes.CDLL("libm.so.6")
    original_mode = math.fegetround()
    # From fenv.h on Linux x86-64. Choose one rounding mode per complete run.
    nearest, downward, upward, toward_zero = 0, 0x400, 0x800, 0xC00
    modes = (nearest, downward, upward, toward_zero)
    rng = random.Random(42)

    def run(*inputs, mode=None):
        rounding = rng.choice(modes) if mode is None else mode
        status = math.fesetround(rounding)
        assert status == 0
        try:
            # The rounding mode is thread-local. Evaluate all arithmetic on this thread.
            return operation(*inputs, backend=backend, threads=1).image().compute()
        finally:
            status = math.fesetround(original_mode)
            assert status == 0

    with dask.config.set(scheduler="synchronous"):
        lower = run(*args, mode=downward)
        upper = run(*args, mode=upward)
        assert lower.dtype == upper.dtype == np.dtype(dtype)
        assert (lower != upper).any(), "The rounding modes must affect the calculation."
        assert operation_numerical_test(run, reference, *args, min_bits=min_bits)


@pytest.mark.parametrize("dtype,min_bits", [("float32", 20), ("float64", 48)])
def test_propagation_retains_significant_bits(backend, dtype, min_bits):
    """Vary rounding and compare TOA radiance to an exact rational reference."""
    reflectances = [Fraction(numerator, 16) for numerator in (3, 5, 9, 13)]
    path, scale, transmission, albedo = 3, 10, Fraction(3, 4), Fraction(1, 8)
    values = np.asarray(reflectances, dtype=dtype).reshape(2, 2, 1)
    scene = xr.DataArray(
        values, dims=("ACT", "ALT", "SPE"), coords={"SPE": [550.0]}, attrs={"units": "1"},
    )
    scene.SPE.attrs["units"] = "nm"
    atmosphere = xr.Dataset(coords={"SPE": scene.SPE})
    for name, value, unit in (
        ("path_radiance", path, "W m-2 sr-1 um-1"),
        ("solar_scale", scale, "W m-2 sr-1 um-1"),
        ("transmission", transmission, "1"),
        ("spherical_albedo", albedo, "1"),
    ):
        atmosphere[name] = ("SPE", np.asarray([value], dtype=dtype), {"units": unit})

    # Integer fractions avoid rounding until the exact answer is converted to binary64.
    expected = [
        float(path + scale * transmission * r / (1 - albedo * r)) for r in reflectances
    ]
    reference = scene.copy(data=np.asarray(expected).reshape(scene.shape))
    if backend == "dask":
        scene = scene.chunk({"ACT": 1, "ALT": 1})

    check_rounding(
        propagate, reference, eopl.image_t(scene), atmosphere,
        backend=backend, dtype=dtype, min_bits=min_bits,
    )


def test_convolution_retains_significant_bits(backend):
    """A faint impulse adds exactly its kernel weight to one retained sample."""
    impulse = Fraction(1, 2**50)
    values = np.ones((5, 5, 1))
    values[2, 2, 0] = float(1 + impulse)
    scene = xr.DataArray(values, dims=("ACT", "ALT", "SPE"))
    kernel = np.array([[0, 1, 2], [1, 3, 2], [2, 2, 3]]) / 16

    expected = np.ones((3, 3, 1))
    expected[1, 1, 0] = float(1 + Fraction(3, 16) * impulse)
    retained = scene.isel(ACT=slice(None, None, 2), ALT=slice(None, None, 2))
    reference = retained.copy(data=expected)
    if backend == "dask":
        scene = scene.chunk({"ACT": 2, "ALT": 2})

    check_rounding(
        acquisition.convolve_downsample, reference, eopl.image_t(scene), kernel, 2, 2,
        backend=backend, dtype="float64", min_bits=48,
    )
