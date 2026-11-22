#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace chime {

namespace {

// The coefficients contain one value per wavelength, shared by all spatial pixels.
template <class T, class Formula, class... Coefficients>
Array<T> per_sample(const Array<T>& source, int threads, Formula formula,
                    const Coefficients&... coefficients) {
    const auto input = source.template unchecked<3>();
    const Index act = input.shape(0), alt = input.shape(1), bands = input.shape(2);
    if (((coefficients.ndim() != 1 || coefficients.size() != bands) || ...)) {
        throw std::invalid_argument("Expected one coefficient per input wavelength.");
    }
    Array<T> result({act, alt, bands});
    auto output = result.template mutable_unchecked<3>();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index pixel = 0; pixel < act * alt; ++pixel) {
        const Index x = pixel / alt, y = pixel % alt;
        for (Index k = 0; k < bands; ++k) {
            output(x, y, k) = formula(input(x, y, k), coefficients.data()[k]...);
        }
    }
    return result;
}

template <class T>
Array<T> propagate(const Array<T>& source, const Array<T>& path, const Array<T>& scale,
                   const Array<T>& transmission, const Array<T>& albedo, int threads) {
    return per_sample(source, threads, [](T rho, T p, T c, T t, T s) {
        const T denominator = 1 - s * rho;
        if (!std::isfinite(rho) || rho < 0 || denominator <= 0) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        return p + c * t * rho / denominator;
    }, path, scale, transmission, albedo);
}

template <class T>
T divide_valid(T numerator, T denominator) {
    return std::isfinite(denominator) && denominator > 0
        ? numerator / denominator : std::numeric_limits<T>::quiet_NaN();
}

template <class T>
Array<T> divide(const Array<T>& source, const Array<T>& scale, int threads) {
    return per_sample(source, threads, divide_valid<T>, scale);
}

template <class T>
Array<T> remove_atmosphere(const Array<T>& source, const Array<T>& path,
                           const Array<T>& transmission, int threads) {
    return per_sample(source, threads, [](T signal, T p, T t) {
        return divide_valid(signal - p, t);
    }, path, transmission);
}

template <class T>
Array<T> surface_reflectance(const Array<T>& source, const Array<T>& albedo, int threads) {
    return per_sample(source, threads, [](T y, T s) {
        return divide_valid(y, 1 + s * y);
    }, albedo);
}

template <class T>
Array<T> relative_radiance(const Array<T>& source, const Array<T>& reference, int threads) {
    auto result = per_sample(source, threads, [](T radiance, T lref) {
        return radiance / lref;
    }, reference);
    const auto* values = result.data();
    const auto size = result.size();
    int negative = 0;
    {
        py::gil_scoped_release release;
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:negative)
        for (Index i = 0; i < size; ++i) {
            negative |= std::isfinite(values[i]) && values[i] < 0;
        }
    }
    if (negative) {
        throw std::invalid_argument("Noise-free radiance must be non-negative.");
    }
    return result;
}

template <class T>
Array<T> temporal_snr(const Array<T>& source, const Array<T>& g, const Array<T>& n0, int threads) {
    return per_sample(source, threads, [](T r, T gain, T offset) {
        return gain * r / std::sqrt(r + offset);
    }, g, n0);
}

template <class T>
Array<T> temporal_sigma(const Array<T>& source, const Array<T>& scale, const Array<T>& n0, int threads) {
    return per_sample(source, threads, [](T r, T factor, T offset) {
        return factor * std::sqrt(r + offset);
    }, scale, n0);
}

template <class T>
Array<T> electron_signal(const Array<T>& source, const Array<T>& a, const Array<T>& b, int threads) {
    return per_sample(source, threads, [](T r, T intercept, T slope) {
        return intercept + slope * r;
    }, a, b);
}

template <class T>
void require_same_shape(const Array<T>& source, const Array<T>& other) {
    if (source.ndim() != 3 || other.ndim() != 3
        || !std::equal(source.shape(), source.shape() + 3, other.shape())) {
        throw std::invalid_argument("Radiometric images must have matching ACT, ALT and SPE shapes.");
    }
}

template <class T>
Array<double> scale_noise(const Array<T>& sigma, Array<double> draws, int threads) {
    if (sigma.ndim() != 3 || draws.ndim() != 3
        || !std::equal(sigma.shape(), sigma.shape() + 3, draws.shape())) {
        throw std::invalid_argument("Noise draws must match the image shape.");
    }
    const auto* amplitude = sigma.data();
    auto* samples = draws.mutable_data();
    const auto size = sigma.size();
    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index i = 0; i < size; ++i) {
        samples[i] *= amplitude[i];
    }
    return draws;
}

template <class T>
Array<double> absolute_offset(
    const Array<T>& source, const Array<T>& electrons, const Array<double>& limit,
    const Array<bool>& zero_a, double scale, int threads
) {
    require_same_shape(source, electrons);
    const auto bands = source.shape(2);
    if (limit.ndim() != 1 || zero_a.ndim() != 1 || limit.size() != bands || zero_a.size() != bands) {
        throw std::invalid_argument("Expected one ARA coefficient per channel.");
    }
    Array<double> result({source.shape(0), source.shape(1), bands});
    auto* output = result.mutable_data();
    const auto* signal = source.data();
    const auto* n = electrons.data();
    const auto* zero_limit = limit.data();
    const auto* zero_intercept = zero_a.data();
    const auto size = source.size();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index i = 0; i < size; ++i) {
        const Index band = i % bands;
        // Keep the reference division in pixel precision before applying the offset.
        const double ratio = signal[i] == 0 && zero_intercept[band] ? zero_limit[band]
            : (n[i] != 0 ? static_cast<double>(signal[i] / n[i]) : 0);
        output[i] = scale * ratio;
    }
    return result;
}

template <class T>
Array<T> apply_absolute_error(
    const Array<T>& source, const Array<T>& noise, const Array<T>& offset, T gain, int threads
) {
    require_same_shape(source, noise);
    require_same_shape(source, offset);
    const auto size = source.size();
    Array<T> result({source.shape(0), source.shape(1), source.shape(2)});
    auto* output = result.mutable_data();
    const auto* signal = source.data();
    const auto* noise_values = noise.data();
    const auto* offset_values = offset.data();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index i = 0; i < size; ++i) {
        output[i] = gain * signal[i] + noise_values[i] + offset_values[i];
    }
    return result;
}

template <class T>
Array<T> apply_striping(const Array<T>& source, const Array<T>& gains, int threads) {
    const auto input = source.template unchecked<3>();
    const auto gain = gains.template unchecked<2>();
    const Index act = input.shape(0), alt = input.shape(1), bands = input.shape(2);
    if (gain.shape(0) != act || gain.shape(1) != bands) {
        throw std::invalid_argument("Expected one RXRA gain per ACT column and channel.");
    }
    Array<T> result({act, alt, bands});
    auto output = result.template mutable_unchecked<3>();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index pixel = 0; pixel < act * alt; ++pixel) {
        const Index x = pixel / alt, y = pixel % alt;
        for (Index band = 0; band < bands; ++band) {
            output(x, y, band) = input(x, y, band) * gain(x, band);
        }
    }
    return result;
}

template <class T>
void bind_precision(py::module_& module) {
    using namespace py::literals;
    module.def("relative_radiance", &relative_radiance<T>, "source"_a.noconvert(),
               "reference"_a.noconvert(), "threads"_a);
    module.def("temporal_snr", &temporal_snr<T>, "source"_a.noconvert(),
               "g"_a.noconvert(), "n0"_a.noconvert(), "threads"_a);
    module.def("temporal_sigma", &temporal_sigma<T>, "source"_a.noconvert(),
               "scale"_a.noconvert(), "n0"_a.noconvert(), "threads"_a);
    module.def("electron_signal", &electron_signal<T>, "source"_a.noconvert(),
               "a"_a.noconvert(), "b"_a.noconvert(), "threads"_a);
    module.def("scale_noise", &scale_noise<T>, "sigma"_a.noconvert(),
               "draws"_a.noconvert(), "threads"_a);
    module.def("absolute_offset", &absolute_offset<T>, "source"_a.noconvert(),
               "electrons"_a.noconvert(), "limit"_a.noconvert(), "zero_a"_a.noconvert(),
               "scale"_a, "threads"_a);
    module.def("apply_absolute_error", &apply_absolute_error<T>, "source"_a.noconvert(),
               "noise"_a.noconvert(), "offset"_a.noconvert(), "gain"_a, "threads"_a);
    module.def("apply_striping", &apply_striping<T>, "source"_a.noconvert(),
               "gains"_a.noconvert(), "threads"_a);
    module.def("propagate", &propagate<T>, "source"_a.noconvert(), "path"_a.noconvert(),
               "scale"_a.noconvert(), "transmission"_a.noconvert(), "albedo"_a.noconvert(),
               "threads"_a);
    module.def("divide", &divide<T>, "source"_a.noconvert(), "scale"_a.noconvert(), "threads"_a);
    module.def("remove_atmosphere", &remove_atmosphere<T>, "source"_a.noconvert(),
               "path"_a.noconvert(), "transmission"_a.noconvert(), "threads"_a);
    module.def("surface_reflectance", &surface_reflectance<T>, "source"_a.noconvert(),
               "albedo"_a.noconvert(), "threads"_a);
}

}  // namespace

void bind_radiometry(py::module_& module) {
    bind_precision<float>(module);
    bind_precision<double>(module);
}

}  // namespace chime
