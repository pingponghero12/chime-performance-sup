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
void bind_precision(py::module_& module) {
    using namespace py::literals;
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
