#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace chime {

namespace {

// NumPy/Numba spatial interpolation uses the interval on the left of an exact node.
Index left_interval(const Array<double>& coordinates, double position) {
    const auto* begin = coordinates.data();
    return std::clamp<Index>(
        std::lower_bound(begin, begin + coordinates.size(), position) - begin - 1,
        0, coordinates.size() - 2
    );
}

// SciPy's regular-grid interpolator uses the interval on the right of an exact node.
Index right_interval(const Array<double>& coordinates, double position) {
    const auto* begin = coordinates.data();
    return std::clamp<Index>(
        std::upper_bound(begin, begin + coordinates.size(), position) - begin - 1,
        0, coordinates.size() - 2
    );
}

void check_axis(const Array<double>& axis, Index length) {
    if (axis.ndim() != 1 || axis.size() != length || length < 2) {
        throw std::invalid_argument("Expected a coordinate vector matching the image axis.");
    }
}

template <class T>
Array<double> interpolate_spatial(
    const Array<T>& values, const Array<double>& source, const Array<double>& target,
    int axis, int threads
) {
    const auto input = values.template unchecked<3>();
    if (axis < 0 || axis > 1 || target.ndim() != 1) {
        throw std::invalid_argument("Spatial interpolation requires ACT or ALT coordinates.");
    }
    check_axis(source, input.shape(axis));
    const auto nx = axis == 0 ? target.size() : input.shape(0);
    const auto ny = axis == 1 ? target.size() : input.shape(1);
    const auto bands = input.shape(2);
    Array<double> result({nx, ny, bands});
    auto output = result.mutable_unchecked<3>();

    std::vector<Index> lower(target.size());
    std::vector<double> distance(target.size()), spacing(target.size());
    for (Index i = 0; i < target.size(); ++i) {
        lower[i] = left_interval(source, *target.data(i));
        distance[i] = *target.data(i) - *source.data(lower[i]);
        spacing[i] = *source.data(lower[i] + 1) - *source.data(lower[i]);
    }

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index pixel = 0; pixel < nx * ny; ++pixel) {
        const Index x = pixel / ny, y = pixel % ny;
        const Index i = axis == 0 ? x : y;
        const Index lo = lower[i];
        for (Index band = 0; band < bands; ++band) {
            const T first = axis == 0 ? input(lo, y, band) : input(x, lo, band);
            const T second = axis == 0 ? input(lo + 1, y, band) : input(x, lo + 1, band);
            // Subtract in the input precision before applying the double grid spacing.
            output(x, y, band) = first + (second - first) / spacing[i] * distance[i];
        }
    }
    return result;
}

template <class T>
Array<double> interpolate_spectral(
    const Array<T>& values, const Array<double>& source, const Array<double>& target,
    int threads
) {
    const auto input = values.template unchecked<3>();
    check_axis(source, input.shape(2));
    if (target.ndim() != 1) {
        throw std::invalid_argument("Expected a vector of target wavelengths.");
    }
    const auto nx = input.shape(0), ny = input.shape(1);
    const auto count = target.size();
    Array<double> result({nx, ny, count});
    auto output = result.mutable_unchecked<3>();
    const auto src = source.unchecked<1>();
    const auto dst = target.unchecked<1>();
    std::vector<Index> lower(count);
    for (Index i = 0; i < count; ++i) {
        lower[i] = right_interval(source, dst(i));
    }

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index pixel = 0; pixel < nx * ny; ++pixel) {
        const Index x = pixel / ny, y = pixel % ny;
        for (Index i = 0; i < count; ++i) {
            const Index j = lower[i];
            const double first = input(x, y, j), second = input(x, y, j + 1);
            if (dst(i) <= src(j)) {
                output(x, y, i) = first;
            } else if (dst(i) >= src(j + 1)) {
                output(x, y, i) = second;
            } else {
                const double slope = (second - first) / (src(j + 1) - src(j));
                double value = first + slope * (dst(i) - src(j));
                if (std::isnan(value)) {
                    value = second + slope * (dst(i) - src(j + 1));
                    if (std::isnan(value) && first == second) {
                        value = first;
                    }
                }
                output(x, y, i) = value;
            }
        }
    }
    return result;
}

template <class T>
Array<double> rotate(
    const Array<T>& values, const Array<double>& act, const Array<double>& alt,
    const Array<double>& lookup_x, const Array<double>& lookup_y, int threads
) {
    const auto input = values.template unchecked<3>();
    check_axis(act, input.shape(0));
    check_axis(alt, input.shape(1));
    const auto xmap = lookup_x.unchecked<2>();
    const auto ymap = lookup_y.unchecked<2>();
    const auto nx = xmap.shape(0), ny = xmap.shape(1), bands = input.shape(2);
    if (ymap.shape(0) != nx || ymap.shape(1) != ny) {
        throw std::invalid_argument("Rotation lookup arrays must have matching shapes.");
    }
    Array<double> result({nx, ny, bands});
    auto output = result.mutable_unchecked<3>();
    const auto xcoord = act.unchecked<1>();
    const auto ycoord = alt.unchecked<1>();
    const auto* xbegin = act.data();
    const auto* ybegin = alt.data();
    const Index xlength = act.size(), ylength = alt.size();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index pixel = 0; pixel < nx * ny; ++pixel) {
        const Index x = pixel / ny, y = pixel % ny;
        const double xt = xmap(x, y), yt = ymap(x, y);
        const Index i = std::clamp<Index>(
            std::upper_bound(xbegin, xbegin + xlength, xt) - xbegin - 1, 0, xlength - 2);
        const Index j = std::clamp<Index>(
            std::upper_bound(ybegin, ybegin + ylength, yt) - ybegin - 1, 0, ylength - 2);
        const double u = (xt - xcoord(i)) / (xcoord(i + 1) - xcoord(i));
        const double v = (yt - ycoord(j)) / (ycoord(j + 1) - ycoord(j));
        const double w00 = (1 - u) * (1 - v), w01 = (1 - u) * v;
        const double w10 = u * (1 - v), w11 = u * v;
        for (Index band = 0; band < bands; ++band) {
            output(x, y, band) = input(i, j, band) * w00 + input(i, j + 1, band) * w01
                + input(i + 1, j, band) * w10 + input(i + 1, j + 1, band) * w11;
        }
    }
    return result;
}

template <class T>
Array<double> integrate(
    const Array<T>& values, const Array<Index>& columns, const Array<double>& weights,
    const Array<Index>& column_ids, int threads
) {
    const auto input = values.template unchecked<3>();
    const auto pixels = columns.unchecked<2>();
    const auto response = weights.unchecked<3>();
    const auto nx = input.shape(0), ny = input.shape(1), bands = input.shape(2);
    const auto channels = response.shape(1);
    if (pixels.shape(0) != nx || pixels.shape(1) != ny || response.shape(2) != bands
        || column_ids.ndim() != 1 || column_ids.size() != response.shape(0)) {
        throw std::invalid_argument("Spectral response and detector columns must match the image.");
    }
    const auto ids = column_ids.unchecked<1>();
    std::unordered_map<Index, Index> response_row;
    for (Index row = 0; row < ids.shape(0); ++row) {
        response_row[ids(row)] = row;
    }
    std::vector<Index> column_index(nx * ny);
    for (Index p = 0; p < nx * ny; ++p) {
        const auto id = pixels(p / ny, p % ny);
        const auto match = response_row.find(id);
        if (match == response_row.end()) {
            throw std::invalid_argument("A detector column is missing its spectral response.");
        }
        column_index[p] = match->second;
    }
    // A missing sample outside a channel's support must never contaminate the sum.
    using Link = std::pair<Index, double>;
    std::vector<std::vector<Link>> support(ids.shape(0) * channels);
    for (Index col = 0; col < ids.shape(0); ++col) {
        for (Index channel = 0; channel < channels; ++channel) {
            auto& links = support[col * channels + channel];
            for (Index band = 0; band < bands; ++band) {
                if (response(col, channel, band) > 0) {
                    links.emplace_back(band, response(col, channel, band));
                }
            }
        }
    }
    Array<double> result({nx, ny, channels});
    auto output = result.mutable_unchecked<3>();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index p = 0; p < nx * ny; ++p) {
        const Index x = p / ny, y = p % ny;
        const Index first = column_index[p] * channels;
        for (Index channel = 0; channel < channels; ++channel) {
            double sum = 0;
            for (const auto& [band, weight] : support[first + channel]) {
                sum += input(x, y, band) * weight;
            }
            output(x, y, channel) = sum;
        }
    }
    return result;
}

template <class T>
Array<double> convolve_downsample(
    const Array<T>& values, const Array<double>& kernel, Index factor_act, Index factor_alt,
    int threads
) {
    const auto input = values.template unchecked<3>();
    const auto weights = kernel.unchecked<2>();
    if (factor_act < 1 || factor_alt < 1 || weights.shape(0) % 2 != 1
        || weights.shape(1) % 2 != 1 || input.shape(0) < 1 || input.shape(1) < 1) {
        throw std::invalid_argument("Expected positive sampling factors and an odd kernel.");
    }
    const auto nx = (input.shape(0) - 1) / factor_act + 1;
    const auto ny = (input.shape(1) - 1) / factor_alt + 1;
    const auto bands = input.shape(2);
    const auto radius_x = weights.shape(0) / 2, radius_y = weights.shape(1) / 2;
    Array<double> result({nx, ny, bands});
    auto output = result.mutable_unchecked<3>();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index pixel = 0; pixel < nx * ny; ++pixel) {
        const Index x = pixel / ny, y = pixel % ny;
        auto* spectrum = output.mutable_data(x, y, 0);
        std::fill_n(spectrum, bands, 0.0);
        for (Index i = 0; i < weights.shape(0); ++i) {
            const Index u = std::clamp(x * factor_act + i - radius_x, Index{0}, input.shape(0) - 1);
            for (Index j = 0; j < weights.shape(1); ++j) {
                const Index v = std::clamp(y * factor_alt + j - radius_y, Index{0}, input.shape(1) - 1);
                const double weight = weights(weights.shape(0) - i - 1, weights.shape(1) - j - 1);
                const auto* sample = input.data(u, v, 0);
                #pragma omp simd
                for (Index band = 0; band < bands; ++band) {
                    spectrum[band] += sample[band] * weight;
                }
            }
        }
    }
    return result;
}

Array<double> spectral_response(
    const Array<double>& centres, const Array<double>& sigma, const Array<double>& radius,
    const Array<double>& wavelengths, const Array<double>& widths,
    Index left, Index count, int threads
) {
    if (centres.ndim() < 1 || sigma.ndim() != 1 || radius.ndim() != 1
        || wavelengths.ndim() != 1 || widths.ndim() != 1) {
        throw std::invalid_argument("Invalid spectral-response array dimensions.");
    }
    const Index n = centres.size();
    if (n < 1 || sigma.size() != n || radius.size() != n
        || wavelengths.size() != widths.size() || left < 0 || count < 1
        || left + count > wavelengths.size()) {
        throw std::invalid_argument("Spectral-response arrays have incompatible sizes.");
    }
    std::vector<Index> shape(centres.shape(), centres.shape() + centres.ndim());
    shape.push_back(count);
    Array<double> result(shape);
    const auto* centre = centres.data();
    const auto* wave = wavelengths.data();
    const auto* spectral_width = widths.data();
    const auto* stddev = sigma.data();
    const auto* support = radius.data();
    auto* output = result.mutable_data();
    const Index padded = wavelengths.size();
    int empty = 0;
    {
        py::gil_scoped_release release;
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:empty)
        for (Index row = 0; row < n; ++row) {
            auto* result_row = output + row * count;
            std::fill_n(result_row, count, 0.0);
            double edge_left = 0, edge_right = 0;
            for (Index k = 0; k < padded; ++k) {
                const double distance = wave[k] - centre[row];
                if (std::abs(distance) > support[row]) {
                    continue;
                }
                const double z = distance / stddev[row];
                const double weight = std::exp(-0.5 * (z * z)) * spectral_width[k];
                if (k < left) {
                    edge_left += weight;
                } else if (k >= left + count) {
                    edge_right += weight;
                } else {
                    result_row[k - left] = weight;
                }
            }
            result_row[0] += edge_left;
            result_row[count - 1] += edge_right;
            double total = 0;
            for (Index k = 0; k < count; ++k) {
                total += result_row[k];
            }
            if (total == 0) {
                empty = 1;
                continue;
            }
            for (Index k = 0; k < count; ++k) {
                result_row[k] /= total;
            }
        }
    }
    if (empty) {
        throw std::invalid_argument("Input spectral sampling is too sparse to resolve a channel response.");
    }
    return result;
}

template <class T>
void bind_precision(py::module_& module) {
    using namespace py::literals;
    module.def("interpolate_spatial", &interpolate_spatial<T>, "values"_a.noconvert(),
               "source"_a.noconvert(), "target"_a.noconvert(), "axis"_a, "threads"_a);
    module.def("interpolate_spectral", &interpolate_spectral<T>, "values"_a.noconvert(),
               "source"_a.noconvert(), "target"_a.noconvert(), "threads"_a);
    module.def("rotate", &rotate<T>, "values"_a.noconvert(), "act"_a.noconvert(),
               "alt"_a.noconvert(), "lookup_x"_a.noconvert(), "lookup_y"_a.noconvert(), "threads"_a);
    module.def("integrate", &integrate<T>, "values"_a.noconvert(), "columns"_a.noconvert(),
               "weights"_a.noconvert(), "column_ids"_a.noconvert(), "threads"_a);
    module.def("convolve_downsample", &convolve_downsample<T>, "values"_a.noconvert(),
               "kernel"_a.noconvert(), "factor_act"_a, "factor_alt"_a, "threads"_a);
}

}  // namespace

void bind_sampling(py::module_& module) {
    bind_precision<float>(module);
    bind_precision<double>(module);
    module.def("spectral_response", &spectral_response);
}

}  // namespace chime
