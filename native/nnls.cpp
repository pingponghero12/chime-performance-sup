#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace chime {

namespace {

// Solve the least-squares problem on the active columns with Householder QR.
bool solve_active(
    const double* matrix, const double* spectrum, Index bands, Index materials,
    const std::vector<Index>& active, std::vector<double>& x,
    std::vector<double>& qr, std::vector<double>& rhs, double rank_tolerance
) {
    const Index count = static_cast<Index>(active.size());
    if (count > bands) {
        return false;
    }
    std::copy_n(spectrum, bands, rhs.data());
    for (Index row = 0; row < bands; ++row) {
        for (Index col = 0; col < count; ++col) {
            qr[row * count + col] = matrix[row * materials + active[col]];
        }
    }

    for (Index col = 0; col < count; ++col) {
        double norm = 0;
        for (Index row = col; row < bands; ++row) {
            norm = std::hypot(norm, qr[row * count + col]);
        }
        if (norm <= rank_tolerance) {
            return false;
        }
        auto& diagonal = qr[col * count + col];
        const double pivot = -std::copysign(norm, diagonal);
        // Store the reflector with first entry one, and its coefficient tau.
        const double leading = diagonal - pivot;
        for (Index row = col + 1; row < bands; ++row) {
            qr[row * count + col] /= leading;
        }
        const double tau = (pivot - diagonal) / pivot;
        diagonal = pivot;

        for (Index j = col + 1; j < count; ++j) {
            double dot = qr[col * count + j];
            for (Index row = col + 1; row < bands; ++row) {
                dot += qr[row * count + col] * qr[row * count + j];
            }
            qr[col * count + j] -= tau * dot;
            for (Index row = col + 1; row < bands; ++row) {
                qr[row * count + j] -= tau * dot * qr[row * count + col];
            }
        }
        double dot = rhs[col];
        for (Index row = col + 1; row < bands; ++row) {
            dot += qr[row * count + col] * rhs[row];
        }
        rhs[col] -= tau * dot;
        for (Index row = col + 1; row < bands; ++row) {
            rhs[row] -= tau * dot * qr[row * count + col];
        }
    }

    std::fill(x.begin(), x.end(), 0.0);
    for (Index i = count; i-- > 0;) {
        double value = rhs[i];
        for (Index j = i + 1; j < count; ++j) {
            value -= qr[i * count + j] * x[active[j]];
        }
        x[active[i]] = value / qr[i * count + i];
    }
    return true;
}

struct Workspace {
    std::vector<Index> active;
    std::vector<char> in_active, blocked;
    std::vector<double> candidate, residual, qr, rhs;

    Workspace(Index bands, Index materials)
        : in_active(materials), blocked(materials), candidate(materials),
          residual(bands), qr(bands * std::min(bands, materials)), rhs(bands) {
        active.reserve(materials);
    }

    bool fit(
        const double* matrix, const double* spectrum, Index bands, Index materials,
        double rank_tolerance, double matrix_scale, double* result
    ) {
        active.clear();
        std::fill(in_active.begin(), in_active.end(), false);
        std::fill(blocked.begin(), blocked.end(), false);
        std::fill_n(result, materials, 0.0);
        double spectrum_scale = 0;
        for (Index band = 0; band < bands; ++band) {
            if (!std::isfinite(spectrum[band])) {
                std::fill_n(result, materials, std::numeric_limits<double>::quiet_NaN());
                return true;
            }
            spectrum_scale = std::max(spectrum_scale, std::abs(spectrum[band]));
        }
        const double tolerance = 10 * rank_tolerance * spectrum_scale;
        const double zero = 10 * std::numeric_limits<double>::epsilon()
            * spectrum_scale / (matrix_scale > 0 ? matrix_scale : 1);

        // Lawson–Hanson active set. Add the material that most reduces the residual.
        Index steps = 0;
        const Index max_steps = 30 * materials * materials;
        while (steps++ < max_steps) {
            for (Index band = 0; band < bands; ++band) {
                double predicted = 0;
                for (const auto col : active) {
                    predicted += matrix[band * materials + col] * result[col];
                }
                residual[band] = spectrum[band] - predicted;
            }
            Index best = -1;
            double best_gradient = tolerance;
            for (Index col = 0; col < materials; ++col) {
                if (in_active[col] || blocked[col]) {
                    continue;
                }
                double gradient = 0;
                for (Index band = 0; band < bands; ++band) {
                    gradient += matrix[band * materials + col] * residual[band];
                }
                if (gradient > best_gradient) {
                    best_gradient = gradient;
                    best = col;
                }
            }
            if (best < 0) {
                return true;
            }

            active.push_back(best);
            in_active[best] = true;
            if (!solve_active(matrix, spectrum, bands, materials, active,
                              candidate, qr, rhs, rank_tolerance)) {
                // A dependent column adds no independent direction to the fit.
                active.pop_back();
                in_active[best] = false;
                blocked[best] = true;
                continue;
            }

            // Move toward the unconstrained fit until a coefficient reaches zero.
            while (true) {
                double fraction = 1;
                Index leaving = -1;
                for (const auto col : active) {
                    if (candidate[col] <= 0) {
                        const double denominator = result[col] - candidate[col];
                        const double step = denominator > 0 ? result[col] / denominator : 0;
                        if (leaving < 0 || step < fraction) {
                            fraction = step;
                            leaving = col;
                        }
                    }
                }
                if (leaving < 0) {
                    std::copy(candidate.begin(), candidate.end(), result);
                    break;
                }
                if (steps++ >= max_steps) {
                    return false;
                }
                for (const auto col : active) {
                    result[col] += fraction * (candidate[col] - result[col]);
                }
                active.erase(std::remove_if(active.begin(), active.end(), [&](Index col) {
                    if (col == leaving || result[col] <= zero) {
                        in_active[col] = false;
                        result[col] = 0;
                        return true;
                    }
                    return false;
                }), active.end());
                std::fill(blocked.begin(), blocked.end(), false);
                if (!solve_active(matrix, spectrum, bands, materials, active,
                                  candidate, qr, rhs, rank_tolerance)) {
                    return false;
                }
            }
        }
        return false;
    }
};

template <class T>
Array<double> fit_library(const Array<T>& values, const Array<double>& measured, int threads) {
    const auto input = values.template unchecked<3>();
    const auto matrix = measured.unchecked<2>();
    const auto nx = input.shape(0), ny = input.shape(1), bands = input.shape(2);
    const auto materials = matrix.shape(1);
    if (matrix.shape(0) != bands || bands == 0 || materials == 0) {
        throw std::invalid_argument("Expected a nonempty library on the measured bands.");
    }
    const auto* library = measured.data();
    double scale = 0;
    for (Index i = 0; i < measured.size(); ++i) {
        scale = std::max(scale, std::abs(library[i]));
    }
    const double rank_tolerance =
        std::numeric_limits<double>::epsilon() * std::max(bands, materials) * scale;
    Array<double> result({nx, ny, materials});
    auto output = result.mutable_unchecked<3>();
    int failed = 0;
    {
        py::gil_scoped_release release;
        #pragma omp parallel num_threads(threads) reduction(|:failed)
        {
            Workspace work(bands, materials);
            std::vector<double> spectrum(bands);
            #pragma omp for schedule(static)
            for (Index p = 0; p < nx * ny; ++p) {
                const Index x = p / ny, y = p % ny;
                std::copy_n(input.data(x, y, 0), bands, spectrum.data());
                if (!work.fit(library, spectrum.data(), bands, materials,
                              rank_tolerance, scale, output.mutable_data(x, y, 0))) {
                    failed = 1;
                }
            }
        }
    }
    if (failed) {
        throw std::runtime_error("The native NNLS fit did not converge.");
    }
    return result;
}

Array<double> reconstruct_library(
    const Array<double>& weights, const Array<double>& spectra, int threads
) {
    const auto mixture = weights.unchecked<3>();
    const auto library = spectra.unchecked<2>();
    const Index act = mixture.shape(0), alt = mixture.shape(1);
    const Index materials = mixture.shape(2), bands = library.shape(1);
    if (materials != library.shape(0)) {
        throw std::invalid_argument("The weights and library must use the same materials.");
    }
    Array<double> result({act, alt, bands});
    auto output = result.mutable_unchecked<3>();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (Index pixel = 0; pixel < act * alt; ++pixel) {
        const Index x = pixel / alt, y = pixel % alt;
        auto* spectrum = output.mutable_data(x, y, 0);
        std::fill_n(spectrum, bands, 0.0);
        for (Index material = 0; material < materials; ++material) {
            const double proportion = mixture(x, y, material);
            const auto* endmember = library.data(material, 0);
            #pragma omp simd
            for (Index band = 0; band < bands; ++band) {
                spectrum[band] += proportion * endmember[band];
            }
        }
    }
    return result;
}

}  // namespace

void bind_nnls(py::module_& module) {
    using namespace py::literals;
    module.def("reconstruct_library", &reconstruct_library, "weights"_a.noconvert(),
               "library"_a.noconvert(), "threads"_a);
    module.def("fit_library", &fit_library<float>, "values"_a.noconvert(),
               "measured"_a.noconvert(), "threads"_a);
    module.def("fit_library", &fit_library<double>, "values"_a.noconvert(),
               "measured"_a.noconvert(), "threads"_a);
}

}  // namespace chime
