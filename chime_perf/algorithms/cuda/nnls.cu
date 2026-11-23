// Fit one spectrum per thread. The shared library QR has already reduced each
// spectrum to at most one row per material.

__device__ bool solve_active(
    const double* library, const double* spectrum, int rows, int materials,
    const int* active, int count, double tolerance, double* candidate,
    double* qr, double* rhs
) {
    for (int col = 0; col < materials; ++col) candidate[col] = 0;
    if (count > rows) return false;
    for (int row = 0; row < rows; ++row) {
        rhs[row] = spectrum[row];
        for (int col = 0; col < count; ++col) {
            qr[row * count + col] = library[row * materials + active[col]];
        }
    }

    // Householder QR of the selected columns keeps the solve in the linear
    // least-squares problem, without squaring the condition number.
    for (int col = 0; col < count; ++col) {
        double norm = 0;
        for (int row = col; row < rows; ++row) {
            norm = hypot(norm, qr[row * count + col]);
        }
        if (norm <= tolerance) return false;
        double& diagonal = qr[col * count + col];
        const double pivot = -copysign(norm, diagonal);
        const double leading = diagonal - pivot;
        for (int row = col + 1; row < rows; ++row) qr[row * count + col] /= leading;
        const double tau = (pivot - diagonal) / pivot;
        diagonal = pivot;

        for (int j = col + 1; j < count; ++j) {
            double dot = qr[col * count + j];
            for (int row = col + 1; row < rows; ++row) {
                dot += qr[row * count + col] * qr[row * count + j];
            }
            qr[col * count + j] -= tau * dot;
            for (int row = col + 1; row < rows; ++row) {
                qr[row * count + j] -= tau * dot * qr[row * count + col];
            }
        }

        double dot = rhs[col];
        for (int row = col + 1; row < rows; ++row) dot += qr[row * count + col] * rhs[row];
        rhs[col] -= tau * dot;
        for (int row = col + 1; row < rows; ++row) rhs[row] -= tau * dot * qr[row * count + col];
    }

    for (int row = count - 1; row >= 0; --row) {
        double value = rhs[row];
        for (int col = row + 1; col < count; ++col) {
            value -= qr[row * count + col] * candidate[active[col]];
        }
        candidate[active[row]] = value / qr[row * count + row];
    }
    return true;
}

extern "C" __global__ void fit_nnls(
    const double* spectra, const double* library, double* weights, int pixels,
    int rows, int materials, double rank_tolerance, double zero_scale, double missing,
    double* scratch, int* index_scratch, int* failed
) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;

    const double* spectrum = spectra + pixel * rows;
    double* result = weights + pixel * materials;
    double scale = 0;
    for (int row = 0; row < rows; ++row) {
        if (!isfinite(spectrum[row])) {
            for (int col = 0; col < materials; ++col) result[col] = missing;
            return;
        }
        scale = fmax(scale, fabs(spectrum[row]));
    }

    // All workspace is supplied per batch, so its size is independent of scene area.
    double* qr = scratch + (long long)pixel * (rows * materials + 2 * rows + materials);
    double* rhs = qr + rows * materials;
    double* residual = rhs + rows;
    double* candidate = residual + rows;
    int* active = index_scratch + (long long)pixel * 2 * materials;
    int* state = active + materials;  // 0 available, 1 active, 2 dependent
    for (int col = 0; col < materials; ++col) {
        result[col] = 0;
        state[col] = 0;
    }
    const double tolerance = 10 * rank_tolerance * scale;
    const double zero = zero_scale * scale;
    const long long max_steps = 30LL * materials * materials;
    long long steps = 0;
    int count = 0;

    while (steps++ < max_steps) {
        for (int row = 0; row < rows; ++row) {
            double predicted = 0;
            for (int i = 0; i < count; ++i) {
                const int col = active[i];
                predicted += library[row * materials + col] * result[col];
            }
            residual[row] = spectrum[row] - predicted;
        }
        // Add the available material with the largest positive residual gradient.
        int best = -1;
        double best_gradient = tolerance;
        for (int col = 0; col < materials; ++col) {
            if (state[col] != 0) continue;
            double gradient = 0;
            for (int row = 0; row < rows; ++row) {
                gradient += library[row * materials + col] * residual[row];
            }
            if (gradient > best_gradient) {
                best_gradient = gradient;
                best = col;
            }
        }
        if (best < 0) return;

        active[count++] = best;
        state[best] = 1;
        if (!solve_active(library, spectrum, rows, materials, active, count,
                          rank_tolerance, candidate, qr, rhs)) {
            --count;
            state[best] = 2;
            continue;
        }

        // If the unconstrained fit is negative, move to the first zero and refit.
        while (true) {
            int leaving = -1;
            double fraction = 1;
            for (int i = 0; i < count; ++i) {
                const int col = active[i];
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
                for (int col = 0; col < materials; ++col) result[col] = candidate[col];
                break;
            }
            if (steps++ >= max_steps) {
                atomicExch(failed, 1);
                return;
            }
            int kept = 0;
            for (int i = 0; i < count; ++i) {
                const int col = active[i];
                result[col] += fraction * (candidate[col] - result[col]);
                if (col == leaving || result[col] <= zero) {
                    result[col] = 0;
                    state[col] = 0;
                } else {
                    active[kept++] = col;
                }
            }
            count = kept;
            for (int col = 0; col < materials; ++col) {
                if (state[col] == 2) state[col] = 0;
            }
            if (!solve_active(library, spectrum, rows, materials, active, count,
                              rank_tolerance, candidate, qr, rhs)) {
                atomicExch(failed, 1);
                return;
            }
        }
    }
    atomicExch(failed, 1);
}
