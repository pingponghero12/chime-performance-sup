using Index = long long;

// Each thread interpolates one band of one spectrum.
template <class T, class Output>
__global__ void interpolate_spectra(
    const T* source, Output* output, Index size, Index bands, Index count,
    const Index* lower, const double* spacing, const double* left, const double* right
) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const Index band = i % count;
    const Index offset = i / count * bands + lower[band];
    const double first = source[offset];
    double value = first;
    // Exact band matches keep the original sample, including at either edge.
    if (left[band] != 0) {
        const double second = source[offset + 1];
        const double slope = (second - first) / spacing[band];
        value = first + slope * left[band];
        if (isnan(value)) {
            value = second + slope * right[band];
            if (isnan(value) && first == second) value = first;
        }
    }
    output[i] = value;
}

// Each thread calculates one sample along the current spatial axis.
template <class T, class Output>
__global__ void interpolate_axis(
    const T* source, Output* output, Index size, Index width, Index bands,
    Index output_width, int axis, const Index* lower, const double* spacing,
    const double* distance
) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const Index x = i / bands / output_width;
    const Index y = i / bands % output_width;
    const Index coordinate = axis == 0 ? x : y;
    const Index lo = lower[coordinate];
    const Index stride = axis == 0 ? width * bands : bands;
    const Index offset = (axis == 0 ? lo * width + y : x * width + lo) * bands + i % bands;
    const T first = source[offset];
    const T second = source[offset + stride];
    output[i] = first + (second - first) / spacing[coordinate] * distance[coordinate];
}

// Each thread combines the positive weights for one detector channel.
template <class T, class Output>
__global__ void integrate_spectra(
    const T* source, Output* output, Index size, Index bands, Index channels,
    const Index* columns, const double* weights, const Index* first, const Index* stop
) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const Index pixel = i / channels;
    const Index row = columns[pixel] * channels + i % channels;
    const double* response = weights + row * bands;
    const T* spectrum = source + pixel * bands;
    double sum = 0;
    for (Index b = first[row]; b < stop[row]; ++b) {
        // A missing sample outside this channel's support contributes nothing.
        if (response[b] > 0) sum += spectrum[b] * response[b];
    }
    output[i] = sum;
}

// Only convolve at the retained ACT and ALT centres. Repeat the edge pixels.
template <class T, class Output>
__global__ void convolve_centres(
    const T* source, Output* output, Index size, Index height, Index width,
    Index bands, Index output_width, Index factor_x, Index factor_y,
    const double* kernel, Index rows, Index columns
) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const Index band = i % bands;
    const Index x = i / bands / output_width * factor_x;
    const Index y = i / bands % output_width * factor_y;
    double sum = 0;
    for (Index row = 0; row < rows; ++row) {
        const Index u = min(max(x + row - rows / 2, Index(0)), height - 1);
        for (Index col = 0; col < columns; ++col) {
            const Index v = min(max(y + col - columns / 2, Index(0)), width - 1);
            const double weight = kernel[(rows - 1 - row) * columns + columns - 1 - col];
            sum += source[(u * width + v) * bands + band] * weight;
        }
    }
    output[i] = sum;
}
