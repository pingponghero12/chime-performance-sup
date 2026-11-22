#include "kernels.hpp"

PYBIND11_MODULE(_cpp, module) {
    module.doc() = "CHIME image kernels using OpenMP.";
    chime::bind_sampling(module);
    chime::bind_nnls(module);
    chime::bind_radiometry(module);
}
