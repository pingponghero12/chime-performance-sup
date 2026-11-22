#pragma once

#include <pybind11/numpy.h>

namespace chime {

namespace py = pybind11;
using Index = py::ssize_t;

template <class T>
using Array = py::array_t<T, py::array::c_style>;

void bind_sampling(py::module_& module);
void bind_nnls(py::module_& module);
void bind_radiometry(py::module_& module);

}  // namespace chime
