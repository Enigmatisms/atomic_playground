#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "atomic_ops.cuh"

namespace nb = nanobind;

NB_MODULE(atomic_ops_py, m) {
    m.def("multiply_uint8", [](nb::ndarray<const uint8_t> arr) {
        if (arr.ndim() != 1) {
            throw std::runtime_error("Input must be 1-dimensional");
        }
        return hostAtomicMultiply(arr.data(), arr.size());
    }, nb::arg("array"), "Atomic multiply for uint8 arrays (ignores overflow)");
    
    m.def("multiply_int16", [](nb::ndarray<const int16_t> arr) {
        if (arr.ndim() != 1) {
            throw std::runtime_error("Input must be 1-dimensional");
        }
        return hostAtomicMultiply(arr.data(), arr.size());
    }, nb::arg("array"), "Atomic multiply for int16 arrays (ignores overflow)");
}