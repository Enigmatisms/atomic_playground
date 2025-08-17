#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "atomic_ops.cuh"

namespace nb = nanobind;

#define DEFINE_OP(OpName, PyOpName, Type, TypeStr)                                      \
    m.def("atomic_"#PyOpName"_"#TypeStr, [](nb::ndarray<const Type> arr) {                 \
        return hostAtomic##OpName(arr.data(), arr.size());                              \
    }, nb::arg("array"), "Atomic "#OpName" for "#Type" arrays (ignores overflow)");     \
    m.def(#PyOpName"_"#TypeStr, [](nb::ndarray<const Type> arr) {                         \
        return hostReference##OpName(arr.data(), arr.size());                           \
    }, nb::arg("array"), "Host serial "#OpName" for "#Type" arrays (ignores overflow)");

NB_MODULE(atomic_ops_py, m) {
    DEFINE_OP(Multiply, mul, uint8_t, u8)
    DEFINE_OP(Multiply, mul, int16_t, i16)

    // m.def("atomic_mul_u8", [](nb::ndarray<const uint8_t> arr) {
    //     return hostAtomicMultiply(arr.data(), arr.size());
    // }, nb::arg("array"), "Atomic multiply for uint8 arrays (ignores overflow)");

    // m.def("mul_u8", [](nb::ndarray<const uint8_t> arr) {
    //     return hostReferenceMultiply(arr.data(), arr.size());
    // }, nb::arg("array"), "Host serial multiply for uint8 arrays (ignores overflow)");
    
    // m.def("atomic_mul_i16", [](nb::ndarray<const int16_t> arr) {
    //     return hostAtomicMultiply(arr.data(), arr.size());
    // }, nb::arg("array"), "Atomic multiply for int16 arrays (ignores overflow)");

    // m.def("mul_u8", [](nb::ndarray<const uint8_t> arr) {
    //     return hostReferenceMultiply(arr.data(), arr.size());
    // }, nb::arg("array"), "Host serial multiply for uint8 arrays (ignores overflow)");
}