#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "atomic_ops.cuh"

namespace nb = nanobind;

#define DEFINE_OP(OpName, PyOpName, Type, TypeStr)                                              \
    m.def("atomic_"#PyOpName"_"#TypeStr, [](nb::ndarray<const Type> arr) {                      \
        return aop::hostAtomicReduce<Type, aop::ReduceOp::OpName>(arr.data(), arr.size());      \
    }, nb::arg("array"), "Atomic "#OpName" for "#Type" arrays (ignores overflow)");             \
    m.def(#PyOpName"_"#TypeStr, [](nb::ndarray<const Type> arr) {                               \
        return aop::hostReferenceReduce<Type, aop::ReduceOp::OpName>(arr.data(), arr.size());   \
    }, nb::arg("array"), "Host serial "#OpName" for "#Type" arrays (ignores overflow)");

NB_MODULE(atomic_ops_py, m) {
    DEFINE_OP(MUL, mul, uint8_t, u8)
    DEFINE_OP(MAX, max, uint8_t, u8)
    DEFINE_OP(MIN, min, uint8_t, u8)

    DEFINE_OP(MUL, mul, int16_t, i16)
    DEFINE_OP(MAX, max, int16_t, i16)
    DEFINE_OP(MIN, min, int16_t, i16)
}