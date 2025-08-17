#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <numeric>
#include "constants.cuh"
#include "cuda_utils.cuh"

namespace aop {

enum class ReduceOp { MIN, MAX, MUL };

template <typename T, ReduceOp Op>
T hostAtomicReduce(const T* host_arr, size_t n);

template <typename T, ReduceOp Op>
T hostReferenceReduce(const T* host_arr, size_t n) {
    T init_v = static_cast<T>(1);
    if constexpr (Op == ReduceOp::MIN) {
        init_v = getInitVal<T, true>();
        return std::accumulate(host_arr, host_arr + n, init_v, [](const T v1, const T v2) { return std::min(v1, v2); });
    } else if constexpr (Op == ReduceOp::MAX) {
        init_v = getInitVal<T, false>();
        return std::accumulate(host_arr, host_arr + n, init_v, [](const T v1, const T v2) { return std::max(v1, v2); });
    } else if constexpr (Op == ReduceOp::MUL) {
        return std::accumulate(host_arr, host_arr + n, init_v, [](const T v1, const T v2) { return v1 * v2; });
    } else {
        THROW_IN_HOST("Unrecognized reduce operation enum: %d", static_cast<int>(Op))
    }
}

}   // end namespace aop
