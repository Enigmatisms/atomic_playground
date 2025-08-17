#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <numeric>

enum class DataType {
    UINT8,
    INT16
};

template <typename T>
__global__ void atomicMultiplyKernel(T* arr, T* result, size_t n);

template <typename T>
T hostAtomicMultiply(const T* host_arr, size_t n);

template <typename T>
T hostReferenceMultiply(const T* host_arr, size_t n) {
    return std::accumulate(host_arr, host_arr + n, T(1), [](const T v1, const T v2) { return v1 * v2; });
}