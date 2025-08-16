#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// 支持的数据类型
enum class DataType {
    UINT8,
    INT16
};

// 全局函数模板声明
template <typename T>
__global__ void atomicMultiplyKernel(T* arr, T* result, size_t n);

// Host封装函数
template <typename T>
T hostAtomicMultiply(const T* host_arr, size_t n);
