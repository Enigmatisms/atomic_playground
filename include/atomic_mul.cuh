#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace aop {

__device__ int16_t atomicMul(int16_t* address, int16_t val);

__device__ uint8_t atomicMul(uint8_t* address, uint8_t val);

}   // end namespace aop