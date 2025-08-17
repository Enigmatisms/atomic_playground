#include "atomic_mul.cuh"

namespace aop {

__device__ int16_t atomicMul(int16_t* address, int16_t val) {
    uint32_t* ptr32 = reinterpret_cast<uint32_t*>(address);
    uint32_t shift = (reinterpret_cast<size_t>(address) % 4) * 8;
    uint32_t mask = 0xFFFFU << shift;

    uint32_t old32 = *ptr32, assumed32;
    do {
        assumed32 = old32;
        int16_t current = static_cast<int16_t>((old32 & mask) >> shift);
        int16_t new_val = current * val;
        uint32_t new32 = (old32 & ~mask) | (static_cast<uint32_t>(new_val) << shift);
        old32 = atomicCAS(ptr32, assumed32, new32);
    } while (assumed32 != old32);
    
    return static_cast<int16_t>((old32 & mask) >> shift);
}

__device__ uint8_t atomicMul(uint8_t* address, uint8_t val) {
    uint32_t* ptr32 = reinterpret_cast<uint32_t*>(address);
    uint32_t shift = (reinterpret_cast<size_t>(address) % 4) * 8;
    uint32_t mask = 0xFFU << shift;

    uint32_t old32 = *ptr32, assumed32;
    do {
        assumed32 = old32;
        uint8_t current = static_cast<uint8_t>((old32 & mask) >> shift);
        uint8_t new_val = current * val;
        uint32_t new32 = (old32 & ~mask) | (static_cast<uint32_t>(new_val) << shift);
        old32 = atomicCAS(ptr32, assumed32, new32);
    } while (assumed32 != old32);
    
    return static_cast<uint8_t>((old32 & mask) >> shift);
}

}   // end namespace aop