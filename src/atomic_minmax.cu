#include "atomic_minmax.cuh"

namespace aop {

#define DEFINE_ATOMIC_MINMAX(Dtype, Mask, OpType, operator)                             \
__device__ Dtype atomic##OpType(Dtype* address, Dtype val) {                            \
    uint32_t* ptr32 = reinterpret_cast<uint32_t*>(address);                             \
    const uint32_t shift = (reinterpret_cast<size_t>(address) % 4) * 8;                 \
    const uint32_t mask = Mask << shift;                                                \
    uint32_t old32 = *ptr32, assumed32 = 0;                                             \
    Dtype current = 0, new_val = 0;                                                     \
    do {                                                                                \
        assumed32 = old32;                                                              \
        current = static_cast<Dtype>((old32 & mask) >> shift);                          \
        new_val = operator(current, val);                                               \
        uint32_t new32 = (old32 & ~mask) | (static_cast<uint32_t>(new_val) << shift);   \
        old32 = atomicCAS(ptr32, assumed32, new32);                                     \
    } while (assumed32 != old32);                                                       \
    return static_cast<Dtype>((old32 & mask) >> shift);                                 \
}

DEFINE_ATOMIC_MINMAX(int16_t, 0xFFFFU, Min, min)
DEFINE_ATOMIC_MINMAX(int16_t, 0xFFFFU, Max, max)
DEFINE_ATOMIC_MINMAX(uint8_t, 0xFFU, Min, min)
DEFINE_ATOMIC_MINMAX(uint8_t, 0xFFU, Max, max)

#undef DEFINE_ATOMIC_MINMAX

}   // end namespace aop