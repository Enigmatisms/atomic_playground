#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace aop {

// load and store aligned 4B positions for int16 and uint8, avoiding reading in the following way:
// (misaligned) uint8_t* address (0x03) -> auto to_read = reinterpret_cast<uint32_t*>(address);

__device__ __forceinline__ uint32_t loadAligned(const uintptr_t base_addr, uint32_t mask, uint32_t shift) {
    // get 4B aligned address
    uint32_t aligned_value = *reinterpret_cast<const uint32_t*>(base_addr);
    return (aligned_value & mask) >> shift;
}

// ================= nightly, to test in the future ==================

template <typename T>
__device__ __forceinline__ uint32_t safeLoadU32(const T* address, size_t array_size, size_t index) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2, "Only uint8_t and int16_t are supported");
    
    const uint8_t* base_ptr = reinterpret_cast<const uint8_t*>(address);
    const uint8_t* current_ptr = base_ptr + index * sizeof(T);
    const uint8_t* end_ptr = base_ptr + array_size * sizeof(T);
    
    if (current_ptr + sizeof(uint32_t) <= end_ptr) {
        return *reinterpret_cast<const uint32_t*>(current_ptr);
    }

    const size_t bytes_to_read = end_ptr - current_ptr;
    uint32_t value = 0;
    for (int i = 0; i < bytes_to_read; ++i) {
        value |= (static_cast<uint32_t>(current_ptr[i]) << (i * 8));
    }
    return value;
}

template <typename T>
__device__ __forceinline__ void safeStoreU32(T* address, uint32_t value, size_t array_size, size_t index) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2, "Only uint8_t and int16_t are supported");
    
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(address);
    uint8_t* current_ptr = base_ptr + index * sizeof(T);
    uint8_t* end_ptr = base_ptr + array_size * sizeof(T);
    
    if (current_ptr + sizeof(uint32_t) <= end_ptr) {
        *reinterpret_cast<uint32_t*>(current_ptr) = value;
        return;
    }

    const size_t bytes_to_read = end_ptr - current_ptr;
    for (int i = 0; i < bytes_to_read; ++i) {
        current_ptr[i] = static_cast<uint8_t>(value >> (i * 8));
    }
}


}   // end namespace aop