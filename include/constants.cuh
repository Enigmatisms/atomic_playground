#pragma once

#include <cstdint>

namespace aop {

inline constexpr int16_t I16_MAX = std::numeric_limits<int16_t>::max();
inline constexpr int16_t I16_MIN = std::numeric_limits<int16_t>::min();

inline constexpr uint8_t U8_MAX = std::numeric_limits<uint8_t>::max();
inline constexpr uint8_t U8_MIN = std::numeric_limits<uint8_t>::min();

template <typename T, bool is_min>
__host__ __device__ inline constexpr T getInitVal() {
    if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (is_min) return I16_MAX;
        else return I16_MIN;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (is_min) return U8_MAX;
        else return U8_MIN;
    } else {
        return static_cast<T>(0);
    }
}

}   // end namespace aop
