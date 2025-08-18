#include "atomic_mul.cuh"
#include "atomic_ops.cuh"
#include "atomic_minmax.cuh"
#include "cuda_utils.cuh"

static constexpr bool TEST_SAFE_ALIGNED_KERNELS = true;

#define DEPLOY_ATOMIC(FuncName, ...)            \
    if constexpr (TEST_SAFE_ALIGNED_KERNELS) {  \
        FuncName##Safe(__VA_ARGS__);            \
    } else {                                    \
        FuncName(__VA_ARGS__);                  \
    }

namespace aop {

template <typename T, ReduceOp Op>
__global__ void atomicReduceKernel(T* arr, T* result, size_t n) {
    __shared__ T shared_val;
    
    if (threadIdx.x == 0) {
        if constexpr (Op == ReduceOp::MIN) {
            shared_val = getInitVal<T, true>();
        } else if constexpr (Op == ReduceOp::MAX) {
            shared_val = getInitVal<T, false>();
        } else if constexpr (Op == ReduceOp::MUL) {
            shared_val = static_cast<T>(1);
        } else {
            THROW_IN_GLOBAL("Unrecognized reduce operation enum: %d", static_cast<int>(Op));
        }
    }
    __syncthreads();
    
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        
        T val = arr[i];

        if constexpr (Op == ReduceOp::MIN) {
            DEPLOY_ATOMIC(atomicMin, &shared_val, val)
        } else if constexpr (Op == ReduceOp::MAX) {
            DEPLOY_ATOMIC(atomicMax, &shared_val, val)
        } else if constexpr (Op == ReduceOp::MUL) {
            DEPLOY_ATOMIC(atomicMul, &shared_val, val)
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        if constexpr (Op == ReduceOp::MIN) {
            DEPLOY_ATOMIC(atomicMin, result, shared_val)
        } else if constexpr (Op == ReduceOp::MAX) {
            DEPLOY_ATOMIC(atomicMax, result, shared_val)
        } else if constexpr (Op == ReduceOp::MUL) {
            DEPLOY_ATOMIC(atomicMul, result, shared_val)
        }
    }
}

template <typename T, ReduceOp Op>
T hostAtomicReduce(const T* host_arr, size_t n) {
    if (n == 0) return 0;
    
    T *d_arr, *d_result;
    CUDA_CHECK_RETURN(cudaMalloc(&d_arr, n * sizeof(T)));
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_result, sizeof(T)));
    
    CUDA_CHECK_RETURN(cudaMemcpy(d_arr, host_arr, n * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (Op == ReduceOp::MIN) {
        *d_result = getInitVal<T, true>();
    } else if constexpr (Op == ReduceOp::MAX) {
        *d_result = getInitVal<T, false>();
    } else if constexpr (Op == ReduceOp::MUL) {
        *d_result = static_cast<T>(1);
    }

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    atomicReduceKernel<T, Op><<<grid, block>>>(d_arr, d_result, n);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    T res_host = *d_result;
    CUDA_CHECK_RETURN(cudaFree(d_arr));
    CUDA_CHECK_RETURN(cudaFree(d_result));
    
    return res_host;
}

template uint8_t hostAtomicReduce<uint8_t, ReduceOp::MAX>(const uint8_t*, size_t);
template uint8_t hostAtomicReduce<uint8_t, ReduceOp::MIN>(const uint8_t*, size_t);
template uint8_t hostAtomicReduce<uint8_t, ReduceOp::MUL>(const uint8_t*, size_t);
template int16_t hostAtomicReduce<int16_t, ReduceOp::MAX>(const int16_t*, size_t);
template int16_t hostAtomicReduce<int16_t, ReduceOp::MIN>(const int16_t*, size_t);
template int16_t hostAtomicReduce<int16_t, ReduceOp::MUL>(const int16_t*, size_t);

}   // end namespace aop