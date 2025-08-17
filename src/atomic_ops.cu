#include "atomic_mul.cuh"
#include "atomic_ops.cuh"
#include "cuda_utils.cuh"

template <typename T>
__global__ void atomicMultiplyKernel(T* arr, T* result, size_t n) {
    __shared__ T shared_acc;
    
    if (threadIdx.x == 0) {
        shared_acc = static_cast<T>(1);
    }
    __syncthreads();
    
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        atomicMul(&shared_acc, arr[i]);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicMul(result, shared_acc);
    }
}

// Host函数模板实现
template <typename T>
T hostAtomicMultiply(const T* host_arr, size_t n) {
    if (n == 0) return 0;
    
    T *d_arr, *d_result;
    T h_result = static_cast<T>(1);
    
    CUDA_CHECK_RETURN(cudaMalloc(&d_arr, n * sizeof(T)));
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_result, sizeof(T)));
    
    CUDA_CHECK_RETURN(cudaMemcpy(d_arr, host_arr, n * sizeof(T), cudaMemcpyHostToDevice));
    *d_result = 1;
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    atomicMultiplyKernel<<<grid, block>>>(d_arr, d_result, n);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    T res_host = *d_result;
    CUDA_CHECK_RETURN(cudaFree(d_arr));
    CUDA_CHECK_RETURN(cudaFree(d_result));
    
    return res_host;
}

template uint8_t hostAtomicMultiply<uint8_t>(const uint8_t*, size_t);
template int16_t hostAtomicMultiply<int16_t>(const int16_t*, size_t);