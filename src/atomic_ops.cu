#include "atomic_mul.cuh"
#include "atomic_ops.cuh"

// CUDA全局函数模板实现
template <typename T>
__global__ void atomicMultiplyKernel(T* arr, T* result, size_t n) {
    // 使用共享内存作为临时累加器
    __shared__ T shared_acc;
    
    if (threadIdx.x == 0) {
        shared_acc = static_cast<T>(1); // 乘法初始值为1
    }
    __syncthreads();
    
    // 每个线程处理一个元素
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        // 原子乘法（忽略溢出）
        atomicMul(&shared_acc, arr[i]);
    }
    __syncthreads();
    
    // 块内第一个线程更新全局结果
    if (threadIdx.x == 0) {
        atomicMul(result, shared_acc);
    }
}

// Host函数模板实现
template <typename T>
T hostAtomicMultiply(const T* host_arr, size_t n) {
    if (n == 0) return 0;
    
    T *d_arr, *d_result;
    T h_result = static_cast<T>(1); // 初始乘积为1
    
    // 设备内存分配
    cudaMalloc(&d_arr, n * sizeof(T));
    cudaMalloc(&d_result, sizeof(T));
    
    // 数据复制到设备
    cudaMemcpy(d_arr, host_arr, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(T), cudaMemcpyHostToDevice);
    
    // 计算内核配置
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    // 启动内核
    atomicMultiplyKernel<<<grid, block>>>(d_arr, d_result, n);
    cudaDeviceSynchronize();
    
    // 获取结果
    cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    
    // 清理
    cudaFree(d_arr);
    cudaFree(d_result);
    
    return h_result;
}

template uint8_t hostAtomicMultiply<uint8_t>(const uint8_t*, size_t);
template int16_t hostAtomicMultiply<int16_t>(const int16_t*, size_t);