# Atomics
---

The testing infra for improving the existing PaddlePaddle atomic primitives (like atomicMul for different data types) and creating new primitives 
(like `atomicMul` for `uint8_t` and `int16_t`). This repo depends on `numpy` and `nanobind` (as git submodule).

To test the code, please simply clone this repo and build with CMake (CUDA needed), Python API will be exported (in `build/bindings/atomic_ops_py`). 
You can head to `py` and use the `BasicTester` to check the comparison results out. The comparison is derived from comparing CUDA results with `std::accumulation` (serial), 
which also has been exported to Python end by nanobind. 
