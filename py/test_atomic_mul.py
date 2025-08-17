import numpy as np
import sys
sys.path.append("../build/bindings/")
import atomic_ops_py as aop

def possible_overflow_mul_test(
    atomic_func,
    host_func,
    dtype,
    shape_2d=[16, 16], 
    min_max=[1, 6],
    verbose=False):
    values: np.ndarray = np.random.randint(*min_max, shape_2d).astype(dtype)
    if min_max[0] < 1:
        is_zero = values == 0
        values[is_zero] = 1
    if verbose:
        print(values)
    # host serial accumulation (with mul)
    true_result = host_func(values)
    print(f"CUDA atomic Mul for {dtype} starts...")
    test_result = atomic_func(values)
    print(f"CUDA atomic Mul for {dtype} returns.")
    if test_result != true_result:
        print(f"Error: not equal, test = {test_result}, expect = {true_result}")
    else:
        print(f"Comparison Passed: result = {true_result}, shape: {shape_2d}")

if __name__ == "__main__":
    possible_overflow_mul_test(
        atomic_func=aop.atomic_mul_u8,
        host_func=aop.mul_u8,
        dtype=np.uint8,
        shape_2d=[4, 4], 
        min_max=[1, 3],
        verbose = True)

    possible_overflow_mul_test(
        atomic_func=aop.atomic_mul_i16,
        host_func=aop.mul_i16,
        dtype=np.int16,
        shape_2d=[4, 4], 
        min_max=[-3, 4],
        verbose = True)
