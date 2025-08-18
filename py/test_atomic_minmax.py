import numpy as np
import sys
sys.path.append("../build/bindings/")
import atomic_ops_py as aop
from basic_tester import BasicTester


if __name__ == "__main__":
    tester = BasicTester(
        shape=[3, 3], 
        ranges=[5, 256],
        verbose = False
    )
    for i in range(30):
        print(f"Test for uint8: {i + 1:2d}: ")
        tester.test(
            atomic_func=aop.atomic_min_u8,
            host_func=aop.min_u8,
            dtype=np.uint8
        )

        tester.test(
            atomic_func=aop.atomic_max_u8,
            host_func=aop.max_u8,
            dtype=np.uint8
        )

    tester.shape = [3, 3]
    tester.ranges = [-32760, 32761]

    for i in range(30):
        print(f"Test for int16: {i + 1:2d}: ")

        tester.test(
            atomic_func=aop.atomic_min_i16,
            host_func=aop.min_i16,
            dtype=np.int16
        )
        tester.test(
            atomic_func=aop.atomic_max_i16,
            host_func=aop.max_i16,
            dtype=np.int16
        )

    np.random.seed(0)
    tester.shape = [4, 4, 4]
    for i in range(30):
        print(f"Test for int16, new shape: {i + 1:2d}: ")

        tester.test(
            atomic_func=aop.atomic_min_i16,
            host_func=aop.min_i16,
            dtype=np.int16
        )
        tester.test(
            atomic_func=aop.atomic_max_i16,
            host_func=aop.max_i16,
            dtype=np.int16
        )
