import numpy as np
import sys
sys.path.append("../build/bindings/")
import atomic_ops_py as aop
from basic_tester import BasicTester

if __name__ == "__main__":
    tester = BasicTester(
        shape=[127, 9], 
        ranges=[1, 3],
        verbose = False
    )
    tester.test(
        atomic_func=aop.atomic_mul_u8,
        host_func=aop.mul_u8,
        dtype=np.uint8
    )

    tester.shape = [33, 63]
    tester.ranges = [-3, 4]

    tester.test(
        atomic_func=aop.atomic_mul_i16,
        host_func=aop.mul_i16,
        dtype=np.int16
    )