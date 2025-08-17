import numpy as np

class BasicTester:
    def __init__(self, shape=[16, 16], ranges=[1, 6], verbose=False):
        self.shape = shape
        self.ranges = ranges
        self.verbose = verbose

    def log(self, fmt, *args, **kwargs):
        if not self.verbose: return
        print(fmt, *args, **kwargs)

    def test(
        self,
        atomic_func,
        host_func,
        dtype
    ):
        values: np.ndarray = np.random.randint(*self.ranges, self.shape).astype(dtype)
        if self.ranges[0] < 1:
            is_zero = values == 0
            values[is_zero] = 1
        self.log(values)

        true_result = host_func(values)
        self.log(f"CUDA atomic func for {dtype} starts...")
        test_result = atomic_func(values)
        self.log(f"CUDA atomic func for {dtype} returns.")
        if test_result != true_result:
            print(f"Error: not equal, test = {test_result}, expect = {dtype}, shape: {self.shape}")
        else:
            print(f"Comparison Passed: result = {test_result}, shape: {self.shape}")
