import random
from functools import partial
from typing import Callable


class Initializer:
    # QUESTION: what does this initializer not allow that we might need?
    def __init__(self, func: Callable, dtype: Callable):
        super().__init__()
        self.func = func
        self.dtype = dtype

    def __call__(self, size):
        # can parallelize later
        output = []
        for _ in range(size):
            value = self.func()
            value = value if isinstance(value, self.dtype) else self.dtype(value)
            output.append(value)
        return tuple(output)


class ConstantInitializer(Initializer):
    def __init__(self, c, dtype):
        func = lambda: c
        super().__init__(func, dtype)


class UniformInitializer(Initializer):
    def __init__(self, a=0, b=1, dtype=float):
        func = lambda: random.uniform(a, b)
        super().__init__(func, dtype)


OnesInitializer = partial(ConstantInitializer, c=1)
ZerosInitializer = partial(ConstantInitializer, c=0)
