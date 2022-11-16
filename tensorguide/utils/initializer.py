import array
import math
import random
from functools import partial


class Initializer:
    def __call__(self, shape, dtype):
        # can parallelize later
        size = math.prod(shape)
        generator = (self._generate_value() for _ in range(size))
        return array.array(dtype, generator)

    def _generate_value(self):
        raise NotImplementedError()


class ConstantInitializer(Initializer):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def _generate_value(self):
        # NOTE: this may have an issue by pointing to same constant in memory?
        return self.c


class UniformInitializer(Initializer):
    def __init__(self, a=0, b=1):
        super().__init__()
        self.a = a
        self.b = b

    def _generate_value(self):
        return random.uniform(self.a, self.b)


OnesInitializer = partial(ConstantInitializer, c=1)
ZerosInitializer = partial(ConstantInitializer, c=0)
