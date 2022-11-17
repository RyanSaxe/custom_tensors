import math

from tensorguide.framework.ops import Operation, functionize_op

__all__ = ["log"]


class TransformOp(Operation):
    # A -> B such that A.shape = B.shape
    n_inputs = 1
    def __init__(self, tensor, **kwargs):
        super().__init__(tensor, **kwargs)

    @property
    def tensor(self):
        return self.inputs[0]

    def get_output_tensor_kwargs(self):
        return super().get_output_tensor_kwargs() | dict(
            shape=self.tensor.shape,
        )

    def inputs_iterator(self):
        return self.tensor._flat_iterator()


@functionize_op
class log(TransformOp):
    def __init__(self, tensor, base=math.e, **kwargs):
        self.base = base
        super().__init__(tensor, **kwargs)

    def forward(self, scalar):
        return math.log(scalar, self.base)

    def backward(self):
        self.tensor.grad += self.output.grad / self.tensor * math.log(self.base)


@functionize_op
class neg(TransformOp):
    def __init__(self, tensor, **kwargs):
        super().__init__(tensor, **kwargs)

    def forward(self, scalar):
        return -scalar

    def backward(self):
        self.tensor.grad -= self.output.grad


@functionize_op
class _abs(TransformOp):
    def __init__(self, tensor, **kwargs):
        super().__init__(tensor, **kwargs)

    def forward(self, scalar):
        return abs(scalar)

    def backward(self):
        # TODO: get where scaler was negative and have those grads signed flips
        ...
