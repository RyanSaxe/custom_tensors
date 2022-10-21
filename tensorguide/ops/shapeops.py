from tensorguide.framework.ops import Operation
from tensorguide.ops import register_op_as_function


class ShapeOp(Operation):
    """Types of operations that never modify the data on a Tensor, and instead just modify things like the shape"""

    def __init__(self, tensor, copy=False):
        self.copy = copy
        super().__init__(
            tensor,
        )

    def _set_output_attrs(self):
        super()._set_output_attrs()
        self._output_tensor_kwargs.update(
            dict(
                contiguous=self.input._contiguous,
                offset=self.input._offset,
            )
        )

    def _forward(self, tensor):
        return tensor._flat_iterator if self.copy else tensor._storage


@register_op_as_function("reshape")
class Reshape(ShapeOp):
    def __init__(self, tensor, shape):
        super().__init__(tensor)
        if math.prod(shape) != math.prod(self.input.shape):
            raise ValueError(f"shape {shape} is invalid for tensor of size {math.prod(self.input.shape)}")
        self.copy = not tensor._contiguous
        self._output_tensor_kwargs["shape"] = shape

    def _backward(self):
        self.input.grad += reshape(self.output.grad, shape=self.input.shape)


@register_op_as_function("broadcast")
class Broadcast(ShapeOp):
    def __init__(self, tensor, shape):
        super().__init__(tensor, copy=False)
        stride = [
            0 if (self.input.shape[i] == 1) and (shape[i] != 1) else self.input._stride[i]
            for i in range(self.input.rank)
        ]
        if 0 not in stride:
            raise ValueError(f"Tensor object of shape {self.input.shape} does not need to be broadcasted to {shape}")
        self._output_tensor_kwargs.update(
            dict(
                stride=stride,
                contiguous=False,
                shape=shape,
            )
        )

    def _backward(self):
        # note to reduce sum the gradients over the broadcasted dimensions
        self.input.grad += reshape(self.output.grad, shape=self.input.shape)


@register_op_as_function("expand")
class Expand(ShapeOp):
    def __init__(self, tensor, dims):
        super().__init__(tensor, copy=False)
        dims = [self.input.rank + d if d < 0 else d for d in dims]
        shape = []
        axis = 0
        for i in range(len(dims) + self.input.rank):
            if i in dims:
                shape.append(1)
            else:
                shape.append(self.input.shape[axis])
                axis += 1
        self._output_tensor_kwargs.update(shape=shape)

    def _backward(self):
        # note to reduce sum the gradients over the broadcasted dimensions
        self.input.grad += squeeze(self.output.grad)


@register_op_as_function("squeeze")
class Squeeze(ShapeOp):
    def __init__(self, tensor):
        super().__init__(tensor, copy=False)
        shape, stride = list(
            zip(
                *(
                    (self.input.shape[i], self.input._stride[i])
                    for i in range(self.input.rank)
                    if self.input.shape[i] != 1
                )
            )
        )
        self.input_dims = [i for i in range(self.input.rank) if self.input.shape[i] == 1]
        self._output_tensor_kwargs.update(
            dict(
                shape=shape,
                stride=stride,
            )
        )

    def _backward(self):
        # note to reduce sum the gradients over the broadcasted dimensions
        self.input.grad += expand(self.output.grad, self.input_dims)
