from tensorguide.framework.ops import Operation, functionize_op
from tensorguide.ops import math_ops

__all__ = ["expand"]


class ArrayOp(Operation):
    """Types of operations that never modify the data on a Tensor, and instead just modify things like the shape"""

    n_inputs = 1

    def __init__(self, tensor, **kwargs):
        super().__init__(tensor, njobs=1, **kwargs)

    @property
    def tensor(self):
        return self.inputs[0]

    def get_output_tensor_kwargs(self):
        return super().get_output_tensor_kwargs() | dict(
            _contiguous=self.tensor._contiguous,
            _offset=self.tensor._offset,
        )

    def inputs_iterator(self):
        return self.tensor._value

    def forward(self, tensor):
        # the only modification of these operations are in the attributes of the tensor
        # which are defined in `_get_output_tensor_kwargs`, hence there is no need for `forward`
        return tensor


@functionize_op
class _slice(ArrayOp):
    def __init__(self, tensor, key, **kwargs):
        self.key = key if isinstance(key, tuple) else (key,)
        super().__init__(tensor, **kwargs)

    def get_output_tensor_kwargs(self):
        if self.tensor.rank == 0:
            raise ValueError("Cannot index into a Tensor with 0 Dimension (a Scalar)")
        shape = []
        stride = []
        offset = self.tensor._offset
        axis = 0
        contiguous = True
        for axis_slice in self.key:
            axis_size = self.tensor.shape[axis]
            if axis_slice is None:
                shape.append(1)
                # stride number shouldn't really matter here, but this is what pytorch does
                if axis == 0:
                    stride.append(len(self.tensor._value))  # are you supposed to subtract offset here?
                else:
                    stride.append(self.tensor._stride[axis - 1])
                continue
            if isinstance(axis_slice, int):
                if axis_slice >= axis_size or axis_slice < -axis_size:
                    raise IndexError(f"index {axis_slice} is out of bounds for axis {axis} with size {axis_size}")
                if axis_slice < 0:
                    axis_slice = axis_size + axis_slice
                offset += axis_slice * self.tensor._stride[axis]
            else:
                step = 1 if axis_slice.step is None else axis_slice.step
                if step == 0 or not isinstance(step, int):
                    raise ValueError(f"{step} is an invalid input for slice.step. Must be a non-zero integer")

                if axis_slice.start is None:
                    if step < 0:
                        start = axis_size - 1
                    else:
                        start = 0
                else:
                    start = axis_slice.start if axis_slice.start >= 0 else axis_size + axis_slice.start

                    if start < 0 or start > axis_size:
                        raise IndexError(f"index {axis_slice} is out of bounds for axis {axis} with size {axis_size}")
                if axis_slice.stop is None:
                    if step < 0:
                        stop = -1
                    else:
                        stop = axis_size
                else:
                    stop = axis_slice.stop if axis_slice.stop >= 0 else axis_size + axis_slice.stop

                    if stop > axis_size or stop < 0:
                        raise IndexError(f"index {axis_slice} is out of bounds for axis {axis} with size {axis_size}")

                length = stop - start
                if (length * step) < 0:
                    raise ValueError("the stop of the slice must be greater than the start of the slice")
                shape.append(-(length // -step))
                stride.append(self.tensor._stride[axis] * step)
                if step != 1:
                    contiguous = False
                offset += start * self.tensor._stride[axis]
            axis += 1

        shape = (*shape, *self.tensor.shape[axis:])
        stride = (*stride, *self.tensor._stride[axis:])
        return super().get_output_tensor_kwargs() | dict(
            shape=shape,
            _offset=offset,
            _stride=stride,
            _contiguous=contiguous,
        )


@functionize_op
class broadcast(ArrayOp):
    def __init__(self, tensor, shape, **kwargs):
        self.shape = shape
        super().__init__(tensor, **kwargs)

    def get_output_tensor_kwargs(self):
        n_expands = len(self.shape) - len(self.tensor.shape)
        if n_expands < 0:
            raise ValueError("cannot broadcast from a tensor of higher rank to a tensor of lower rank")
        shape = (*([1] * n_expands), *self.tensor.shape)
        stride = [0 if self.shape[i] != 1 else self.tensor._stride[0] for i in range(n_expands)] + [
            0 if (shape[i] == 1) and (self.shape[i] != 1) else self.tensor._stride[i - n_expands]
            for i in range(n_expands, self.tensor.rank + n_expands)
        ]
        if 0 not in stride:
            raise ValueError(f"Tensor object of shape {self.tensor.shape} does not need to be broadcasted to {shape}")

        return super().get_output_tensor_kwargs() | dict(
            _stride=stride,
            _contiguous=False,
            shape=self.shape,
        )

    def backward(self):
        broadcasted_axes = [ax for ax in range(self.output.rank) if self.output._stride[ax] == 0]
        grad = self.output.grad
        for ax_to_aggregate in broadcasted_axes[::-1]:
            grad = grad.sum(axis=ax_to_aggregate)
        self.tensor.grad += grad.reshape(self.tensor.shape)


@functionize_op
class expand(ArrayOp):
    def __init__(self, tensor, dims, **kwargs):
        self.dims = dims
        super().__init__(tensor, **kwargs)

    def get_output_tensor_kwargs(self):
        dims = [self.tensor.rank + d + len(self.dims) if d < 0 else d for d in self.dims]
        shape = []
        stride = []
        waiting_on_stride = 0
        axis = 0
        for i in range(len(dims) + self.tensor.rank):
            if i in dims:
                shape.append(1)
                waiting_on_stride += 1
            else:
                shape.append(self.tensor.shape[axis])
                for _ in range(waiting_on_stride + 1):
                    stride.append(self.tensor._stride[axis])
                waiting_on_stride = 0
                axis += 1
        return super().get_output_tensor_kwargs() | dict(shape=shape, _stride=stride)


@functionize_op
class _sum(ArrayOp):
    def __init__(self, tensor, axis=0, keepdims=False, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(tensor, **kwargs)

    def get_output_tensor_kwargs(self):
        self.axis = self.axis if self.axis >= 0 else self.tensor.rank + self.axis
        shape = [*self.tensor.shape[: self.axis], *self.tensor.shape[self.axis + 1 :]]
        if self.keepdims:
            shape.insert(axis, 1)
        return super().get_output_tensor_kwargs() | dict(shape=shape)

    def inputs_iterator(self):
        yield [
            _slice(self.tensor, key=tuple([slice(None)] * self.axis + [idx]))
            for idx in range(self.tensor.shape[self.axis])
        ]

    def forward(self, inputs):
        return math_ops.add(*inputs)

    def backward(self):
        # need to do axis expansion first
        tensor.grad += self.output.grad.broadcast_to(self.tensor.shape)
