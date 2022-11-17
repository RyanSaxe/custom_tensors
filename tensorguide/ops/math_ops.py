import math

from tensorguide.framework.ops import Operation, functionize_op

__all__ = ["add", "multiply", "subtract", "power", "divide"]


class MathOp(Operation):
    def transform_and_check_input_tensors(self):
        # from tensorguide.ops.array_ops import broadcast, expand

        tensors = super().transform_and_check_input_tensors()
        out_rank = max(t.rank for t in tensors)
        expanded_shapes = [(*([1] * (out_rank - t.rank)), *t.shape) for t in tensors]
        out_shape = []
        for i, axis_size in enumerate(zip(*expanded_shapes)):
            sizes = set(axis_size)
            if len(sizes) - (1 in sizes) > 1:
                raise ValueError(
                    f"{str(self)} Operation cannot be performed across tensors where the {i}th axis has sizes {axis_size}"
                )
            out_shape.append(max(axis_size))

        output_tensors = []
        for tensor in tensors:
            if tensor.shape != out_shape:
                tensor = tensor.broadcast_to(shape=out_shape)
            output_tensors.append(tensor)
        return output_tensors

    def get_output_tensor_kwargs(self):
        return super().get_output_tensor_kwargs() | dict(shape=self.inputs[0].shape)

    def inputs_iterator(self):
        return zip(*[tensor._flat_iterator() for tensor in self.inputs])


@functionize_op
class add(MathOp):
    min_n_inputs = 2

    def forward(self, inputs):
        return sum(inputs)

    def backward(self):
        for tensor in self.inputs:
            tensor.grad += self.output.grad


@functionize_op
class multiply(MathOp):
    min_n_inputs = 2

    def forward(self, inputs):
        return math.prod(inputs)

    def backward(self, inputs):
        for tensor in self.inputs:
            tensor.grad += self.output.grad * self.output / tensor


@functionize_op
class subtract(MathOp):
    n_inputs = 2

    def forward(self, inputs):
        a, b = inputs
        return a - b

    def backward(self):
        a, b = self.inputs
        a.grad += self.output.grad
        b.grad -= self.output.grad


@functionize_op
class power(MathOp):
    n_inputs = 2

    def forward(self, inputs):
        number, exponent = inputs
        return number**exponent

    def backward(self):
        number, exponent = self.inputs
        number.grad = self.output.grad * exponent * number ** (exponent - 1)


@functionize_op
class divide(MathOp):
    n_inputs = 2

    def forward(self, inputs):
        numerator, denominator = inputs
        return numerator / denominator

    def backward(self):
        numerator, denominator = self.inputs
        numerator.grad += output.grad / denominator
        deniminator.grad += output.grad * -(numerator / denominator**2)


@functionize_op
class matmul(Operation):
    n_inputs = 2

    def transform_and_check_input_tensors(self):
        a, b = super().transform_and_check_input_tensors()
        if a.rank < 2 or b.rank < 2:
            raise ValueError("cannot call matmul on a Tensor with rank less than 2")
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(
                "The second to last dimension of the first tensor in a matmul must be equal to the last dimension of the other tensor"
            )

        # we want to make sure we broadcast the tensors to have the same dimensions
        rank_diff = a.rank - b.rank
        a_shape, b_shape = a.shape[:-2], b.shape[:-2]
        if rank_diff > 0:
            b_shape = (*([1] * rank_diff), *b_shape[:-2])
        elif rank_diff < 0:
            a_shape = (*([1] * rank_diff), *a_shape[:-2])

        broadcast_dims = []
        for a_ax, b_ax in zip(a_shape, b_shape):
            if (a_ax != b_ax) and (a_ax != 1) and (b_ax != 1):
                raise ValueError(f"Cannot compute A @ B for tensors of shapes {a.shape} and {b.shape}.")
            broadcast_dims.append(max(a_ax, b_ax))

        a_shape = (*broadcast_dims, *a.shape[-2:])
        b_shape = (*broadcast_dims, *b.shape[-2:])
        if a.shape != a_shape:
            a = a.broadcast_to(a_shape)
        if b.shape != b_shape:
            b = b.broadcast_to(b_shape)

        return [a, b]

    def get_output_tensor_kwargs(self):
        a, b = self.inputs
        shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
        return super().get_output_tensor_kwargs() | dict(shape=shape)

    def inputs_iterator(self):
        # because the forward function is triggering sub operation on the direct input
        # tensors, we don't need to specify the way that we iterate over them
        yield self.inputs

    def forward(self, inputs):
        # NOTE: can make this significantly more efficient
        a, b = inputs
        # a.shape = (...,m,n)
        # b.shape = (...,n,k)
        # we need to broadcast them to shape (...,m,n,k) to do the operation
        # and we do this in forward instead of the initial transform such that we can
        # ignore some operations to compute the gradients directly.
        broadcasted_shape = (*a.shape, b.shape[-1])
        a = a.expand(dims=(-1,)).broadcast_to(shape=broadcasted_shape)
        b = b.expand(dims=(-3,)).broadcast_to(shape=broadcasted_shape)
        # NOTE: this could use an efficiency improvement
        return (a * b).sum(axis=-2)

    def backward(self):
        a, b = self.inputs
        a.grad = self.output.grad @ b.swapaxes(-1, -2)
        b.grad = a.swapaxes(-1, -2) @ self.output.grad
