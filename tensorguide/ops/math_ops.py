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
                    f"{self.__name__} Operation cannot be performed across tensors where the {i}th axis has sizes {axis_size}"
                )
            out_shape.append(max(axis_size))

        output_tensors = []
        for tensor in tensors:
            # if the desired output tensor and current tensor are of different ranks, expand current tensor
            if tensor.rank != out_rank:
                tensor = tensor.expand(dims=list(range(out_rank - tensor.rank)))
            # if the desired output tensor and the current tensor are of the same rank but different shapes, broadcast
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
    def forward(self, inputs):
        return sum(inputs)

    def backward(self):
        for tensor in self.inputs:
            tensor.grad += self.output.grad


@functionize_op
class multiply(MathOp):
    def forward(self, inputs):
        return math.prod(inputs)

    def backward(self, inputs):
        for tensor in self.inputs:
            tensor.grad += self.output.grad * self.output / tensor


@functionize_op
class subtract(MathOp):
    def __init__(self, *inputs, **kwargs):
        if len(inputs) != 2:
            raise ValueError("the power operation requires exactly two input Tensors")
        super().__init__(*inputs, **kwargs)

    def forward(self, inputs):
        a, b = inputs
        return a - b

    def backward(self):
        a, b = self.inputs
        a.grad += self.output.grad
        b.grad -= self.output.grad


@functionize_op
class power(MathOp):
    def __init__(self, *inputs, **kwargs):
        if len(inputs) != 2:
            raise ValueError("the power operation requires exactly two input Tensors")
        super().__init__(*inputs, **kwargs)

    def forward(self, inputs):
        number, exponent = inputs
        return number**exponent

    def backward(self):
        number, exponent = self.inputs
        number.grad = self.output.grad * exponent * number ** (exponent - 1)


@functionize_op
class divide(MathOp):
    def __init__(self, *inputs, **kwargs):
        if len(inputs) != 2:
            raise ValueError("the power operation requires exactly two input Tensors")
        super().__init__(*inputs, **kwargs)

    def forward(self, inputs):
        numerator, denominator = inputs
        return numerator / denominator

    def backward(self):
        numerator, denominator = self.inputs
        numerator.grad += output.grad / denominator
        deniminator.grad += output.grad * -(numerator / denominator**2)


# @functionize_op
# class matmul(MathOp):
#     def __init__(self, tensor1, tensor2, **kwargs):
#         super().__init__(tensor1, tensor2, **kwargs)

#     def _compatibility_check_and_transform(self, *tensors):
#         from tensorguide.framework.tensor import convert_to_tensor
#         from tensorguide.ops.array_ops import broadcast, expand

#         tensors = list(map(convert_to_tensor, tensors))
#         if len(set(t.dtype for t in tensors)) != 1:
#             raise ValueError(
#                 f"Operations require all Tensors to have the same dtype.\
#                 Found dtypes: {set(t.dtype for t in tensors)}"
#             )
#         if any([t.rank < 2 for t in tensors]):
#             raise ValueError("cannot call matmul on a Tensor with rank less than 2")
#         if tensors[0].shape[-1] != tensors[1].shape[-2]:
#             raise ValueError(
#                 "The second to last dimension of the first tensor in a matmul must be equal to the last dimension of the other tensor"
#             )

#         tensors = [
#             broadcast(expand(tensors[0], dims=(-1,)), shape=(*tensors[0].shape, tensors[1].shape[-1])),
#             broadcast(expand(tensors[1], dims=(0,)), shape=(tensors[0].shape[0], *tensors[1].shape)),
#         ]
#         return tensors

#     def _forward(self, tensor1, tensor2):
#         step1 = tensor1 * tensor2
#         return step1.sum(axis=-2)

#     def backward(self):
#         for tensor in self.input:
#             if tensor.grad is None:
#                 tensor.zero_grad()
#             tensor.grad += self.output.grad
