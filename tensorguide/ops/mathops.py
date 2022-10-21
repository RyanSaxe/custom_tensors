from tensorguide.framework.ops import Operation

from tensorguide.ops import register_op_as_function


@register_op_as_function("add")
class Add(Operation):
    def forward(self, *inputs):
        return sum(*inputs)

    def _backward(self):
        for tensor in self.input:
            tensor.grad += self.output.grad
