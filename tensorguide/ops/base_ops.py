from tensorguide.framework.ops import Operation


class ArrayOp(Operation):
    """Types of operations that never modify the data on a Tensor, and instead just modify things like the shape"""

    def __init__(self, tensor, copy=False, name=None):
        self.copy = copy
        super().__init__(tensor, name=name, parallel=False)

    def _get_output_tensor_kwargs(self):
        return super()._get_output_tensor_kwargs() | dict(
            contiguous=self.input._contiguous,
            offset=self.input._offset,
        )

    def _forward(self, tensor):
        return tensor._flat_iterator if self.copy else tensor._storage
