import array
import math
from functools import partial
from uuid import uuid4 as uid

import tensorguide.framework.tensor as tensor_test
from tensorguide.framework.graph import Graph


class Operation:
    def __init__(self, *tensors, name=None, parallel=None):
        assert len(tensors) in [1, 2], "currently only supporting operations on a 1 or 2 tensors"
        self._inputs = self._compatibility_check_and_broadcast(*tensors)
        self._graph = Graph._get_current_graph()
        self.parallel = self._graph._njobs != 1 if parallel is None else parallel
        self.name = self._graph._register_op(self, self.__class__.__name__ if name is None else name)
        # NOTE: for simplicity currently requiring all Operations to only have a single output
        self._output = None
        self._id = uid()

        self._set_output_attrs()

    @property
    def input(self):
        if len(self._inputs) == 1:
            return self._inputs[0]
        else:
            return self._inputs

    @property
    def output(self):
        if self._output is None:
            raise ValueError("Cannot access output of an Operation before calling the forward pass.")
        return self._output

    def _set_output_attrs(self):
        self._output_tensor_kwargs = dict(
            name=f"{self._inputs[0].name} -> {self.name}",
            shape=self._inputs[0].shape,
            dtype=self._inputs[0].dtype,
            trainable=False,
            require_grad=any(t.require_grad for t in self._inputs),
            op=self,
            graph=self._graph,
        )

    @classmethod
    def _create_and_apply(cls, *tensors, **kwargs):
        f"""create an instance of {cls.__name__} and run __call__. Used alongside the
        @register_op_as_function decorator for making a functional API"""
        return cls(*tensors, **kwargs)()

    @classmethod
    def _compatibility_check_and_broadcast(cls, *tensors):
        """check that this operation is valid across the tensors. Then expand + broadcast any of them if necessary"""
        tensors = list(map(tensor_test.convert_to_tensor, tensors))
        if len(tensors) <= 1:
            return tensors
        # observe that any operation that operates on a single tensor never executes code below this line
        # this is why we can have Operation function like `expand` and `broadcast`, as they operate on one tensor
        if len(set(t.dtype for t in tensors)) != 1:
            raise ValueError(
                f"Operations require all Tensors to have the same dtype.\
                Found dtypes: {set(t.dtype for t in tensors)}"
            )
        out_rank = max(t.rank for t in tensors)
        expanded_shapes = [(*([1] * (out_rank - t.rank)), *t.shape) for t in tensors]
        out_shape = []
        for i, axis_size in enumerate(zip(*expanded_shapes)):
            sizes = set(axis_size)
            if len(sizes) - (1 in sizes) > 1:
                raise ValueError(
                    f"{cls.__name__} Operation cannot be performed across tensors where the {i}th axis has sizes {axis_size}"
                )
            out_shape.append(max(axis_size))
        output_tensors = []
        for tensor in tensors:
            # if the desired output tensor and current tensor are of different ranks, expand current tensor
            if tensor.rank != out_rank:
                tensor = expand(tensor, dims=list(range(out_rank - tensor.rank)))
            # if the desired output tensor and the current tensor are of the same rank but different shapes, broadcast
            if tensor.shape != out_shape:
                tensor = broadcast(tensor, shape=out_shape)
            output_tensors.append(tensor)
        return output_tensors

    def __call__(self):
        self._output_tensor_kwargs["value"] = self._forward(*self._inputs)
        self._output = tensor_test.Tensor(**self._output_tensor_kwargs)
        return self._output

    def _forward(self, *tensors):
        iterator = tensors[0]._flat_iterator() if len(tensors) == 1 else zip(*[t._flat_iterator() for t in tensors])
        if self.parallel:
            output = Parallel(n_jobs=self._graph.njobs)(delayed(self.forward)(item) for item in iterator)
        else:
            output = (self.forward(item) for item in iterator)
        return array.array(self._output_tensor_kwargs["dtype"], output)

    def forward(self, inputs):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()
