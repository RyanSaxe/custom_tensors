import array
import math
from functools import partial
from uuid import uuid4 as uid

from tensorguide.framework.graph import Graph
from tensorguide.framework.tensor import Tensor, convert_to_tensor


class Operation:
    def __init__(self, *tensors, name=None, parallel=None):
        assert len(tensors) in [1, 2], "currently only supporting operations on a 1 or 2 tensors"
        self._inputs = self._compatibility_check_and_transform(*tensors)
        self._graph = Graph._get_current_graph()
        self.parallel = self._graph._njobs != 1 if parallel is None else parallel
        self.name = self._graph._register_op(self, self.__class__.__name__ if name is None else name)
        # NOTE: for simplicity currently requiring all Operations to only have a single output
        self._output = None
        self._id = uid()

        self._set_output_tensor_kwargs()

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

    def _set_output_tensor_kwargs(self):
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
    def _compatibility_check_and_transform(cls, *tensors):
        """check that this operation is valid across the tensors. Then expand + broadcast any of them if necessary"""
        tensors = list(map(convert_to_tensor, tensors))
        if len(set(t.dtype for t in tensors)) != 1:
            raise ValueError(
                f"Operations require all Tensors to have the same dtype.\
                Found dtypes: {set(t.dtype for t in tensors)}"
            )
        return tensors

    def __call__(self):
        self._output_tensor_kwargs["value"] = self._forward(*self._inputs)
        self._output = Tensor(**self._output_tensor_kwargs)
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
