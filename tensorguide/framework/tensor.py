from __future__ import annotations

import array
import itertools
import math
from typing import Callable, Iterable
from uuid import uuid4 as uid

import numpy as np
import tensorguide.ops as ops
from tensorguide.framework.graph import Graph


def _flatten(iterable: Iterable) -> Iterable:
    """given any iterable, create a generator for flattening it"""
    for item in iterable:
        if hasattr(item, "__iter__"):
            yield from _flatten(item)
        else:
            yield item


Numeric = int | float


def _compute_shape(obj: Iterable | Numeric, depth: int = 1, shape: tuple = ()) -> tuple[int, ...]:
    """given any obj, return the expected higher dimensional shape of the obj"""
    if isinstance(obj, Numeric):
        return ()
    if hasattr(obj, "shape"):
        return (*shape, obj.shape)
    shape = (*shape, len(obj))
    for item in obj:
        if hasattr(item, "__iter__"):
            shape = _compute_shape(item, depth=depth + 1, shape=shape)
        else:
            return shape
        break
    return shape


def convert_to_tensor(obj: Tensor | Numeric | Iterable, dtype: str = "f") -> Tensor:
    """given a Tensor, number, or iterable, return the Tensor representation of obj"""
    if isinstance(obj, Tensor):
        return obj
    if not hasattr(obj, "__iter__"):
        return Tensor(shape=(), dtype=dtype, value=array.array(dtype, [obj]))
    return Tensor(dtype=dtype, shape=_compute_shape(obj), value=array.array(dtype, _flatten(obj)))


class TensorShape:
    """Container for Shape objects with immutable structures"""

    def __init__(self, shape: int | Iterable[int, ...]):
        super().__init__()
        self._data = (shape,) if isinstance(shape, int) else tuple(shape)
        self._nd = len(self._data)

    def __len__(self) -> int:
        return self._nd

    def __getitem__(self, key: int | slice) -> int | TensorShape:
        item = self._data[key]
        if isinstance(key, slice):
            return TensorShape(item)
        return item

    def __repr__(self) -> str:
        return f"TensorShape[{self._data}]"

    def __setitem__(self, key: int, value: int):
        raise TypeError("TensorShape is Immutable and does not support item assignment")

    def __eq__(self, obj):
        if isinstance(obj, TensorShape):
            return obj._data == self._data
        if isinstance(obj, int):
            obj = (obj,)
        return self._data == tuple(obj)

    def index(self, idx: int) -> int:
        return self._data.index(idx)


class Storage:
    def __init__(self, for_tensor: Tensor, data: array.array | None = None):
        self.for_tensor = for_tensor
        self._data = data

    def assign_data(self, data: array.array, graph: Graph):
        if graph != self.for_tensor._graph:
            raise ValueError("cannot assign Storage for a tensor from a different Graph")
        self._data = data

    def __getitem__(self, key: int) -> Numeric:
        return self._data[key]

    def __repr__(self):
        return self._data.__repr__()


class Tensor:
    def __init__(
        self,
        # below parameters are relevant for API for actually creating the Tensor objects
        value: Iterable | Numeric | Callable | Storage,
        shape: int | Iterable | TensorShape | None = None,
        dtype: str = "f",
        name: str | None = None,
        trainable: bool = True,
        require_grad: bool = True,
        # below parameters are for enabling Tensor to function with contiguous memory
        # as well as minimizing copying any memory (e.g. transpose -> O(1) operation)
        offset: int = 0,
        stride: tuple | None = None,
        contiguous: bool = True,
        # below parameters are for proper storage and pointers for this Tensor on the Graph
        graph: Graph | None = None,
        op: Operation | None = None,
    ):
        self._graph = Graph._get_current_graph() if graph is None else graph
        self._op = op
        self._id = uid()
        self._children = []

        self.name = self._graph._register_tensor(self, name)
        self.dtype = dtype
        self._set_shape_and_store_value(shape, value)
        self.rank = self.shape._nd
        self.trainable = trainable
        self.require_grad = require_grad
        self._grad = None

        self._stride = tuple(math.prod(self.shape[i + 1 :]) for i in range(self.rank)) if stride is None else stride
        self._offset = offset
        self._contiguous = contiguous

    def __add__(self, tensor):
        return ops.add(self, tensor)

    def _set_shape_and_store_value(self, shape, value):
        if isinstance(value, Numeric):
            value = [value]
            assert shape is None, "cannot pass the shape argument for a scalar Tensor"
            shape = ()
        elif isinstance(value, Iterable):
            if self._op is None:
                assert shape is None, "cannot pass the shape argument when initializing a Tensor with an iterable"
                shape = _compute_shape(value)
            else:
                assert shape is not None, "Operations must specify the output shape of the computed Tensor"
            if isinstance(value[0], Iterable):
                value = _flatten(value)
            value = array.array(self.dtype, value)
        elif isinstance(value, Callable):
            assert shape is not None, "shape argument is required in order to use an Callable"
            value = value(shape, self.dtype)
        self._storage = value if isinstance(value, Storage) else Storage(self, value)
        self.shape = shape if isinstance(shape, TensorShape) else TensorShape(shape)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def _flat_iterator(self):
        # a scalar tensor has automatic broadcasting via this iterator
        if self.rank == 0:
            while True:
                yield self._storage[0]
        # if not Scalar tensor, get all possible indices in the Tensor and iterate over them
        cell_idxs = itertools.product(*[list(range(s)) for s in self.shape])
        for cell_idx in cell_idxs:
            out = self._offset + sum([s_idx * self._stride[i] for i, s_idx in enumerate(cell_idx)])
            yield self._storage[out]

    def transpose(self):
        axes = list(range(self.rank - 1, -1, -1))
        return ops.permute(self, axes=axes)

    def swapaxes(self, ax1, ax2):
        axes = list(range(self.rank))
        axes[ax2] = ax1
        axes[ax1] = ax2
        return ops.permute(self, axes=axes)

    def __getitem__(self, idx):
        return ops.slice(self, idx)

    @property
    def _data(self):
        # only to be used for debugging and printing
        return np.fromiter(self._flat_iterator(), dtype=self.dtype).reshape(self.shape)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__().replace("array", "Tensor")
