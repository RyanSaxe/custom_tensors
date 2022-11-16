import array
import itertools
import math
from uuid import uuid4 as uid

import numpy as np
from tensorguide.ops import array_ops, math_ops, transform_ops
from tensorguide.utils.initializer import Initializer


def _flatten(iterable):
    """given any iterable, create a generator for flattening it"""
    for item in iterable:
        if hasattr(item, "__iter__"):
            yield from _flatten(item)
        else:
            yield item


def _compute_shape(obj, depth=1, shape=()):
    """given any obj, return the expected higher dimensional shape of the obj"""
    if isinstance(obj, (int, float)):
        return ()
    if hasattr(obj, "shape"):
        return (*shape, obj.shape)
    shape = (*shape, len(obj))
    for item in obj:
        if hasattr(item, "__iter__"):
            shape = _compute_shape(item, depth=depth + 1, shape=shape)
        else:
            return shape
        # TODO: check if nested objects are not square and raise an error
        break
    return shape


def _create_shape_and_value(shape, value, dtype):
    if isinstance(value, (int, float)):
        if shape not in [(), None]:
            raise ValueError(f"{shape} is not a valid shape for a scalar Tensor")
        value = array.array(dtype, [value])
        shape = ()
    elif isinstance(value, np.ndarray):
        if shape is None:
            shape = value.shape
        value = array.array(dtype, value.flatten())
    elif isinstance(value, Initializer):
        shape = () if shape is None else shape
        value = value(shape, dtype)
    elif isinstance(value, list):
        if shape is None:
            shape = _compute_shape(value)
        value = array.array(dtype, _flatten(value))
    elif isinstance(value, array.ArrayType):
        if self.dtype != value.typecode:
            raise ValueError(
                f"cannot create a tensor of dtype {dtype} when the dtype of the passed value is {value.typecode}"
            )
        if shape is None:
            shape = ()
    else:
        raise ValueError(
            f"value is of type {type(value)}. Supported types are list, int, float, array.array, np.ndarray, and Initializer"
        )
    return shape, value


class TensorShape:
    """Container for Shape objects with immutable structures"""

    def __init__(self, shape):
        super().__init__()
        self._data = (shape,) if isinstance(shape, int) else tuple(shape)
        self._nd = len(self._data)

    def __len__(self):
        return self._nd

    def __getitem__(self, key):
        item = self._data[key]
        if isinstance(key, slice):
            return TensorShape(item)
        return item

    def __repr__(self):
        return f"TensorShape[{self._data}]"

    def __setitem__(self, key, value):
        raise TypeError("TensorShape is Immutable and does not support item assignment")

    def __eq__(self, obj):
        if isinstance(obj, TensorShape):
            return obj._data == self._data
        if isinstance(obj, int):
            obj = (obj,)
        return self._data == tuple(obj)

    def index(self, idx):
        return self._data.index(idx)


def convert_to_tensor(obj):
    # avoid circular import by importing Tensor inside function scope
    # from tensorguide.framework.tensor import Tensor
    if isinstance(obj, Tensor):
        return obj
    return Tensor(
        value=obj,
        require_grad=False,
        trainable=False,
    )


class Tensor:
    def __init__(
        self,
        value,
        shape=None,
        dtype="f",
        trainable=True,
        require_grad=True,
        _offset=0,
        _stride=None,
        _contiguous=True,
        _op=None,
    ):
        self._op = _op

        self.trainable = trainable
        self.require_grad = require_grad

        if isinstance(dtype, type):
            if dtype not in [int, float]:
                raise ValueError("dtype specification via builtin types is only supported for int and float")
            dtype = "f" if dtype == float else "i"
        self.dtype = dtype

        self.shape, self._value = _create_shape_and_value(shape, value, self.dtype)
        if not isinstance(self.shape, TensorShape):
            self.shape = TensorShape(self.shape)
        self.rank = len(self.shape)

        self._stride = tuple(math.prod(self.shape[i + 1 :]) for i in range(self.rank)) if _stride is None else _stride
        self._offset = _offset
        self._contiguous = _contiguous

        self._grad = 0.0
        self._children = []
        self._id = str(uid())

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def _flat_iterator(self):
        # a scalar tensor has automatic broadcasting via this iterator
        if self.rank == 0:
            while True:
                yield self._value[self._offset]
        # if not Scalar tensor, get all possible indices in the Tensor and iterate over them
        cell_idxs = itertools.product(*[list(range(s)) for s in self.shape])
        for cell_idx in cell_idxs:
            out = self._offset + sum([s_idx * self._stride[i] for i, s_idx in enumerate(cell_idx)])
            yield self._value[out]

    @property
    def _data(self):
        # only to be used for debugging and printing
        if self.rank == 0:
            return self._value[self._offset]
        return np.fromiter(self._flat_iterator(), dtype=self.dtype).reshape(self.shape)

    def __str__(self):
        return "Tensor" if (self.trainable or self._op is not None) else "Constant"

    def __repr__(self):
        return self._data.__repr__().replace("array", self.__str__())

    # def sum(self, axis=0):
    #     return array_ops._sum(self, axis=axis)

    def __add__(self, tensor):
        if tensor in [0, 0.0]:
            return self
        return math_ops.add(self, tensor)

    def __mul__(self, tensor):
        if tensor in [1, 1.0]:
            return self
        return math_ops.multiply(self, tensor)

    def __pow__(self, tensor):
        if tensor in [1, 1.0]:
            return self
        return math_ops.power(self, tensor)

    def __truediv__(self, tensor):
        if tensor in [1, 1.0]:
            return self
        return math_ops.divide(self, tensor)

    def __sub__(self, tensor):
        if tensor in [0, 0.0]:
            return self
        return math_ops.subtract(self, tensor)

    def __radd__(self, tensor):
        return self + tensor

    def __rmul__(self, tensor):
        return self * tensor

    def __rpow__(self, tensor):
        return self**tensor

    def __rtruediv__(self, tensor):
        return self / tensor

    def __rsub__(self, tensor):
        return self - tensor

    def __abs__(self):
        return transform_ops._abs(self)

    def __neg__(self):
        return transform_ops.neg(self)

    def transpose(self):
        axes = list(range(self.rank - 1, -1, -1))
        return permute(self, axes=axes)

    def swapaxes(self, ax1, ax2):
        axes = list(range(self.rank))
        axes[ax2] = ax1
        axes[ax1] = ax2
        return permute(self, axes=axes)

    def expand(self, dims):
        return array_ops.expand(self, dims=dims)

    def broadcast_to(self, shape):
        return array_ops.broadcast(self, shape=shape)

    def __getitem__(self, idx):
        return array_ops._slice(self, idx)
