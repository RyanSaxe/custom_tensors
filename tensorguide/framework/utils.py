import array

import numpy as np
from tensorguide.utils.initializer import Initializer


class ClassWithNamedScope:
    _COUNTER = Counter()

    def __init__(self, _scope: str | None = None, _name: str | None = None):
        self._scope = _scope
        if _name is None:
            _name = self.__str__()
        self._name = f"{_name}" if self._scope is None else f"{self._scope}/{_name}"
        self._COUNTER[self._name] += 1

    @property
    def name(self):
        if self._COUNTER[self._name] > 1:
            return f"{self._name}_{self._COUNTER}"
        return self._name

    def __str__(self):
        return self.__class__.__name__

def _flatten(iterable):
    """given any iterable, create a generator for flattening it"""
    for item in iterable:
        if hasattr(item, "__iter__"):
            yield from _flatten(item)
        else:
            yield item


def _compute_shape(obj, depth=1, shape=()):
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
        value = Initializer(shape, dtype)
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
