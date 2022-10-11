from __future__ import annotations

import itertools
import math

import numpy as np

from initializer import UniformInitializer

# NOTE: BELOW CODE DEPRECEATED FOR NOW DUE TO NO NAMED TENSORS
# # this should never happen, so placing an assert statement
# assert tensor1.nd == tensor2.nd
# # if names overlap, but are on different axes, transpose in order to align those axes
# t2_ax_changes = dict()
# for t1_axis, t1_name in enumerate(tensor1.shape._names):
#     for t2_axis, t2_name in enumerate(tensor2.shape._names):
#         if (t1_name == t2_name) & (t1_axis != t2_axis):
#             t2_ax_changes[t2_axis] = t1_axis
# t2_new_axes = list(range(tensor2.nd))
# need_to_swap = len(t2_ax_changes) > 0
# while len(t2_ax_changes) > 0:
#     # get element that needs to swap axes
#     key = list(t2_ax_changes.keys())[0]
#     value = t2_ax_changes.pop(key)
#     # swap the indices
#     temp = t2_new_axes[value]
#     t2_new_axes[value] = t2_new_axes[key]
#     t2_new_axes[key] = temp
#     # update the instructions for swapping
#     if value in t2_ax_changes:
#         t2_ax_changes[key] = t2_ax_changes[value]
#         del t2_ax_changes[value]
# # make sure, before calling permute, that the shapes will be compatible
# for s1, s2 in zip(tensor1.shape, [tensor2.shape[ax] for ax in t2_new_axes]):
#     if s1 in [None, 1] or s2 in [None, 1]:
#         continue
#     if s1 != s2:
#         raise ValueError(f"cannot align shapes {tensor1.shape}, {tensor2.shape} for operations")
# if need_to_swap:
#     tensor2 = tensor2.permute(*t2_new_axes)


def _op_check_and_transform(tensor1, tensor2):
    # can always work with a scaler (that's what rank 0 means)
    if tensor1.nd == 0 or tensor2.nd == 0:
        return tensor1, tensor2
    # if one tensor has different dimensions, expand the dimensions on the right side
    if tensor1.nd < tensor2.nd:
        diff = tensor2.nd - tensor1.nd
        tensor1 = tensor1.reshape((*tuple(1 for _ in range(diff)), *tensor1.shape))
    elif tensor2.nd < tensor1.nd:
        diff = tensor1.nd - tensor2.nd
        tensor2 = tensor2.reshape((*tuple(1 for _ in range(diff)), *tensor2.shape))
    return tensor1._broadcast(tensor2.shape), tensor2._broadcast(tensor1.shape)


def tensor_op(class_function):
    def op_wrapper(self, *args, **kwargs):
        tensor_kwargs = class_function(self, *args, **kwargs)
        # need to allow op functions to do things like `return self` for
        # efficiency. Hence an operation can return a tensor object directly.
        if isinstance(tensor_kwargs, Tensor):
            return tensor_kwargs
        tensor_kwargs.setdefault("name", class_function.__name__)
        tensor_kwargs["name"] = f"{self.name} -> {tensor_kwargs['name']}"
        child = self.__class__(**tensor_kwargs, parents=[self])
        self._children.append(child)
        return child

    return op_wrapper


def two_tensor_op(class_function):
    def op_wrapper(self, tensor, **kwargs):
        self, tensor = _op_check_and_transform(self, tensor)
        wrapped_op = tensor_op(class_function)
        new_tensor = wrapped_op(self, tensor, **kwargs)
        tensor._children.append(new_tensor)
        new_tensor._parents.append(tensor)
        return new_tensor

    return op_wrapper


class TensorShape:
    """_summary_"""

    def __init__(self, shape: int | tuple | list):
        super().__init__()
        self._data = (shape,) if isinstance(shape, int) else tuple(shape)
        self._size = len(self._data)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key: int | slice) -> int | TensorShape:
        item = self._data[key]
        if isinstance(key, slice):
            return TensorShape(item)
        return item

    def __repr__(self) -> str:
        return f"TensorShape[{self._data}]"

    def __setitem__(self, key: int, value: int):
        raise TypeError("TensorShape is Immutable and does not support item assignment")

    def index(self, idx: int) -> int:
        return self._data.index(idx)


class Tensor:
    """_summary_"""

    GLOBAL_NAME_SPACE = dict()
    GLOBAL_TENSOR_SPACE = dict()

    def __init__(
        self,
        shape: int | tuple | list | TensorShape,
        storage: tuple | None = None,
        dtype: Callable = float,
        name: str | None = None,
        initializer: Initializer | None = None,
        stride: tuple | None = None,
        contiguous: bool = True,
        offset: int = 0,
        parents: list | None = None,
    ):
        initializer = UniformInitializer(dtype=dtype) if initializer is None else initializer(dtype=dtype)

        name = "tensor" if name is None else name
        name_count = self.GLOBAL_NAME_SPACE.setdefault(name, 0)
        self._op = name.split(" -> ")[-1]
        self.name = f"{name}[{name_count}]"
        self.GLOBAL_NAME_SPACE[name] += 1
        self.GLOBAL_TENSOR_SPACE[self.name] = self

        self.dtype = dtype
        self.shape = (shape,) if isinstance(shape, int) else shape
        if not isinstance(self.shape, TensorShape):
            self.shape = TensorShape(self.shape)
        self.nd = 0 if self.shape._data == (1,) else len(self.shape)
        self.offset = offset
        self.contiguous = contiguous

        self.flattened_shape = math.prod(self.shape)
        self.stride = (
            tuple(math.prod(self.shape[i + 1 :]) for i in range(len(self.shape))) if stride is None else stride
        )
        self.storage = initializer(self.flattened_shape) if storage is None else storage

        self._parents = [] if parents is None else parents
        self._children = []

    @two_tensor_op
    def __add__(self, tensor):
        left = self._get_substorage()
        right = tensor._get_substorage()
        return dict(shape=self.shape, dtype=self.dtype, storage=[a + b for a, b in zip(left, right)])

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def _substorage(self):
        cell_idxs = itertools.product(*[list(range(s)) for s in self.shape])
        for cell_idx in cell_idxs:
            out = self.offset + sum([s_idx * self.stride[i] for i, s_idx in enumerate(cell_idx)])
            yield self.storage[out]

    def _get_substorage(self):
        substorage = []
        cell_idxs = itertools.product(*[list(range(s)) for s in self.shape])
        for cell_idx in cell_idxs:
            out = self.offset + sum([s_idx * self.stride[i] for i, s_idx in enumerate(cell_idx)])
            substorage.append(self.storage[out])
        return substorage

    def expand_dims(self, dims):
        dims = [self.nd + d if d < 0 else d for d in dims]
        slicer = []
        for i in range(len(expand_dims) + len(t1.shape)):
            if i in expand_dims:
                slicer.append(None)
            else:
                slicer.append(slice(None))
        return self[tuple(slicer)]

    @tensor_op
    def _broadcast(self, to_shape):
        stride = [0 if (self.shape[i] == 1) and (to_shape[i] != 1) else self.stride[i] for i in range(self.nd)]
        if 0 not in stride:
            return self
        return dict(
            dtype=self.dtype,
            shape=to_shape,
            offset=self.offset,
            storage=self.storage,
            contiguous=False,
            stride=stride,
        )

    @tensor_op
    def reshape(self, newshape):
        if math.prod(self.shape) != math.prod(newshape):
            raise ValueError(f"shape {newshape} is invalid for tensor of size {self.flattened_shape}")
        if self.contiguous:
            storage = self.storage
        else:
            warnings.warn("reshaping a non-contiguous memory representation requires copying the data")
            storage = self._get_substorage()
        return dict(
            dtype=self.dtype,
            shape=newshape,
            offset=self.offset,
            storage=storage,
        )

    @tensor_op
    def permute(self, *axes):
        return dict(
            dtype=self.dtype,
            shape=[self.shape[ax] for ax in axes],
            stride=[self.stride[ax] for ax in axes],
            offset=self.offset,
            storage=self.storage,
        )

    def transpose(self):
        axes = list(range(self.nd - 1, -1, -1))
        return self.permute(*axes)

    def swapaxes(self, ax1, ax2):
        axes = list(range(self.nd))
        axes[ax2] = ax1
        axes[ax1] = ax2
        return self.permute(*axes)

    @property
    def _data(self):
        # only to be used for debugging and printing
        return np.asarray(self._get_substorage(), dtype=self.dtype).reshape(self.shape)

    @tensor_op
    def __getitem__(self, idx):
        if self.nd == 0:
            raise ValueError("Cannot index into a Tensor with 0 Dimension (a Scalar)")
        if not isinstance(idx, (int, tuple, slice, type(None))):
            raise ValueError(f"{idx} is of type {type(idx)}, but must be of type int, tuple, slice, or None")
        idx = idx if isinstance(idx, tuple) else (idx,)
        shape = []
        stride = []
        offset = self.offset
        axis = 0
        contiguous = True
        for axis_slice in idx:
            if axis_slice is None:
                shape.append(1)
                # stride number shouldn't really matter here, but this is what pytorch does
                if axis == 0:
                    stride.append(len(self.storage))
                else:
                    stride.append(self.stride[axis - 1])
                continue
            if isinstance(axis_slice, int):
                if axis_slice >= self.shape[axis]:
                    raise IndexError(
                        f"index {axis_slice} is out of bounds for axis {axis} with size {self.shape[axis]}"
                    )
                offset += axis_slice * self.stride[axis]
            else:
                start = 0 if axis_slice.start is None else axis_slice.start
                if start < 0:
                    start = 0 if -start > self.shape[axis] else self.shape[axis] + start
                stop = self.shape[axis] if axis_slice.stop is None else axis_slice.stop
                if stop < 0:
                    stop = 0 if -stop > self.shape[axis] else self.shape[axis] + stop
                else:
                    stop = stop if stop < self.shape[axis] else self.shape[axis]
                step = 1 if axis_slice.step is None else axis_slice.step
                if step <= 0:
                    raise ValueError("step must be greater than 0")
                length = stop - start
                if length < 0:
                    raise ValueError("the stop of the slice must be greater than the start of the slice")
                shape.append(-(length // -step))
                stride.append(self.stride[axis] * step)
                if step != 1:
                    contiguous = False
                offset += start * self.stride[axis]
            axis += 1

        shape = (*shape, *self.shape[axis:])
        stride = (*stride, *self.stride[axis:])
        return dict(
            name=f"slice",
            shape=shape,
            offset=offset,
            storage=self.storage,
            dtype=self.dtype,
            stride=stride,
            contiguous=contiguous,
        )

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__().replace("array", "Tensor")
