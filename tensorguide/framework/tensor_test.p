class Tensor:
    def __init__(
        self,
        value,
        shape=None,
        dtype="f",
        trainable=True,
        require_grad=True,
        offset=0,
        stride=None,
        contiguous=True,
        _scope="",
        _op=None,
    ):
        ...


class Tensor:
    def __init__(
        self,
        # below parameters are relevant for API for actually creating the Tensor objects
        value: Iterable | Numeric | Callable | Storage,
        shape: int | Iterable[int, ...] | TensorShape | None = None,
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

    def sum(self, axis=0):
        return array_ops._sum(self, axis=axis)

    def __add__(self, tensor):
        if tensor in [0, 0.0]:
            return self
        return math_ops.add(self, tensor)

    def __mul__(self, tensor):
        if tensor in [1, 1.0]:
            return self
        return math_ops.multiply(self, tensor)

    def __radd__(self, tensor):
        return self + tensor

    def __rmul__(self, tensor):
        return self * tensor

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
        self._storage = value if isinstance(value, Storage) else Storage(value)
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

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        axes = list(range(self.rank - 1, -1, -1))
        return tensorguide.ops.permute(self, axes=axes)

    def swapaxes(self, ax1, ax2):
        axes = list(range(self.rank))
        axes[ax2] = ax1
        axes[ax1] = ax2
        return tensorguide.ops.permute(self, axes=axes)

    def __getitem__(self, idx):
        return array_ops._slice(self, idx)

    @property
    def _data(self):
        # only to be used for debugging and printing
        return np.fromiter(self._flat_iterator(), dtype=self.dtype).reshape(self.shape)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__().replace("array", "Tensor")
