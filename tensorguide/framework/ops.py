import abc
from functools import wraps
from uuid import uuid4 as uid

from joblib import Parallel, delayed
from tensorguide.utils import numeric


def functionize_op(op_cls):
    @wraps(op_cls)
    def call_op(*args, **kwargs):
        return op_cls(*args, **kwargs).__call__()

    return call_op


class Operation:
    min_n_inputs = 1
    max_n_inputs = numeric.inf
    n_inputs = None

    def __init__(self, *inputs, njobs=1, stop_gradient=False):
        from tensorguide.framework.tensor import convert_to_tensor

        self.inputs = list(map(convert_to_tensor, inputs))
        self.inputs = self.transform_and_check_input_tensors()
        self.output_tensor_kwargs = self.get_output_tensor_kwargs()
        if stop_gradient:
            self.output_tensor_kwargs["require_grad"] = False
        self.output = None
        self.njobs = njobs

        self.name = self.__class__.__name__
        self._id = str(uid())

    def get_output_tensor_kwargs(self):
        return dict(
            trainable=False,
            require_grad=any(tensor.require_grad for tensor in self.inputs),
            dtype=self.inputs[0].dtype,
            _op=self,
        )

    def transform_and_check_input_tensors(self):
        n_inputs = len(self.inputs)
        if self.n_inputs is not None:
            if self.n_inputs != n_inputs:
                raise ValueError(f"This operation requires exactly {self.n_inputs} inputs, but {n_inputs} were given.")
        elif n_inputs < self.min_n_inputs or n_inputs > self.max_n_inputs:
            raise ValueError(
                f"Number of inputs to this operation is {n_inputs}, which is outside the required range [{self.min_n_inputs},{self.max_n_inputs}]"
            )
        else:
            self.n_inputs = n_inputs
        dtypes = set(t.dtype for t in self.inputs)
        if len(dtypes) != 1:
            raise ValueError(
                f"Operations require all Tensors to have the same dtype.\
                Found dtypes: {dtypes}"
            )
        return self.inputs

    def __call__(self):
        from tensorguide.framework.tensor import Tensor

        inputs = self.inputs_iterator()
        # disabling parallel for quicker debugging
        # self.output = Parallel(n_jobs=self.njobs)(delayed(self.forward)(item) for item in inputs)
        output = list(map(self.forward, inputs))
        if not isinstance(output[0], Tensor):
            self.output_tensor_kwargs["value"] = output
            self.output = Tensor(**self.output_tensor_kwargs)
        else:
            if len(output) != 1:
                raise ValueError("An operation is returning multiple tensors. This should not happen.")
            # in case there are sub-operations, we overwrite the pointer of output to the current op
            self.output = output[0]
            self.output._op = self
        return self.output

    @abc.abstractmethod
    def inputs_iterator(self):
        ...

    @abc.abstractmethod
    def forward(self, *inputs):
        ...

    @abc.abstractmethod
    def backward(self):
        ...
