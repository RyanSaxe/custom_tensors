import abc
from functools import wraps

from joblib import Parallel, delayed


def functionize_op(op_cls):
    @wraps(op_cls)
    def call_op(*args, **kwargs):
        return op_cls(*args, **kwargs).__call__()

    return call_op


class Operation:
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

    def get_output_tensor_kwargs(self):
        return dict(
            trainable=False,
            require_grad=any(tensor.require_grad for tensor in self.inputs),
            dtype=self.inputs[0].dtype,
            _op=self,
        )

    def transform_and_check_input_tensors(self):
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
        self.output = Parallel(n_jobs=self.njobs)(delayed(self.forward)(item) for item in inputs)
        if not isinstance(self.output, Tensor):
            self.output_tensor_kwargs["value"] = self.output
            self.output = Tensor(**self.output_tensor_kwargs)
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
