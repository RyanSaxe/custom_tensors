from functools import wraps

__all__ = []


def register_op_as_function(name, localvars, default_import=True):
    """Class decorator for Operation classes

    Args:
        name (str):             API name to register calling the operation with
        localvars (dict):       variables from wherever decorator was called to add the function to the namespace
        default_import (bool):  if False, the operation will only be accessable from the module where it was
                                created, and not from tensorguide.ops

    Every [name]_ops.py file should import this function, and every class that inherits from
    tensorguide.framework.ops.Operation should use this decorator to yield the following usage:

    >>> @register_op_as_function("custom_op")
    >>> class CustomOperation(tensorguide.framework.ops.Operation):
    >>>     ...

    This will now register that operation in the globals of this module as a function such that

    >>> custom_module(*args, **kwargs) == CustomOperation(*args, **kwargs).__call__()
    """

    def op(cls):
        @wraps(cls)
        def function(*tensors, **kwargs):
            # NOTE: possible to rewrite so __call__ could take arguments, but for now it is unnecessary. The reason to
            #       do this would be if those arguments are unnecessary to be stored in the Operation on the Graph.
            return cls(*tensors, **kwargs).__call__()

        if name in __all__:
            raise ValueError(
                f"{cls} cant be registered as a function with name {name} because that name has been registered"
            )
        if default_import:
            __all__.append(name)
        globals()[name] = function
        localvars[name] = function
        return cls

    return op
