from tensorguide.ops import mathops, shapeops


# all functional ops are registered in the general scope of ops
# >>> import ops
# >>> ops.add # this is a functional alias for ops.mathops.Add
def register_op_as_function(name):
    def op(cls):
        def function(*tensors, **kwargs):
            return cls._create_and_apply(*tensors, **kwargs)

        globals()[name] = function
        return cls

    return op
