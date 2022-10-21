from tensorguide.ops import array_ops, math_ops

# the code below import loads every Operation wrapped in @register_op_as_function("name") such that
# >>> from tensorguide.ops import name
# will work for all operation modules specified in the first line of this file: from tensorguide.ops import . . .
from tensorguide.ops.op_registry import *
