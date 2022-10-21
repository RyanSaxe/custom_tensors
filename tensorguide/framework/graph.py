from uuid import uuid4 as uid


class Graph:

    GRAPHS = dict()

    def __init__(self, njobs=1, seed=None, name: str = "DEFAULT"):
        self._njobs = njobs
        self._seed = seed
        self._ops = dict()
        self._ops_counter = dict()
        self._tensors = dict()
        self._unnamed_tensor_counter = 0
        self._id = uid()
        assert name not in self.GRAPHS, f"there is already a Graph with the name {name}"
        self.name = name
        self.GRAPHS[self.name] = self

    @classmethod
    def _get_current_graph(cls):
        graph = cls.GRAPHS.get("DEFAULT")
        if graph is None:
            graph = cls()
            cls.GRAPHS["DEFAULT"] = graph
        return graph

    def _register_tensor(self, tensor, name):
        # if name in self._tensors:
        #     raise ValueError(f"a Tensor with the name {name} already exists on this Graph.")
        if name is None:
            name = f"UNNAMEDTENSOR{self._unnamed_tensor_counter}"
            self._unnamed_tensor_counter += 1
        # elif name.startswith("UNNAMEDTENSOR"):
        #     raise ValueError("cannot create a tensor starting with the string 'UNNAMEDTENSOR'")
        self._tensors[name] = tensor
        return name

    def _register_op(self, op, name):
        counter = self._ops_counter.setdefault(name, 0)
        self._ops_counter[name] += 1
        name = f"{name}{counter}"
        self._ops[name] = op
        return name
