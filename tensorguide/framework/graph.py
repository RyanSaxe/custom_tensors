from uuid import uuid4 as uid

from graphviz import Digraph


class Graph:

    GRAPHS = dict()

    def __init__(self, njobs=1, seed=None, name: str = "DEFAULT"):
        self.njobs = njobs
        self._seed = seed
        self._ops = dict()
        self._ops_counter = dict()
        self._tensors = dict()
        self._unnamed_tensor_counter = 0
        self._id = uid()
        assert name not in self.GRAPHS, f"there is already a Graph with the name {name}"
        self.name = name
        self.GRAPHS[self.name] = self

    def trace(self, tensor):
        nodes, edges = set(), set()

        def build_upward(t):
            if t not in nodes:
                nodes.add(t)
                if t._op:
                    for parent in t._op._inputs:
                        edges.add((parent, t))
                        build_upward(parent)

        build_upward(tensor)
        return nodes, edges

    def draw(self, tensor, format="svg", rankdir="TB"):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)

        visualize the DAG leading into a `tensor` such that `tensor` is the bottom leaf.
        """
        assert rankdir in ["LR", "TB"]
        nodes, edges = self.trace(tensor)
        graph = Digraph(format=format, graph_attr={"rankdir": rankdir})

        for n in nodes:
            graph.node(name=n.name, shape="record", label=f"Tensor(name={n.name},shape={n.shape._data})")
            if n._op:
                graph.node(name=n._op.name, label=n._op.__class__.__name__.lower())
                for input_tensor in n._op._inputs:
                    graph.edge(input_tensor.name, n._op.name)
                graph.edge(n._op.name, n.name)

        return graph

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
