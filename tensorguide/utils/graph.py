from graphviz import Digraph


def trace(tensor):
    nodes, edges = set(), set()

    def build_upward(t):
        if t not in nodes:
            nodes.add(t)
            if t._op:
                for parent in t._op.inputs:
                    edges.add((parent, t))
                    build_upward(parent)

    build_upward(tensor)
    return nodes, edges


def draw(tensor, format="svg", rankdir="TB"):
    """visualize the DAG that flows into `tensor` as an output leaf."""
    nodes, edges = trace(tensor)
    graph = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        if n._id == tensor._id:
            color = "blue"
        elif n.trainable:
            color = "green"
        else:
            color = "black"
        graph.node(name=n._id, shape="record", label=f"{str(n)}( shape={n.shape._data} )", color=color)
        if n._op:
            graph.node(name=n._op._id, label=n._op.__class__.__name__.lower())
            for input_tensor in n._op.inputs:
                graph.edge(input_tensor._id, n._op._id, color="green" if input_tensor.require_grad else "black")
                # if input_tensor.require_grad:
                #     graph.edge(n._op._id, input_tensor._id, color="green")
            graph.edge(n._op._id, n._id, color="green" if n.require_grad else "black")
            # if n.require_grad:
            #     graph.edge(n._op._id, n._id, color="green")

    return graph
