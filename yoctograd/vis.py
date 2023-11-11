import graphviz as gv


def traverse(node):

    nodes = set()
    edges = set()
    queue = [node]

    while queue:
        node = queue.pop(0)
        nodes.add(node)
        for child in node.children:
            edges.add((child, node))
            queue.append(child)

    return nodes, edges


def draw_graph(node):

    nodes, edges = traverse(node)

    # Draw the graph from left to right
    d = gv.Digraph(format='svg',
                   node_attr={'shape': 'record'},
                   graph_attr={'rankdir': 'LR'})

    for node in nodes:

        if node.label:
            label = node.label + "|"
        else:
            label = ""
        d.node(str(id(node)),
               (rf"{{{label}data: {node.data:.3f}|grad: {node.grad:.3f}}}"))
        if node.op:
            d.node(str(id(node)) + 'op', node.op, shape='circle')
            d.edge(str(id(node)) + 'op', str(id(node)))

    for edge in edges:
        d.edge(str(id(edge[0])), str(id(edge[1])) + 'op')

    return d
