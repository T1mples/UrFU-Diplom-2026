from algorithm import element_to_str


def vertex_to_page_path(vertex):
    return f"/page/{element_to_str(vertex)}"


def graph_to_web_graph(graph):
    return {
        vertex_to_page_path(vertex): [
            vertex_to_page_path(neighbor) for neighbor in neighbors
        ]
        for vertex, neighbors in graph.items()
    }


def cycle_to_web_route(cycle):
    if not cycle:
        return []
    closed_cycle = cycle + [cycle[0]]
    return [vertex_to_page_path(vertex) for vertex in closed_cycle]


def format_web_route(cycle):
    return " -> ".join(cycle_to_web_route(cycle))
