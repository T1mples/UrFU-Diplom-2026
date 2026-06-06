from algorithm import element_to_str


def load_page_paths(filename):
    pages = []
    with open(filename, "r", encoding="utf-8") as page_file:
        for line in page_file:
            page = line.strip()
            if page and not page.startswith("#"):
                pages.append(page)
    if not pages:
        raise ValueError("Файл со страницами пуст.")
    return pages


def vertex_to_page_path(vertex):
    return f"/page/{element_to_str(vertex)}"


def build_vertex_page_map(vertices, page_paths):
    if len(page_paths) < len(vertices):
        raise ValueError(
            "Недостаточно страниц для сопоставления: "
            f"нужно {len(vertices)}, получено {len(page_paths)}."
        )
    return dict(zip(vertices, page_paths[:len(vertices)]))


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


def cycle_to_page_route(cycle, vertex_page_map):
    if not cycle:
        return []
    closed_cycle = cycle + [cycle[0]]
    return [vertex_page_map[vertex] for vertex in closed_cycle]


def format_page_route(cycle, vertex_page_map):
    return " -> ".join(cycle_to_page_route(cycle, vertex_page_map))
