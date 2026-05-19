import time


def dihedral_multiply(g, h, n):
    k1, f1 = g
    k2, f2 = h
    if f1 == 0:
        return ((k1 + k2) % n, f2)
    return ((k1 - k2) % n, f1 ^ f2)


def identity():
    return (0, 0)


def build_dihedral_group(n):
    return [(k, f) for k in range(n) for f in (0, 1)]


def is_involution(elem, n):
    return dihedral_multiply(elem, elem, n) == identity()


def commutes(a, b, n):
    return dihedral_multiply(a, b, n) == dihedral_multiply(b, a, n)


def generate_subgroup(generators, n):
    closure = {identity()}
    stack = [identity()]
    while stack:
        current = stack.pop()
        for gen in generators:
            for neighbor in (dihedral_multiply(current, gen, n), dihedral_multiply(gen, current, n)):
                if neighbor not in closure:
                    closure.add(neighbor)
                    stack.append(neighbor)
    return closure


def build_cayley_graph(group, generators, n):
    graph = {}
    for g in group:
        graph[g] = [dihedral_multiply(g, s, n) for s in generators]
    return graph


def find_hamiltonian_cycle_with_stats(graph, time_limit_seconds=None):
    nodes = list(graph.keys())
    n = len(nodes)
    started_at = time.perf_counter()
    states_checked = 0

    def elapsed_seconds():
        return time.perf_counter() - started_at

    def build_result(cycle, status):
        return {
            "cycle": cycle,
            "status": status,
            "states_checked": states_checked,
            "elapsed_seconds": elapsed_seconds(),
        }

    if time_limit_seconds is not None and time_limit_seconds <= 0:
        return build_result(None, "timeout")

    def time_limit_exceeded():
        return (
            time_limit_seconds is not None
            and elapsed_seconds() >= time_limit_seconds
        )

    for start in nodes:
        if time_limit_exceeded():
            return build_result(None, "timeout")
        stack = [([start], {start})]
        while stack:
            if time_limit_exceeded():
                return build_result(None, "timeout")
            path, visited = stack.pop()
            states_checked += 1
            if len(path) == n:
                if path[0] in graph[path[-1]]:
                    return build_result(path, "found")
                continue

            for neighbor in reversed(graph[path[-1]]):
                if neighbor not in visited:
                    stack.append((path + [neighbor], visited | {neighbor}))
    return build_result(None, "not_found")


def find_hamiltonian_cycle(graph):
    return find_hamiltonian_cycle_with_stats(graph)["cycle"]


def build_dihedral_involution_generators(n, third_shift=1):
    if n % 2 != 0:
        raise ValueError("Для трёх инволюций с двумя коммутирующими требуется чётный n.")
    a = (0, 1)
    b = (n // 2, 1)
    c = (third_shift % n, 1)
    return [a, b, c]


def element_to_str(elem):
    k, f = elem
    if elem == identity():
        return "e"
    if f == 0:
        return f"r{k}"
    return f"r{k}s"
