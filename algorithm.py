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


def find_hamiltonian_cycle(graph):
    nodes = list(graph.keys())
    n = len(nodes)
    for start in nodes:
        stack = [([start], {start})]
        while stack:
            path, visited = stack.pop()
            if len(path) == n:
                if path[0] in graph[path[-1]]:
                    return path
                continue

            for neighbor in reversed(graph[path[-1]]):
                if neighbor not in visited:
                    stack.append((path + [neighbor], visited | {neighbor}))
    return None


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
