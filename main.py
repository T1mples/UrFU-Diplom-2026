import datetime
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from algorithm import (
    build_cayley_graph,
    build_dihedral_group,
    build_dihedral_involution_generators,
    commutes,
    element_to_str,
    find_hamiltonian_cycle,
    generate_subgroup,
    is_involution,
)


def format_duration(total_seconds):
    total_seconds = float(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def find_non_commuting_pairs(generators, n):
    non_commuting_pairs = []
    for i in range(len(generators)):
        for j in range(i + 1, len(generators)):
            if not commutes(generators[i], generators[j], n):
                non_commuting_pairs.append((i, j))
    return non_commuting_pairs


def draw_graph(graph, generators):
    graph_view = nx.DiGraph()
    labels = {}
    edge_colors = ["#d1495b", "#2f9e44", "#2458b3"]
    edge_groups = [set() for _ in generators]

    for node, neighbors in graph.items():
        labels[node] = element_to_str(node)
        for index, neighbor in enumerate(neighbors):
            graph_view.add_edge(node, neighbor)
            if index < len(edge_groups):
                edge_groups[index].add(frozenset((node, neighbor)))

    undirected_view = nx.Graph(graph_view)
    try:
        pos = nx.spring_layout(undirected_view, seed=42)
    except ModuleNotFoundError:
        print(
            "Для spring_layout не найден scipy. Используется круговая схема размещения."
        )
        pos = nx.circular_layout(undirected_view)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(
        graph_view,
        pos,
        node_size=700,
        node_color="#1f4e79",
    )
    nx.draw_networkx_labels(graph_view, pos, labels=labels, font_color="white")

    legend_handles = []
    for index, generator in enumerate(generators):
        color = edge_colors[index % len(edge_colors)]
        undirected_edges = [tuple(edge) for edge in edge_groups[index]]
        nx.draw_networkx_edges(
            graph_view,
            pos,
            edgelist=undirected_edges,
            arrows=True,
            arrowstyle="<|-|>",
            arrowsize=16,
            edge_color=color,
            width=1.7,
            min_source_margin=15,
            min_target_margin=15,
            connectionstyle="arc3,rad=0.0",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                marker=">",
                markersize=7,
                label=f"g{index + 1} = {element_to_str(generator)}",
            )
        )

    plt.legend(
        handles=legend_handles,
        title="Инволюции",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    plt.axis("off")
    plt.tight_layout(rect=(0, 0, 0.82, 1))
    plt.show()


def write_report(
    filename,
    iterations,
    hamilton_count,
    exception_info,
    stopped_by_user,
    last_checked,
    skipped_non_generating,
    checked_hamiltonian,
    total_elapsed_seconds,
):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Отчет создан: {now}",
        f"Итераций выполнено: {iterations}",
        f"Общее время вычислений: {format_duration(total_elapsed_seconds)}",
        f"Пропущено непорождающих систем: {skipped_non_generating}",
        f"Проверено на гамильтоновость: {checked_hamiltonian}",
        f"С гамильтоновым циклом: {hamilton_count}",
    ]

    if exception_info:
        lines.append("Исключение: найдено")
        lines.append(f"Проверка: {exception_info['check']}")
        lines.append(f"n = {exception_info['n']}, k = {exception_info['k']}")
        lines.append(
            f"Генераторы: {[element_to_str(generator) for generator in exception_info['generators']]}"
        )
        if exception_info.get("details"):
            lines.append(f"Подробности: {exception_info['details']}")
    else:
        lines.append("Исключение: не найдено")
        if stopped_by_user:
            lines.append("Остановлено пользователем.")
        else:
            lines.append("Остановлено без исключения.")

    if last_checked:
        lines.append(f"Последняя проверка: n = {last_checked['n']}, k = {last_checked['k']}")

    if MAX_HAMILTON_CHECK_N is not None:
        lines.append(
            f"Проверка гамильтонова цикла выполнялась для n <= {MAX_HAMILTON_CHECK_N}."
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    if os.path.isabs(filename):
        filename_full = filename
    else:
        filename_full = os.path.join(reports_dir, filename)

    dirname = os.path.dirname(filename_full)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename_full, "w", encoding="utf-8") as report:
        report.write("\n".join(lines))

    print(f"Отчет сохранен в {filename_full}")


def check_parameters(n, k):
    generators = build_dihedral_involution_generators(n, k)
    if not all(is_involution(generator, n) for generator in generators):
        return {
            "generators": generators,
            "generates": False,
            "hamiltonian": False,
            "cycle_checked": False,
            "details": "Один или несколько генераторов не являются инволюцией.",
        }

    non_commuting_pairs = find_non_commuting_pairs(generators, n)
    if non_commuting_pairs:
        pair_labels = ", ".join(f"g{i + 1}-g{j + 1}" for i, j in non_commuting_pairs)
        return {
            "generators": generators,
            "generates": False,
            "hamiltonian": False,
            "cycle_checked": False,
            "details": f"Не все пары генераторов коммутируют: {pair_labels}.",
        }

    group_size = 2 * n
    closure = generate_subgroup(generators, n)
    generates = len(closure) == group_size
    hamiltonian = False
    cycle_checked = n <= MAX_HAMILTON_CHECK_N

    if cycle_checked and generates:
        graph = build_cayley_graph(build_dihedral_group(n), generators, n)
        cycle = find_hamiltonian_cycle(graph)
        hamiltonian = cycle is not None

    return {
        "generators": generators,
        "generates": generates,
        "hamiltonian": hamiltonian,
        "cycle_checked": cycle_checked,
        "details": None,
    }


def search_exceptions(max_iterations=None):
    print("=== Начинаем поиск исключений ===")
    try:
        max_n = int(
            input("Введите максимальное n для проверки гамильтонова цикла: ")
        )
    except ValueError:
        max_n = 10

    global MAX_HAMILTON_CHECK_N
    MAX_HAMILTON_CHECK_N = max_n
    print(
        f"Проверка гамильтонова цикла будет выполняться для n <= {MAX_HAMILTON_CHECK_N}"
    )

    iterations = 0
    hamilton_count = 0
    skipped_non_generating = 0
    checked_hamiltonian = 0
    total_elapsed_seconds = 0.0
    exception_info = None
    last_checked = None
    n = 2

    try:
        while True:
            if n % 2 != 0:
                n += 1
                continue

            for k in range(1, n):
                if max_iterations is not None and iterations >= max_iterations:
                    raise StopIteration

                started_at = time.perf_counter()
                params = check_parameters(n, k)
                iteration_elapsed = time.perf_counter() - started_at

                iterations += 1
                total_elapsed_seconds += iteration_elapsed
                last_checked = {"n": n, "k": k}

                generator_str = ", ".join(
                    element_to_str(generator) for generator in params["generators"]
                )
                status = "порождают" if params["generates"] else "не порождают"
                print(
                    f"[итерация {iterations}] n={n}, k={k}, генераторы=[{generator_str}] "
                    f"=> {status} группу D_{n}; "
                    f"время итерации: {iteration_elapsed:.4f} с; "
                    f"суммарно: {format_duration(total_elapsed_seconds)}"
                )

                if not params["generates"]:
                    print(f"  Пропуск: система не является порождающей для D_{n}.")
                    skipped_non_generating += 1
                    continue

                if params["cycle_checked"]:
                    checked_hamiltonian += 1
                    if params["hamiltonian"]:
                        hamilton_count += 1
                        print("  Проверка гамильтонова цикла: найден.")
                    else:
                        print("  Проверка гамильтонова цикла: не найден.")
                        exception_info = {
                            "check": "отсутствие гамильтонова цикла",
                            "n": n,
                            "k": k,
                            "generators": params["generators"],
                            "details": "Цикл не найден при полном графе Кэли.",
                        }
                        raise StopIteration
                else:
                    print(
                        f"  Проверка гамильтонова цикла пропущена для n={n} "
                        f"(n > {MAX_HAMILTON_CHECK_N})."
                    )

            n += 2
    except KeyboardInterrupt:
        print("\nПользователь остановил поиск.")
        report_name = f"gamilthon_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
        write_report(
            report_name,
            iterations,
            hamilton_count,
            exception_info,
            True,
            last_checked,
            skipped_non_generating,
            checked_hamiltonian,
            total_elapsed_seconds,
        )
        return
    except StopIteration:
        if exception_info:
            print("\nНайден контрпример к гипотезе о гамильтоновости графов Кэли.")
        else:
            print("\nДостигнуто заданное ограничение по числу итераций.")
        report_name = f"gamilthon_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
        write_report(
            report_name,
            iterations,
            hamilton_count,
            exception_info,
            False,
            last_checked,
            skipped_non_generating,
            checked_hamiltonian,
            total_elapsed_seconds,
        )
        return


def single_mode():
    print("=== Диэдральный граф Кэли с тремя инволюциями ===")
    n = int(
        input(
            "Введите параметр n для диэдральной группы D_n (n должно быть четным): "
        )
    )
    if n % 2 != 0:
        raise SystemExit(
            "Для трёх инволюций с двумя коммутирующими требуется четное n."
        )

    k = int(
        input(
            f"Введите параметр k — смещение третьей инволюции (целое число от 1 до {n - 1}): "
        )
    )
    generators = build_dihedral_involution_generators(n, k)

    print("\nГенераторы (инволюции):")
    for index, generator in enumerate(generators, start=1):
        print(
            f"  g{index} = {element_to_str(generator)}  "
            f"(инволюция: {is_involution(generator, n)})"
        )
    non_commuting_pairs = find_non_commuting_pairs(generators, n)
    print(f"  Все пары генераторов коммутируют: {not non_commuting_pairs}")
    if non_commuting_pairs:
        pair_labels = ", ".join(f"g{i + 1}-g{j + 1}" for i, j in non_commuting_pairs)
        print(f"  Некоммутирующие пары: {pair_labels}")

    group = build_dihedral_group(n)
    closure = generate_subgroup(generators, n)

    print(f"\nРазмер группы D_{n} = {len(group)}")
    print(f"Размер порождённой подгруппы = {len(closure)}")
    if len(closure) == len(group):
        print("Генераторы порождают всю группу.")
    else:
        print("Генераторы не порождают всю группу.")

    graph = build_cayley_graph(group, generators, n)
    cycle = find_hamiltonian_cycle(graph)

    if cycle:
        print("\nГамильтонов цикл найден:")
        print([element_to_str(vertex) for vertex in cycle] + [element_to_str(cycle[0])])
    else:
        print("\nГамильтонов цикл не найден")

    draw_graph(graph, generators)


if __name__ == "__main__":
    print("Выберите режим:")
    print("1. Поиск исключений (бесконечный)")
    print("2. Поиск исключений (ограниченное число итераций)")
    print("3. Одиночная проверка")
    mode = input("Введите 1, 2 или 3 [1]: ").strip()

    if mode == "3":
        single_mode()
    elif mode == "2":
        try:
            max_iterations = int(input("Введите максимальное число итераций: "))
            if max_iterations <= 0:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "Число итераций должно быть положительным целым числом."
            )
        search_exceptions(max_iterations=max_iterations)
    else:
        search_exceptions()
