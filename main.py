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
from webgraph import format_web_route


def format_duration(total_seconds):
    total_seconds = float(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def format_element_route(cycle):
    if not cycle:
        return ""
    closed_cycle = cycle + [cycle[0]]
    return " -> ".join(element_to_str(vertex) for vertex in closed_cycle)


def find_non_commuting_pairs(generators, n):
    non_commuting_pairs = []
    for i in range(len(generators)):
        for j in range(i + 1, len(generators)):
            if not commutes(generators[i], generators[j], n):
                non_commuting_pairs.append((i, j))
    return non_commuting_pairs


def draw_graph(graph, generators, cycle=None):
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

    cycle_edges = set()
    if cycle:
        closed_cycle = cycle + [cycle[0]]
        cycle_edges = {
            frozenset((closed_cycle[index], closed_cycle[index + 1]))
            for index in range(len(cycle))
        }

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

    if cycle_edges:
        nx.draw_networkx_edges(
            graph_view,
            pos,
            edgelist=[tuple(edge) for edge in cycle_edges],
            arrows=True,
            arrowstyle="<|-|>",
            arrowsize=20,
            edge_color="#f08c00",
            width=4.0,
            min_source_margin=17,
            min_target_margin=17,
            connectionstyle="arc3,rad=0.08",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#f08c00",
                lw=4,
                marker=">",
                markersize=7,
                label="гамильтонов цикл",
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
    max_hamilton_check_n=None,
    skipped_invalid_conditions=0,
    last_hamiltonian=None,
):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Отчет создан: {now}",
        "Модель: веб-граф как граф Кэли диэдральной группы.",
        f"Итераций выполнено: {iterations}",
        f"Общее время вычислений: {format_duration(total_elapsed_seconds)}",
        f"Пропущено систем вне условий теоремы: {skipped_invalid_conditions}",
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

    if last_hamiltonian and last_hamiltonian.get("cycle"):
        lines.append("Последний найденный гамильтонов цикл:")
        lines.append(format_element_route(last_hamiltonian["cycle"]))
        lines.append("Веб-маршрут последнего найденного цикла:")
        lines.append(format_web_route(last_hamiltonian["cycle"]))

    if max_hamilton_check_n is not None:
        lines.append(
            f"Проверка гамильтонова цикла выполнялась для n <= {max_hamilton_check_n}."
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


def check_parameters(n, k, max_hamilton_check_n=None):
    generators = build_dihedral_involution_generators(n, k)
    group_size = 2 * n
    involutions = [is_involution(generator, n) for generator in generators]
    all_involutions = all(involutions)
    distinct_generators = len(set(generators)) == len(generators)
    commuting_pair = commutes(generators[0], generators[1], n)
    generator_conditions = all_involutions and distinct_generators and commuting_pair
    closure = generate_subgroup(generators, n)
    subgroup_size = len(closure)
    generates = len(closure) == group_size
    theorem_conditions = generator_conditions and generates
    hamiltonian = False
    cycle = None
    cycle_allowed_by_limit = max_hamilton_check_n is None or n <= max_hamilton_check_n
    cycle_checked = cycle_allowed_by_limit and generates
    details = []

    if not all_involutions:
        details.append("Один или несколько генераторов не являются инволюцией.")
    if not distinct_generators:
        details.append("Среди трех генераторов есть совпадающие элементы.")
    if not commuting_pair:
        details.append("Первые две инволюции g1 и g2 не коммутируют.")
    if not generates:
        details.append(f"Генераторы порождают подгруппу размера {subgroup_size}, а не D_{n}.")

    if cycle_checked and generates:
        graph = build_cayley_graph(build_dihedral_group(n), generators, n)
        cycle = find_hamiltonian_cycle(graph)
        hamiltonian = cycle is not None

    return {
        "n": n,
        "k": k,
        "generators": generators,
        "group_size": group_size,
        "subgroup_size": subgroup_size,
        "involutions": involutions,
        "distinct_generators": distinct_generators,
        "commuting_pair": commuting_pair,
        "generator_conditions": generator_conditions,
        "theorem_conditions": theorem_conditions,
        "generates": generates,
        "hamiltonian": hamiltonian,
        "cycle_checked": cycle_checked,
        "cycle": cycle,
        "details": " ".join(details) if details else None,
    }


def search_exceptions(max_iterations=None):
    print("=== Начинаем поиск исключений ===")
    try:
        max_n = int(
            input("Введите максимальное n для проверки гамильтонова цикла: ")
        )
    except ValueError:
        max_n = 10

    print(
        f"Проверка гамильтонова цикла будет выполняться для n <= {max_n}"
    )

    iterations = 0
    hamilton_count = 0
    skipped_invalid_conditions = 0
    skipped_non_generating = 0
    checked_hamiltonian = 0
    total_elapsed_seconds = 0.0
    exception_info = None
    last_hamiltonian = None
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
                params = check_parameters(n, k, max_hamilton_check_n=max_n)
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

                if not params["generator_conditions"]:
                    print(
                        "  Пропуск: система не удовлетворяет условиям теоремы "
                        "о трех инволюциях с коммутирующей парой."
                    )
                    if params["details"]:
                        print(f"  Детали: {params['details']}")
                    skipped_invalid_conditions += 1
                    continue

                if not params["generates"]:
                    print(f"  Пропуск: система не является порождающей для D_{n}.")
                    if params["details"]:
                        print(f"  Детали: {params['details']}")
                    skipped_non_generating += 1
                    continue

                if params["cycle_checked"]:
                    checked_hamiltonian += 1
                    if params["hamiltonian"]:
                        hamilton_count += 1
                        last_hamiltonian = params
                        print("  Проверка гамильтонова цикла: найден.")
                        print(f"  Цикл: {format_element_route(params['cycle'])}")
                        print(f"  Веб-маршрут: {format_web_route(params['cycle'])}")
                    else:
                        print("  Проверка гамильтонова цикла: не найден.")
                        exception_info = {
                            "check": "отсутствие гамильтонова цикла",
                            "n": n,
                            "k": k,
                            "generators": params["generators"],
                            "cycle": params["cycle"],
                            "details": "Цикл не найден при полном графе Кэли.",
                        }
                        raise StopIteration
                else:
                    print(
                        f"  Проверка гамильтонова цикла пропущена для n={n} "
                        f"(n > {max_n})."
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
            max_n,
            skipped_invalid_conditions,
            last_hamiltonian,
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
            max_n,
            skipped_invalid_conditions,
            last_hamiltonian,
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
    if not 1 <= k <= n - 1:
        raise SystemExit(f"k должно быть целым числом от 1 до {n - 1}.")

    params = check_parameters(n, k)
    generators = params["generators"]

    print("\nГенераторы (инволюции):")
    for index, generator in enumerate(generators, start=1):
        print(
            f"  g{index} = {element_to_str(generator)}  "
            f"(инволюция: {is_involution(generator, n)})"
        )

    print("\nУсловия теоремы о трех инволюциях:")
    print(f"  Все генераторы являются инволюциями: {all(params['involutions'])}")
    print(f"  Генераторы попарно различны: {params['distinct_generators']}")
    print(f"  g1 и g2 коммутируют: {params['commuting_pair']}")
    print(f"  Базовые условия на генераторы выполнены: {params['generator_conditions']}")
    print(f"  Полные условия теоремы с учетом порождения: {params['theorem_conditions']}")
    if params["details"]:
        print(f"  Детали: {params['details']}")

    non_commuting_pairs = find_non_commuting_pairs(generators, n)
    print(f"  Информационно: все пары генераторов коммутируют: {not non_commuting_pairs}")
    if non_commuting_pairs:
        pair_labels = ", ".join(f"g{i + 1}-g{j + 1}" for i, j in non_commuting_pairs)
        print(f"  Информационно: некоммутирующие пары: {pair_labels}")

    group = build_dihedral_group(n)

    print(f"\nРазмер группы D_{n} = {len(group)}")
    print(f"Размер порождённой подгруппы = {params['subgroup_size']}")
    if params["generates"]:
        print("Генераторы порождают всю группу.")
    else:
        print("Генераторы не порождают всю группу.")

    graph = build_cayley_graph(group, generators, n)
    cycle = params["cycle"]

    if cycle:
        print("\nГамильтонов цикл найден:")
        print(format_element_route(cycle))
        print("\nТот же цикл как маршрут веб-страниц:")
        print(format_web_route(cycle))
    else:
        print("\nГамильтонов цикл не найден")

    draw_graph(graph, generators, cycle=cycle)


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
