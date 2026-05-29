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
    find_hamiltonian_cycle_with_stats,
    generate_subgroup,
    is_involution,
)
from experiments import run_csv_experiment, write_experiment_summary
from generator_systems import (
    ALL_FAMILIES,
    FAMILY_LABELS,
    ROTATION_REFLECTION,
    THREE_INVOLUTIONS,
    TWO_REFLECTIONS,
    iter_generator_systems,
)
from webgraph import (
    build_vertex_page_map,
    format_page_route,
    format_web_route,
    load_page_paths,
)


def format_duration(total_seconds):
    total_seconds = float(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def reports_file_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    return os.path.join(reports_dir, filename)


def mode_filename_prefix(mode):
    return f"{sanitize_filename_part(mode)})_" if mode else ""


def default_graph_image_filename(n, k, filename_prefix=""):
    return f"{filename_prefix}gamilthon_graph_D{n}_k{k}_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"


def default_single_graph_image_filename(family, n, parameters, filename_prefix=""):
    return (
        f"{filename_prefix}gamilthon_graph_{sanitize_filename_part(family)}_"
        f"D{n}_{sanitize_filename_part(parameters)}_"
        f"{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
    )


def default_specific_graph_report_filename(family, n, parameters, filename_prefix=""):
    return (
        f"{filename_prefix}specific_graph_report_{sanitize_filename_part(family)}_"
        f"D{n}_{sanitize_filename_part(parameters)}_"
        f"{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    )


def default_examples_dirname(filename_prefix=""):
    return f"{filename_prefix}diploma_examples_{datetime.datetime.now():%Y%m%d_%H%M%S}"


def sanitize_filename_part(value):
    result = []
    for char in str(value):
        if char.isalnum():
            result.append(char)
        elif char in ("-", "_"):
            result.append(char)
        else:
            result.append("_")
    return "".join(result).strip("_") or "params"


def format_generators(generators):
    return ", ".join(element_to_str(generator) for generator in generators)


def format_generator_pairs(generators):
    return "{" + ", ".join(f"({k},{f})" for k, f in generators) + "}"


def format_yes_no(value):
    return "да" if value else "нет"


def read_families_selection(default_all=False):
    print("\nВыберите семейство систем порождающих:")
    print("1. Три инволюции с коммутирующей парой")
    print("2. Поворот, обратный поворот и отражение")
    print("3. Две отражающие симметрии")
    print("4. Все семейства")
    default_label = "4" if default_all else "1"
    family_mode = input(f"Введите 1, 2, 3 или 4 [{default_label}]: ").strip()
    if not family_mode:
        family_mode = default_label

    if family_mode == "2":
        return [ROTATION_REFLECTION]
    if family_mode == "3":
        return [TWO_REFLECTIONS]
    if family_mode == "4":
        return ALL_FAMILIES
    return [THREE_INVOLUTIONS]


def read_single_family_selection():
    print("\nВыберите семейство системы порождающих:")
    print("1. Три инволюции с коммутирующей парой")
    print("2. Поворот, обратный поворот и отражение")
    print("3. Две отражающие симметрии")
    family_mode = input("Введите 1, 2 или 3 [1]: ").strip()
    if not family_mode:
        family_mode = "1"

    if family_mode == "2":
        return ROTATION_REFLECTION
    if family_mode == "3":
        return TWO_REFLECTIONS
    return THREE_INVOLUTIONS


def read_custom_generators(n):
    try:
        generator_count = int(input("Введите количество элементов в S: ").strip())
    except ValueError:
        raise SystemExit("Количество элементов в S должно быть положительным целым числом.")

    if generator_count <= 0:
        raise SystemExit("Количество элементов в S должно быть положительным целым числом.")

    generators = []
    for index in range(1, generator_count + 1):
        raw_value = input(
            f"Введите g{index} в виде пары k,f, например 0,1 или (0,1): "
        ).strip()
        normalized = (
            raw_value
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "")
        )
        parts = normalized.split(",")
        if len(parts) != 2:
            raise SystemExit(
                "Каждый элемент S должен быть задан парой k,f, например 0,1."
            )

        try:
            k, f = (int(parts[0]), int(parts[1]))
        except ValueError:
            raise SystemExit("Значения k и f должны быть целыми числами.")

        if not 0 <= k < n:
            raise SystemExit(f"Значение k должно быть целым числом от 0 до {n - 1}.")
        if f not in (0, 1):
            raise SystemExit("Значение f должно быть равно 0 или 1.")

        generators.append((k, f))

    return generators


def read_optional_page_paths():
    try:
        filename = input(
            "Введите путь к файлу со страницами или оставьте пустым: "
        ).strip()
    except EOFError:
        return None

    if not filename:
        return None

    try:
        page_paths = load_page_paths(filename)
    except OSError as error:
        raise SystemExit(f"Не удалось прочитать файл со страницами: {error}")
    except ValueError as error:
        raise SystemExit(str(error))

    print(f"Загружено страниц: {len(page_paths)}")
    return page_paths


def read_optional_time_limit():
    try:
        raw_limit = input(
            "Введите лимит времени на поиск цикла в секундах или оставьте пустым: "
        ).strip()
    except EOFError:
        return None

    if not raw_limit:
        return None

    try:
        time_limit = float(raw_limit)
        if time_limit <= 0:
            raise ValueError
    except ValueError:
        raise SystemExit("Лимит времени должен быть положительным числом.")

    return time_limit


def format_cycle_status(status):
    labels = {
        "found": "найден",
        "not_found": "не найден",
        "skipped": "пропущен",
        "timeout": "превышен лимит времени",
    }
    return labels.get(status, status)


def format_element_route(cycle):
    if not cycle:
        return ""
    closed_cycle = cycle + [cycle[0]]
    return " -> ".join(element_to_str(vertex) for vertex in closed_cycle)


def format_optional_page_route(n, cycle, page_paths):
    if not cycle or not page_paths:
        return ""
    group = build_dihedral_group(n)
    if len(page_paths) < len(group):
        return ""
    vertex_page_map = build_vertex_page_map(group, page_paths)
    return format_page_route(cycle, vertex_page_map)


def find_non_commuting_pairs(generators, n):
    non_commuting_pairs = []
    for i in range(len(generators)):
        for j in range(i + 1, len(generators)):
            if not commutes(generators[i], generators[j], n):
                non_commuting_pairs.append((i, j))
    return non_commuting_pairs


def format_non_commuting_pairs(generators, n):
    non_commuting_pairs = find_non_commuting_pairs(generators, n)
    if not non_commuting_pairs:
        return "нет"
    return ", ".join(f"g{i + 1}-g{j + 1}" for i, j in non_commuting_pairs)


def format_first_pair_commuting(result):
    if result["generator_count"] < 2:
        return "не применимо"
    return format_yes_no(result["commuting_pair"])


def format_adjacency_list(graph):
    lines = []
    for vertex, neighbors in graph.items():
        neighbor_labels = ", ".join(element_to_str(neighbor) for neighbor in neighbors)
        lines.append(f"{element_to_str(vertex)}: [{neighbor_labels}]")
    return lines


def build_adjacency_matrix(vertices, graph):
    rows = []
    for vertex in vertices:
        neighbors = set(graph.get(vertex, []))
        rows.append([1 if candidate in neighbors else 0 for candidate in vertices])
    return rows


def format_adjacency_matrix(vertices, graph):
    matrix = build_adjacency_matrix(vertices, graph)
    if not matrix:
        return "[]"
    row_lines = [str(row) for row in matrix]
    return "[\n  " + ",\n  ".join(row_lines) + "\n]"


def draw_graph(graph, generators, cycle=None, output_path=None, show=True):
    graph_view = nx.DiGraph()
    labels = {}
    edge_colors = ["#d1495b", "#2f9e44", "#2458b3"]
    edge_groups = [set() for _ in generators]

    for node, neighbors in graph.items():
        labels[node] = element_to_str(node)
        for index, neighbor in enumerate(neighbors):
            graph_view.add_edge(node, neighbor)
            if index < len(edge_groups):
                edge_groups[index].add((node, neighbor))

    cycle_edges = set()
    if cycle:
        closed_cycle = cycle + [cycle[0]]
        cycle_edges = {
            (closed_cycle[index], closed_cycle[index + 1])
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
        title="Генераторы",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    plt.axis("off")
    plt.tight_layout(rect=(0, 0, 0.82, 1))

    if output_path:
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()

    plt.close()
    return output_path


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
    timed_out_hamiltonian=0,
    max_hamilton_time_seconds=None,
):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Отчет создан: {now}",
        f"Итераций выполнено: {iterations}",
        f"Общее время вычислений: {format_duration(total_elapsed_seconds)}",
        f"Пропущено систем вне условий теоремы: {skipped_invalid_conditions}",
        f"Пропущено непорождающих систем: {skipped_non_generating}",
        f"Проверено на гамильтоновость: {checked_hamiltonian}",
        f"С гамильтоновым циклом: {hamilton_count}",
        f"Превышен лимит времени поиска цикла: {timed_out_hamiltonian}",
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

    if max_hamilton_check_n is not None:
        lines.append(
            f"Проверка гамильтонова цикла выполнялась для n <= {max_hamilton_check_n}."
        )
    if max_hamilton_time_seconds is not None:
        lines.append(
            "Лимит времени на один поиск гамильтонова цикла: "
            f"{max_hamilton_time_seconds:.3f} с."
        )

    if os.path.isabs(filename):
        filename_full = filename
    else:
        filename_full = reports_file_path(filename)

    dirname = os.path.dirname(filename_full)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename_full, "w", encoding="utf-8") as report:
        report.write("\n".join(lines))

    print(f"Отчет сохранен в {filename_full}")


def write_specific_graph_report(
    filename,
    result,
    graph,
    image_path,
):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = result["n"]
    group = build_dihedral_group(n)
    generators = result["generators"]
    family = result["family"]
    family_label = FAMILY_LABELS.get(family, "пользовательское множество S")
    generator_labels = [element_to_str(generator) for generator in generators]

    lines = [
        "Отчет по конкретному графу Кэли",
        f"Отчет создан: {now}",
        "",
        "Исходные данные",
        f"Семейство системы порождающих: {family_label}",
        f"Группа: D_{n}",
        f"Порядок группы: {result['group_size']}",
        f"Граф Кэли: Cay(D_{n}, S)",
        f"Параметры: {result['parameters']}",
        f"Порождающие в виде пар (k,f): S = {format_generator_pairs(generators)}",
        f"Порождающие в обозначениях группы: S = {{{', '.join(generator_labels)}}}",
        "",
        "Свойства системы порождающих",
        f"Количество генераторов: {result['generator_count']}",
        f"Генераторы попарно различны: {format_yes_no(result['distinct_generators'])}",
        f"Система содержит нейтральный элемент: {format_yes_no(result['contains_identity'])}",
        f"Все генераторы являются инволюциями: {format_yes_no(all(result['involutions']))}",
        f"Первые два генератора коммутируют: {format_first_pair_commuting(result)}",
        f"Некоммутирующие пары генераторов: {format_non_commuting_pairs(generators, n)}",
        f"Размер порожденной подгруппы: {result['subgroup_size']}",
        f"Система порождает всю группу D_{n}: {format_yes_no(result['generates'])}",
        f"Условия теоремы о трех инволюциях выполнены: {format_yes_no(result['theorem_conditions'])}",
        "",
        "Построение графа и поиск гамильтонова цикла",
        f"Количество вершин: {len(group)}",
        f"Количество переходов в списке смежности: {len(group) * len(generators)}",
        f"Поиск выполнялся: {format_yes_no(result['cycle_checked'])}",
        f"Статус поиска: {format_cycle_status(result['cycle_status'])}",
        f"Просмотрено состояний поиска: {result['cycle_states_checked']}",
        f"Время поиска: {result['cycle_elapsed_seconds']:.6f} с",
    ]

    if result["max_hamilton_time_seconds"] is not None:
        lines.append(
            f"Лимит времени: {result['max_hamilton_time_seconds']:.3f} с"
        )

    if result["cycle"]:
        lines.extend(
            [
                f"Найденный гамильтонов цикл: {format_element_route(result['cycle'])}",
                f"Модельный веб-маршрут: {format_web_route(result['cycle'])}",
            ]
        )
    elif result["cycle_status"] == "not_found":
        lines.append("Найденный гамильтонов цикл: не найден.")
    elif result["cycle_status"] == "timeout":
        lines.append("Найденный гамильтонов цикл: поиск остановлен по лимиту времени.")
    else:
        lines.append("Найденный гамильтонов цикл: отсутствует.")

    if result["details"]:
        lines.append(f"Детали проверки: {result['details']}")

    lines.extend(
        [
            "",
            "Файлы результата",
            f"Изображение графа: {image_path}",
            "",
            "Вершины графа в порядке строк и столбцов матрицы",
            "[" + ", ".join(element_to_str(vertex) for vertex in group) + "]",
            "",
            "Список смежности",
            *format_adjacency_list(graph),
            "",
            "Матрица смежности",
            format_adjacency_matrix(group, graph),
        ]
    )

    filename_full = filename if os.path.isabs(filename) else reports_file_path(filename)
    dirname = os.path.dirname(filename_full)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename_full, "w", encoding="utf-8") as report:
        report.write("\n".join(lines))

    return filename_full


def check_generator_system(
    n,
    generators,
    family="custom",
    parameters="",
    k="",
    max_hamilton_check_n=None,
    max_hamilton_time_seconds=None,
    require_three_involution_conditions=False,
    search_even_if_not_generating=False,
):
    group_size = 2 * n
    involutions = [is_involution(generator, n) for generator in generators]
    all_involutions = all(involutions)
    distinct_generators = len(set(generators)) == len(generators)
    commuting_pair = (
        len(generators) >= 2 and commutes(generators[0], generators[1], n)
    )
    contains_identity = any(generator == (0, 0) for generator in generators)
    family_conditions = distinct_generators and not contains_identity
    three_involution_conditions = (
        len(generators) == 3
        and all_involutions
        and distinct_generators
        and commuting_pair
    )
    generator_conditions = three_involution_conditions
    closure = generate_subgroup(generators, n)
    subgroup_size = len(closure)
    generates = len(closure) == group_size
    theorem_conditions = three_involution_conditions and generates
    hamiltonian = False
    cycle = None
    cycle_allowed_by_limit = max_hamilton_check_n is None or n <= max_hamilton_check_n
    cycle_checked = cycle_allowed_by_limit and (generates or search_even_if_not_generating)
    cycle_status = "skipped"
    cycle_states_checked = 0
    cycle_elapsed_seconds = 0.0
    details = []

    if not distinct_generators:
        details.append("Среди генераторов есть совпадающие элементы.")
    if contains_identity:
        details.append("Система содержит нейтральный элемент.")
    if require_three_involution_conditions:
        if len(generators) != 3:
            details.append("Для этой проверки требуется ровно три генератора.")
        if not all_involutions:
            details.append("Один или несколько генераторов не являются инволюцией.")
        if len(generators) >= 2 and not commuting_pair:
            details.append("Первые две инволюции g1 и g2 не коммутируют.")
    if not generates:
        details.append(f"Генераторы порождают подгруппу размера {subgroup_size}, а не D_{n}.")

    if cycle_checked:
        graph = build_cayley_graph(build_dihedral_group(n), generators, n)
        cycle_result = find_hamiltonian_cycle_with_stats(
            graph,
            time_limit_seconds=max_hamilton_time_seconds,
        )
        cycle = cycle_result["cycle"]
        cycle_status = cycle_result["status"]
        cycle_states_checked = cycle_result["states_checked"]
        cycle_elapsed_seconds = cycle_result["elapsed_seconds"]
        hamiltonian = cycle_status == "found"

    return {
        "n": n,
        "k": k,
        "family": family,
        "parameters": parameters,
        "generators": generators,
        "generator_count": len(generators),
        "group_size": group_size,
        "subgroup_size": subgroup_size,
        "involutions": involutions,
        "distinct_generators": distinct_generators,
        "contains_identity": contains_identity,
        "commuting_pair": commuting_pair,
        "family_conditions": family_conditions,
        "three_involution_conditions": three_involution_conditions,
        "generator_conditions": generator_conditions,
        "theorem_conditions": theorem_conditions,
        "generates": generates,
        "hamiltonian": hamiltonian,
        "cycle_checked": cycle_checked,
        "cycle_status": cycle_status,
        "cycle_states_checked": cycle_states_checked,
        "cycle_elapsed_seconds": cycle_elapsed_seconds,
        "max_hamilton_time_seconds": max_hamilton_time_seconds,
        "cycle": cycle,
        "details": " ".join(details) if details else None,
    }


def check_parameters(
    n,
    k,
    max_hamilton_check_n=None,
    max_hamilton_time_seconds=None,
):
    generators = build_dihedral_involution_generators(n, k)
    return check_generator_system(
        n,
        generators,
        family=THREE_INVOLUTIONS,
        parameters=f"k={k}",
        k=k,
        max_hamilton_check_n=max_hamilton_check_n,
        max_hamilton_time_seconds=max_hamilton_time_seconds,
        require_three_involution_conditions=True,
    )


def is_representative_example(result):
    if not result["family_conditions"]:
        return False
    if not result["generates"] or not result["hamiltonian"] or not result["cycle"]:
        return False
    if result["family"] == THREE_INVOLUTIONS:
        return result["theorem_conditions"]
    return True


def build_examples_summary_lines(
    examples,
    missing_families,
    page_paths_count,
    target_n=None,
):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Дипломные примеры графов Кэли",
        f"Отчет создан: {now}",
        f"Количество найденных примеров: {len(examples)}",
        f"Страниц для сопоставления: {page_paths_count}",
    ]
    if target_n is not None:
        lines.append(f"Выбранное n: {target_n}")
    lines.append("")

    if missing_families:
        missing_labels = [
            FAMILY_LABELS.get(family, family)
            for family in missing_families
        ]
        lines.append("Не удалось подобрать примеры для семейств:")
        lines.extend(f"- {label}" for label in missing_labels)
        lines.append("")

    for index, example in enumerate(examples, start=1):
        result = example["result"]
        lines.extend(
            [
                f"Пример {index}",
                f"Семейство: {FAMILY_LABELS.get(result['family'], result['family'])}",
                f"n = {result['n']}",
                f"Параметры: {result['parameters']}",
                f"Генераторы: {format_generators(result['generators'])}",
                f"Размер группы: {result['group_size']}",
                f"Размер порожденной подгруппы: {result['subgroup_size']}",
                f"Статус поиска цикла: {format_cycle_status(result['cycle_status'])}",
                f"Просмотрено состояний поиска: {result['cycle_states_checked']}",
                f"Время поиска цикла: {result['cycle_elapsed_seconds']:.6f} с",
                f"Изображение: {example['image_path']}",
                f"Гамильтонов цикл: {format_element_route(result['cycle'])}",
                f"Модельный веб-маршрут: {format_web_route(result['cycle'])}",
            ]
        )
        if example["page_route"]:
            lines.append(f"Маршрут по страницам: {example['page_route']}")
        lines.append("")

    return lines


def generate_diploma_examples(
    max_n,
    families=None,
    max_hamilton_check_n=None,
    max_hamilton_time_seconds=None,
    page_paths=None,
    output_dir=None,
    show=False,
    target_n=None,
    filename_prefix="",
):
    if max_n < 2:
        raise ValueError("max_n должно быть не меньше 2.")
    if target_n is not None:
        if target_n < 2:
            raise ValueError("target_n должно быть не меньше 2.")
        if target_n > max_n:
            raise ValueError("target_n не должно быть больше max_n.")
    if max_hamilton_check_n is None:
        max_hamilton_check_n = max_n
    if families is None:
        families = ALL_FAMILIES
    if output_dir is None:
        output_dir = reports_file_path(default_examples_dirname(filename_prefix))

    os.makedirs(output_dir, exist_ok=True)
    selected = {}
    systems_iterator = iter_generator_systems(max_n, families=families)
    if target_n is not None:
        systems_iterator = (
            (n, system)
            for n, system in systems_iterator
            if n == target_n
        )

    for n, system in systems_iterator:
        family = system["family"]
        if family in selected:
            continue

        result = check_generator_system(
            n,
            system["generators"],
            family=family,
            parameters=system["parameters"],
            k=system["k"],
            max_hamilton_check_n=max_hamilton_check_n,
            max_hamilton_time_seconds=max_hamilton_time_seconds,
            require_three_involution_conditions=(family == THREE_INVOLUTIONS),
        )
        if not is_representative_example(result):
            continue

        filename = (
            f"{filename_prefix}{sanitize_filename_part(family)}_"
            f"D{n}_{sanitize_filename_part(system['parameters'])}.png"
        )
        image_path = os.path.join(output_dir, filename)
        graph = build_cayley_graph(
            build_dihedral_group(n),
            result["generators"],
            n,
        )
        draw_graph(
            graph,
            result["generators"],
            cycle=result["cycle"],
            output_path=image_path,
            show=show,
        )

        selected[family] = {
            "result": result,
            "image_path": image_path,
            "page_route": format_optional_page_route(
                n,
                result["cycle"],
                page_paths,
            ),
        }

        if len(selected) == len(families):
            break

    examples = [selected[family] for family in families if family in selected]
    missing_families = [family for family in families if family not in selected]
    summary_path = os.path.join(output_dir, f"{filename_prefix}examples_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(
            "\n".join(
                build_examples_summary_lines(
                    examples,
                    missing_families,
                    len(page_paths) if page_paths else 0,
                    target_n=target_n,
                )
            )
        )

    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "examples": examples,
        "missing_families": missing_families,
    }


def search_exceptions(max_iterations=None, run_mode="1"):
    filename_prefix = mode_filename_prefix(run_mode)
    print("=== Начинаем поиск исключений ===")
    raw_max_n = input(
        "Введите максимальное n для проверки гамильтонова цикла "
        "или оставьте пустым без ограничения: "
    ).strip()
    if raw_max_n:
        try:
            max_n = int(raw_max_n)
            if max_n < 2:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "Максимальное n должно быть целым числом не меньше 2."
            )
    else:
        max_n = None

    if max_n is None:
        print("Проверка гамильтонова цикла будет выполняться без ограничения по n.")
    else:
        print(
            f"Проверка гамильтонова цикла будет выполняться для n <= {max_n}"
        )
    max_hamilton_time_seconds = read_optional_time_limit()
    if max_hamilton_time_seconds is not None:
        print(
            "Лимит времени на один поиск гамильтонова цикла: "
            f"{max_hamilton_time_seconds:.3f} с"
        )

    iterations = 0
    hamilton_count = 0
    timed_out_hamiltonian = 0
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
                params = check_parameters(
                    n,
                    k,
                    max_hamilton_check_n=max_n,
                    max_hamilton_time_seconds=max_hamilton_time_seconds,
                )
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
                    print(
                        "  Статус поиска цикла: "
                        f"{format_cycle_status(params['cycle_status'])}; "
                        f"состояний: {params['cycle_states_checked']}; "
                        f"время поиска: {params['cycle_elapsed_seconds']:.6f} с."
                    )
                    if params["hamiltonian"]:
                        hamilton_count += 1
                        last_hamiltonian = params
                        print("  Проверка гамильтонова цикла: найден.")
                        print(f"  Цикл: {format_element_route(params['cycle'])}")
                        print(f"  Веб-маршрут: {format_web_route(params['cycle'])}")
                    elif params["cycle_status"] == "timeout":
                        timed_out_hamiltonian += 1
                        print(
                            "  Поиск остановлен по лимиту времени; "
                            "это не считается контрпримером."
                        )
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
                    if max_n is not None:
                        print(
                            f"  Проверка гамильтонова цикла пропущена для n={n} "
                            f"(n > {max_n})."
                        )

            n += 2
    except KeyboardInterrupt:
        print("\nПользователь остановил поиск.")
        report_name = f"{filename_prefix}gamilthon_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
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
            timed_out_hamiltonian=timed_out_hamiltonian,
            max_hamilton_time_seconds=max_hamilton_time_seconds,
        )
        return
    except StopIteration:
        if exception_info:
            print("\nНайден контрпример к гипотезе о гамильтоновости графов Кэли.")
        else:
            print("\nДостигнуто заданное ограничение по числу итераций.")
        report_name = f"{filename_prefix}gamilthon_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
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
            timed_out_hamiltonian=timed_out_hamiltonian,
            max_hamilton_time_seconds=max_hamilton_time_seconds,
        )
        return


def single_mode(run_mode="3"):
    filename_prefix = mode_filename_prefix(run_mode)
    print("=== Одиночная проверка графа Кэли диэдральной группы ===")
    n = int(
        input(
            "Введите параметр n для диэдральной группы D_n: "
        )
    )
    if n < 2:
        raise SystemExit("n должно быть целым числом не меньше 2.")

    family = read_single_family_selection()
    if family == THREE_INVOLUTIONS:
        if n % 2 != 0:
            print(
                "\nСемейство трех инволюций с двумя коммутирующими пропущено "
                f"для D_{n}, поскольку n является нечетным."
            )
            print(
                "Для этого семейства требуется элемент (n/2,1), который существует "
                "только при четном n."
            )
            return

        k = int(
            input(
                f"Введите параметр k — смещение третьей инволюции (целое число от 1 до {n - 1}): "
            )
        )
        if not 1 <= k <= n - 1:
            raise SystemExit(f"k должно быть целым числом от 1 до {n - 1}.")
        generators = build_dihedral_involution_generators(n, k)
        parameters = f"k={k}"
        require_three_involution_conditions = True
    elif family == ROTATION_REFLECTION:
        a = int(
            input(
                f"Введите параметр a — смещение поворота (целое число от 1 до {n - 1}): "
            )
        )
        if not 1 <= a <= n - 1:
            raise SystemExit(f"a должно быть целым числом от 1 до {n - 1}.")
        b = int(
            input(
                f"Введите параметр b — смещение отражения (целое число от 0 до {n - 1}): "
            )
        )
        if not 0 <= b <= n - 1:
            raise SystemExit(f"b должно быть целым числом от 0 до {n - 1}.")
        generators = [(a, 0), ((-a) % n, 0), (b, 1)]
        parameters = f"a={a}, b={b}"
        k = ""
        require_three_involution_conditions = False
    else:
        a = int(
            input(
                f"Введите параметр a — первое отражение (целое число от 0 до {n - 1}): "
            )
        )
        b = int(
            input(
                f"Введите параметр b — второе отражение (целое число от 0 до {n - 1}, b>a): "
            )
        )
        if not 0 <= a < b < n:
            raise SystemExit(f"Параметры должны удовлетворять условию 0 <= a < b < {n}.")
        generators = [(a, 1), (b, 1)]
        parameters = f"a={a}, b={b}"
        k = ""
        require_three_involution_conditions = False

    max_hamilton_time_seconds = read_optional_time_limit()
    params = check_generator_system(
        n,
        generators,
        family=family,
        parameters=parameters,
        k=k,
        max_hamilton_time_seconds=max_hamilton_time_seconds,
        require_three_involution_conditions=require_three_involution_conditions,
    )
    generators = params["generators"]

    print(f"\nСемейство: {FAMILY_LABELS.get(family, family)}")
    print(f"Параметры: {parameters}")
    print("\nГенераторы:")
    for index, generator in enumerate(generators, start=1):
        print(
            f"  g{index} = {element_to_str(generator)}  "
            f"(инволюция: {is_involution(generator, n)})"
        )

    if family == THREE_INVOLUTIONS:
        print("\nУсловия теоремы о трех инволюциях:")
        print(f"  Все генераторы являются инволюциями: {all(params['involutions'])}")
        print(f"  Генераторы попарно различны: {params['distinct_generators']}")
        print(f"  g1 и g2 коммутируют: {params['commuting_pair']}")
        print(f"  Базовые условия на генераторы выполнены: {params['generator_conditions']}")
        print(f"  Полные условия теоремы с учетом порождения: {params['theorem_conditions']}")
        if params["details"]:
            print(f"  Детали: {params['details']}")
    else:
        print("\nЭто семейство не относится к случаю трех инволюций с коммутирующей парой.")
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

    print("\nПоиск гамильтонова цикла:")
    print(f"  Статус: {format_cycle_status(params['cycle_status'])}")
    print(f"  Просмотрено состояний: {params['cycle_states_checked']}")
    print(f"  Время поиска: {params['cycle_elapsed_seconds']:.6f} с")
    if params["max_hamilton_time_seconds"] is not None:
        print(
            "  Лимит времени: "
            f"{params['max_hamilton_time_seconds']:.3f} с"
        )

    graph = build_cayley_graph(group, generators, n)
    cycle = params["cycle"]
    page_paths = read_optional_page_paths()
    vertex_page_map = None
    if page_paths:
        try:
            vertex_page_map = build_vertex_page_map(group, page_paths)
        except ValueError as error:
            raise SystemExit(str(error))

    if cycle:
        print("\nГамильтонов цикл найден:")
        print(format_element_route(cycle))
        print("\nТот же цикл как маршрут веб-страниц:")
        print(format_web_route(cycle))
        if vertex_page_map:
            print("\nТот же цикл как маршрут по страницам из файла:")
            print(format_page_route(cycle, vertex_page_map))
    else:
        print("\nГамильтонов цикл не найден")

    image_path = reports_file_path(
        default_single_graph_image_filename(family, n, parameters, filename_prefix)
    )
    draw_graph(graph, generators, cycle=cycle, output_path=image_path)
    print(f"\nИзображение графа сохранено в {image_path}")


def specific_graph_mode(run_mode="6"):
    filename_prefix = mode_filename_prefix(run_mode)
    print("=== Построение конкретного графа Кэли по заданному множеству S ===")
    n = int(
        input(
            "Введите параметр n для диэдральной группы D_n: "
        )
    )
    if n < 2:
        raise SystemExit("n должно быть целым числом не меньше 2.")

    generators = read_custom_generators(n)
    family = "custom_generators"
    parameters = f"S={format_generator_pairs(generators)}"
    max_hamilton_time_seconds = read_optional_time_limit()

    result = check_generator_system(
        n,
        generators,
        family=family,
        parameters=parameters,
        k="",
        max_hamilton_time_seconds=max_hamilton_time_seconds,
        require_three_involution_conditions=False,
        search_even_if_not_generating=True,
    )
    group = build_dihedral_group(n)
    graph = build_cayley_graph(group, generators, n)

    image_path = reports_file_path(
        default_single_graph_image_filename(
            family,
            n,
            parameters,
            filename_prefix,
        )
    )
    draw_graph(graph, generators, cycle=result["cycle"], output_path=image_path)

    report_path = write_specific_graph_report(
        default_specific_graph_report_filename(
            family,
            n,
            parameters,
            filename_prefix,
        ),
        result,
        graph,
        image_path,
    )

    print("\nКонкретный граф построен.")
    print(f"Группа: D_{n}, порядок группы: {result['group_size']}")
    print(f"Порождающие в виде пар (k,f): {format_generator_pairs(generators)}")
    print(f"Порождающие в обозначениях: {{{format_generators(generators)}}}")
    print(f"Система порождает всю группу: {format_yes_no(result['generates'])}")
    print(f"Статус поиска цикла: {format_cycle_status(result['cycle_status'])}")
    if result["cycle"]:
        print(f"Найденный цикл: {format_element_route(result['cycle'])}")
    else:
        print("Гамильтонов цикл не найден или не был получен.")
    print(f"Отчет сохранен в {report_path}")
    print(f"Изображение графа сохранено в {image_path}")


def experiment_mode(run_mode="4"):
    filename_prefix = mode_filename_prefix(run_mode)
    print("=== CSV-эксперимент по графам Кэли диэдральных групп ===")
    try:
        max_n = int(input("Введите максимальное n для перебора: "))
        if max_n < 2:
            raise ValueError
    except ValueError:
        raise SystemExit("Максимальное n должно быть целым числом не меньше 2.")

    raw_max_hamilton_n = input(
        f"Введите максимальное n для поиска гамильтонова цикла [{max_n}]: "
    ).strip()
    if raw_max_hamilton_n:
        try:
            max_hamilton_n = int(raw_max_hamilton_n)
            if max_hamilton_n < 2:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "Максимальное n для поиска цикла должно быть целым числом не меньше 2."
            )
    else:
        max_hamilton_n = max_n

    families = read_families_selection()
    print("Перебор выполняется для всех n от 2 до заданного максимального n.")
    if THREE_INVOLUTIONS in families:
        print(
            "Для нечетных n семейство трех инволюций с коммутирующей парой "
            "не формируется; остальные выбранные семейства проверяются."
        )
    page_paths = read_optional_page_paths()
    max_hamilton_time_seconds = read_optional_time_limit()

    raw_max_iterations = input(
        "Введите максимальное число итераций или оставьте пустым: "
    ).strip()
    if raw_max_iterations:
        try:
            max_iterations = int(raw_max_iterations)
            if max_iterations <= 0:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "Число итераций должно быть положительным целым числом."
            )
    else:
        max_iterations = None

    summary = run_csv_experiment(
        check_generator_system,
        max_n=max_n,
        max_hamilton_check_n=max_hamilton_n,
        max_hamilton_time_seconds=max_hamilton_time_seconds,
        max_iterations=max_iterations,
        families=families,
        page_paths=page_paths,
        filename_prefix=filename_prefix,
    )
    summary_path = write_experiment_summary(summary)

    print("\nЭксперимент завершен.")
    print(f"CSV сохранен в {summary['output_path']}")
    print(f"Сводный отчет сохранен в {summary_path}")
    print(
        "Семейства: "
        + ", ".join(FAMILY_LABELS[family] for family in summary["families"])
    )
    if summary["page_paths_count"]:
        print(f"Страниц для сопоставления: {summary['page_paths_count']}")
    print(f"Итераций выполнено: {summary['iterations']}")
    print(
        "Систем без дубликатов и нейтрального элемента: "
        f"{summary['family_conditions_count']}"
    )
    print(
        "Систем с условиями теоремы на генераторы: "
        f"{summary['generator_conditions_count']}"
    )
    print(f"Порождающих систем: {summary['generating_count']}")
    print(f"Систем, удовлетворяющих условиям теоремы: {summary['theorem_conditions_count']}")
    print(f"Проверено на гамильтоновость: {summary['checked_hamiltonian_count']}")
    print(f"С гамильтоновым циклом: {summary['hamiltonian_count']}")
    print(f"Без найденного цикла: {summary['not_found_count']}")
    print(f"С превышением лимита времени: {summary['timeout_count']}")
    print(f"С пропущенной проверкой цикла: {summary['skipped_count']}")
    print(f"Общее время: {format_duration(summary['total_elapsed_seconds'])}")


def diploma_examples_mode(run_mode="5"):
    filename_prefix = mode_filename_prefix(run_mode)
    print("=== Подготовка дипломных примеров ===")
    try:
        max_n = int(input("Введите максимальное n для поиска примеров: "))
        if max_n < 2:
            raise ValueError
    except ValueError:
        raise SystemExit("Максимальное n должно быть целым числом не меньше 2.")

    raw_target_n = input(
        "Введите конкретное n для примеров [пусто - искать до max_n]: "
    ).strip()
    target_n = None
    if raw_target_n:
        try:
            target_n = int(raw_target_n)
            if target_n < 2 or target_n > max_n:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "n для примеров должно быть целым числом от 2 до max_n."
            )

    default_hamilton_n = target_n if target_n is not None else max_n
    raw_max_hamilton_n = input(
        f"Введите максимальное n для поиска гамильтонова цикла [{default_hamilton_n}]: "
    ).strip()
    if raw_max_hamilton_n:
        try:
            max_hamilton_n = int(raw_max_hamilton_n)
            if max_hamilton_n < 2:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "Максимальное n для поиска цикла должно быть целым числом не меньше 2."
            )
    else:
        max_hamilton_n = default_hamilton_n
    if target_n is not None and max_hamilton_n < target_n:
        raise SystemExit(
            "Максимальное n для поиска цикла должно быть не меньше выбранного n."
        )

    families = read_families_selection(default_all=True)
    page_paths = read_optional_page_paths()
    max_hamilton_time_seconds = read_optional_time_limit()
    result = generate_diploma_examples(
        max_n=max_n,
        families=families,
        max_hamilton_check_n=max_hamilton_n,
        max_hamilton_time_seconds=max_hamilton_time_seconds,
        page_paths=page_paths,
        target_n=target_n,
        filename_prefix=filename_prefix,
    )

    print("\nДипломные примеры подготовлены.")
    print(f"Папка с примерами: {result['output_dir']}")
    print(f"Описание примеров: {result['summary_path']}")
    if target_n is not None:
        print(f"Примеры подбирались только для n = {target_n}")
    print(f"Найдено примеров: {len(result['examples'])}")
    if result["missing_families"]:
        missing = ", ".join(
            FAMILY_LABELS.get(family, family)
            for family in result["missing_families"]
        )
        print(f"Не найдены примеры для семейств: {missing}")


if __name__ == "__main__":
    print("Выберите режим:")
    print("0. Выход")
    print("1. Поиск исключений (бесконечный)")
    print("2. Поиск исключений (ограниченное число итераций)")
    print("3. Одиночная проверка")
    print("4. CSV-эксперимент")
    print("5. Подготовка дипломных примеров")
    print("6. Построение графа по заданному множеству S")
    mode = input("Введите 0, 1, 2, 3, 4, 5 или 6 [1]: ").strip()

    if not mode:
        mode = "1"

    if mode == "0":
        print("Программа завершена.")
    elif mode == "3":
        single_mode(run_mode=mode)
    elif mode == "4":
        experiment_mode(run_mode=mode)
    elif mode == "5":
        diploma_examples_mode(run_mode=mode)
    elif mode == "6":
        specific_graph_mode(run_mode=mode)
    elif mode == "2":
        try:
            max_iterations = int(input("Введите максимальное число итераций: "))
            if max_iterations <= 0:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "Число итераций должно быть положительным целым числом."
            )
        search_exceptions(max_iterations=max_iterations, run_mode=mode)
    else:
        search_exceptions(run_mode=mode)
