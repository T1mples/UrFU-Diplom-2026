import csv
import datetime
import os
import time

from algorithm import build_dihedral_group, element_to_str
from generator_systems import FAMILY_LABELS, THREE_INVOLUTIONS, iter_generator_systems
from webgraph import build_vertex_page_map, format_page_route, format_web_route


CSV_FIELDNAMES = [
    "iteration",
    "family",
    "parameters",
    "n",
    "k",
    "generator_count",
    "group_size",
    "subgroup_size",
    "generators",
    "family_conditions",
    "generator_conditions",
    "three_involution_conditions",
    "theorem_conditions",
    "generates",
    "cycle_checked",
    "hamiltonian",
    "cycle_length",
    "elapsed_seconds",
    "cycle",
    "web_route",
    "page_route",
    "details",
]


def default_experiment_filename():
    return f"gamilthon_experiment_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"


def default_summary_path(csv_path):
    dirname, filename = os.path.split(csv_path)
    stem = os.path.splitext(filename)[0]
    if stem.startswith("gamilthon_experiment"):
        stem = stem.replace("gamilthon_experiment", "gamilthon_summary", 1)
    else:
        stem = f"{stem}_summary"
    return os.path.join(dirname, f"{stem}.txt")


def reports_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    return os.path.join(reports_dir, filename)


def format_generators(generators):
    return ", ".join(element_to_str(generator) for generator in generators)


def format_element_route(cycle):
    if not cycle:
        return ""
    closed_cycle = cycle + [cycle[0]]
    return " -> ".join(element_to_str(vertex) for vertex in closed_cycle)


def empty_stats():
    return {
        "iterations": 0,
        "family_conditions_count": 0,
        "generator_conditions_count": 0,
        "theorem_conditions_count": 0,
        "generating_count": 0,
        "checked_hamiltonian_count": 0,
        "hamiltonian_count": 0,
        "total_elapsed_seconds": 0.0,
        "max_elapsed_seconds": 0.0,
    }


def update_stats(stats, result, elapsed_seconds):
    stats["iterations"] += 1
    stats["total_elapsed_seconds"] += elapsed_seconds
    stats["max_elapsed_seconds"] = max(stats["max_elapsed_seconds"], elapsed_seconds)

    if result["family_conditions"]:
        stats["family_conditions_count"] += 1
    if result["generator_conditions"]:
        stats["generator_conditions_count"] += 1
    if result["theorem_conditions"]:
        stats["theorem_conditions_count"] += 1
    if result["generates"]:
        stats["generating_count"] += 1
    if result["cycle_checked"]:
        stats["checked_hamiltonian_count"] += 1
    if result["hamiltonian"]:
        stats["hamiltonian_count"] += 1


def build_experiment_row(iteration, result, elapsed_seconds, page_paths=None):
    cycle = result.get("cycle")
    page_route = ""
    if cycle and page_paths and len(page_paths) >= result["group_size"]:
        vertex_page_map = build_vertex_page_map(
            build_dihedral_group(result["n"]),
            page_paths,
        )
        page_route = format_page_route(cycle, vertex_page_map)

    return {
        "iteration": iteration,
        "family": result["family"],
        "parameters": result["parameters"],
        "n": result["n"],
        "k": result["k"],
        "generator_count": result["generator_count"],
        "group_size": result["group_size"],
        "subgroup_size": result["subgroup_size"],
        "generators": format_generators(result["generators"]),
        "family_conditions": result["family_conditions"],
        "generator_conditions": result["generator_conditions"],
        "three_involution_conditions": result["three_involution_conditions"],
        "theorem_conditions": result["theorem_conditions"],
        "generates": result["generates"],
        "cycle_checked": result["cycle_checked"],
        "hamiltonian": result["hamiltonian"],
        "cycle_length": len(cycle) if cycle else 0,
        "elapsed_seconds": f"{elapsed_seconds:.6f}",
        "cycle": format_element_route(cycle),
        "web_route": format_web_route(cycle),
        "page_route": page_route,
        "details": result["details"] or "",
    }


def run_csv_experiment(
    check_generator_system,
    max_n,
    max_hamilton_check_n=None,
    max_iterations=None,
    output_path=None,
    families=None,
    page_paths=None,
):
    if max_n < 2:
        raise ValueError("max_n должно быть не меньше 2.")

    if max_hamilton_check_n is None:
        max_hamilton_check_n = max_n
    if families is None:
        families = [THREE_INVOLUTIONS]

    if output_path is None:
        output_path = reports_path(default_experiment_filename())

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    summary = {
        "output_path": output_path,
        "families": list(families),
        "page_paths_count": len(page_paths) if page_paths else 0,
        "family_stats": {},
        "n_stats": {},
        **empty_stats(),
    }

    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for n, system in iter_generator_systems(max_n, families=families):
            if max_iterations is not None and summary["iterations"] >= max_iterations:
                return summary

            started_at = time.perf_counter()
            result = check_generator_system(
                n,
                system["generators"],
                family=system["family"],
                parameters=system["parameters"],
                k=system["k"],
                max_hamilton_check_n=max_hamilton_check_n,
                require_three_involution_conditions=(
                    system["family"] == THREE_INVOLUTIONS
                ),
            )
            elapsed_seconds = time.perf_counter() - started_at

            update_stats(summary, result, elapsed_seconds)

            family_stats = summary["family_stats"].setdefault(
                result["family"],
                empty_stats(),
            )
            update_stats(family_stats, result, elapsed_seconds)

            n_stats = summary["n_stats"].setdefault(result["n"], empty_stats())
            update_stats(n_stats, result, elapsed_seconds)

            writer.writerow(
                build_experiment_row(
                    summary["iterations"],
                    result,
                    elapsed_seconds,
                    page_paths=page_paths,
                )
            )

    return summary


def format_average_elapsed(stats):
    if not stats["iterations"]:
        return "0.000000"
    return f"{stats['total_elapsed_seconds'] / stats['iterations']:.6f}"


def build_stats_line(label, stats):
    return (
        f"{label}: всего={stats['iterations']}; "
        f"без дубликатов={stats['family_conditions_count']}; "
        f"условия теоремы на генераторы={stats['generator_conditions_count']}; "
        f"порождают={stats['generating_count']}; "
        f"проверено на цикл={stats['checked_hamiltonian_count']}; "
        f"с циклом={stats['hamiltonian_count']}; "
        f"среднее время={format_average_elapsed(stats)} с; "
        f"максимальное время={stats['max_elapsed_seconds']:.6f} с"
    )


def build_experiment_summary_lines(summary):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    family_labels = [
        FAMILY_LABELS.get(family, family)
        for family in summary["families"]
    ]
    lines = [
        "Сводный отчет эксперимента",
        f"Отчет создан: {now}",
        f"CSV-файл: {summary['output_path']}",
        f"Семейства: {', '.join(family_labels)}",
        f"Страниц для сопоставления: {summary['page_paths_count']}",
        "",
        "Общая статистика:",
        build_stats_line("Все системы", summary),
        "",
        "Статистика по семействам:",
    ]

    for family in summary["families"]:
        stats = summary["family_stats"].get(family, empty_stats())
        label = FAMILY_LABELS.get(family, family)
        lines.append(build_stats_line(label, stats))

    lines.extend(["", "Статистика по n:"])
    for n in sorted(summary["n_stats"]):
        lines.append(build_stats_line(f"n={n}", summary["n_stats"][n]))

    return lines


def write_experiment_summary(summary, output_path=None):
    if output_path is None:
        output_path = default_summary_path(summary["output_path"])

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(build_experiment_summary_lines(summary)))

    summary["summary_path"] = output_path
    return output_path
