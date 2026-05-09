import csv
import datetime
import os
import time

from algorithm import build_dihedral_group, element_to_str
from generator_systems import THREE_INVOLUTIONS, iter_generator_systems
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
        "iterations": 0,
        "family_conditions_count": 0,
        "generator_conditions_count": 0,
        "theorem_conditions_count": 0,
        "generating_count": 0,
        "checked_hamiltonian_count": 0,
        "hamiltonian_count": 0,
        "total_elapsed_seconds": 0.0,
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

            summary["iterations"] += 1
            summary["total_elapsed_seconds"] += elapsed_seconds

            if result["family_conditions"]:
                summary["family_conditions_count"] += 1
            if result["generator_conditions"]:
                summary["generator_conditions_count"] += 1
            if result["theorem_conditions"]:
                summary["theorem_conditions_count"] += 1
            if result["generates"]:
                summary["generating_count"] += 1
            if result["cycle_checked"]:
                summary["checked_hamiltonian_count"] += 1
            if result["hamiltonian"]:
                summary["hamiltonian_count"] += 1

            writer.writerow(
                build_experiment_row(
                    summary["iterations"],
                    result,
                    elapsed_seconds,
                    page_paths=page_paths,
                )
            )

    return summary
