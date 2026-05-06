import csv
import datetime
import os
import time

from algorithm import element_to_str
from webgraph import format_web_route


CSV_FIELDNAMES = [
    "iteration",
    "n",
    "k",
    "group_size",
    "subgroup_size",
    "generators",
    "generator_conditions",
    "theorem_conditions",
    "generates",
    "cycle_checked",
    "hamiltonian",
    "cycle_length",
    "elapsed_seconds",
    "cycle",
    "web_route",
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


def build_experiment_row(iteration, result, elapsed_seconds):
    cycle = result.get("cycle")
    return {
        "iteration": iteration,
        "n": result["n"],
        "k": result["k"],
        "group_size": result["group_size"],
        "subgroup_size": result["subgroup_size"],
        "generators": format_generators(result["generators"]),
        "generator_conditions": result["generator_conditions"],
        "theorem_conditions": result["theorem_conditions"],
        "generates": result["generates"],
        "cycle_checked": result["cycle_checked"],
        "hamiltonian": result["hamiltonian"],
        "cycle_length": len(cycle) if cycle else 0,
        "elapsed_seconds": f"{elapsed_seconds:.6f}",
        "cycle": format_element_route(cycle),
        "web_route": format_web_route(cycle),
        "details": result["details"] or "",
    }


def run_csv_experiment(
    check_parameters,
    max_n,
    max_hamilton_check_n=None,
    max_iterations=None,
    output_path=None,
):
    if max_n < 2:
        raise ValueError("max_n должно быть не меньше 2.")

    if max_hamilton_check_n is None:
        max_hamilton_check_n = max_n

    if output_path is None:
        output_path = reports_path(default_experiment_filename())

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    summary = {
        "output_path": output_path,
        "iterations": 0,
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

        for n in range(2, max_n + 1, 2):
            for k in range(1, n):
                if max_iterations is not None and summary["iterations"] >= max_iterations:
                    return summary

                started_at = time.perf_counter()
                result = check_parameters(
                    n,
                    k,
                    max_hamilton_check_n=max_hamilton_check_n,
                )
                elapsed_seconds = time.perf_counter() - started_at

                summary["iterations"] += 1
                summary["total_elapsed_seconds"] += elapsed_seconds

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
                    )
                )

    return summary
