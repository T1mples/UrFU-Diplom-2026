import csv
import os
import tempfile
import unittest

import main
from algorithm import (
    build_cayley_graph,
    build_dihedral_group,
    build_dihedral_involution_generators,
    commutes,
)
from experiments import run_csv_experiment
from generator_systems import ROTATION_REFLECTION, THREE_INVOLUTIONS
from webgraph import (
    build_vertex_page_map,
    cycle_to_web_route,
    format_page_route,
    load_page_paths,
)


class CoreLogicTest(unittest.TestCase):
    def test_theorem_pair_commutes_but_third_may_not(self):
        generators = build_dihedral_involution_generators(4, 1)

        self.assertTrue(commutes(generators[0], generators[1], 4))
        self.assertFalse(commutes(generators[0], generators[2], 4))
        self.assertFalse(commutes(generators[1], generators[2], 4))

    def test_check_parameters_accepts_non_commuting_third_generator(self):
        result = main.check_parameters(4, 1, max_hamilton_check_n=4)

        self.assertTrue(result["generator_conditions"])
        self.assertTrue(result["generates"])
        self.assertTrue(result["theorem_conditions"])
        self.assertTrue(result["hamiltonian"])
        self.assertIsNotNone(result["cycle"])

    def test_duplicate_generator_is_reported(self):
        result = main.check_parameters(4, 2, max_hamilton_check_n=4)

        self.assertFalse(result["generator_conditions"])
        self.assertFalse(result["theorem_conditions"])
        self.assertIn("совпадающие элементы", result["details"])

    def test_generic_rotation_reflection_system(self):
        result = main.check_generator_system(
            4,
            [(1, 0), (3, 0), (0, 1)],
            family=ROTATION_REFLECTION,
            parameters="a=1, b=0",
            max_hamilton_check_n=4,
        )

        self.assertEqual(result["family"], ROTATION_REFLECTION)
        self.assertTrue(result["family_conditions"])
        self.assertFalse(result["three_involution_conditions"])
        self.assertTrue(result["generates"])
        self.assertTrue(result["hamiltonian"])

    def test_cycle_to_web_route_closes_cycle(self):
        cycle = [(0, 0), (0, 1), (2, 0)]

        self.assertEqual(
            cycle_to_web_route(cycle),
            ["/page/e", "/page/r0s", "/page/r2", "/page/e"],
        )

    def test_custom_page_route_uses_loaded_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pages_path = os.path.join(temp_dir, "pages.txt")
            with open(pages_path, "w", encoding="utf-8") as page_file:
                page_file.write("# test pages\n")
                page_file.write("https://example.test/e\n")
                page_file.write("https://example.test/r0s\n")
                page_file.write("https://example.test/r1\n")

            pages = load_page_paths(pages_path)
            vertex_page_map = build_vertex_page_map(
                [(0, 0), (0, 1), (1, 0)],
                pages,
            )

            self.assertEqual(
                format_page_route([(0, 0), (0, 1)], vertex_page_map),
                (
                    "https://example.test/e -> "
                    "https://example.test/r0s -> "
                    "https://example.test/e"
                ),
            )

    def test_csv_experiment_writes_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "experiment.csv")

            summary = run_csv_experiment(
                main.check_generator_system,
                max_n=4,
                max_hamilton_check_n=4,
                output_path=output_path,
                families=[THREE_INVOLUTIONS],
            )

            self.assertEqual(summary["iterations"], 4)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, "r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 4)
            row = next(row for row in rows if row["n"] == "4" and row["k"] == "1")
            self.assertEqual(row["family"], THREE_INVOLUTIONS)
            self.assertEqual(row["generators"], "r0s, r2s, r1s")
            self.assertEqual(row["generator_conditions"], "True")
            self.assertEqual(row["theorem_conditions"], "True")
            self.assertEqual(row["hamiltonian"], "True")
            self.assertIn("/page/e", row["web_route"])

    def test_csv_experiment_writes_custom_page_route(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "experiment.csv")
            page_paths = [
                f"https://example.test/page-{index}"
                for index in range(8)
            ]

            run_csv_experiment(
                main.check_generator_system,
                max_n=4,
                max_hamilton_check_n=4,
                output_path=output_path,
                families=[THREE_INVOLUTIONS],
                page_paths=page_paths,
            )

            with open(output_path, "r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            row = next(row for row in rows if row["n"] == "4" and row["k"] == "1")
            self.assertIn("https://example.test/page-0", row["page_route"])

    def test_csv_experiment_supports_multiple_families(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "experiment.csv")

            summary = run_csv_experiment(
                main.check_generator_system,
                max_n=4,
                max_hamilton_check_n=4,
                output_path=output_path,
                families=[THREE_INVOLUTIONS, ROTATION_REFLECTION],
                max_iterations=8,
            )

            self.assertEqual(summary["iterations"], 8)

            with open(output_path, "r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertTrue(
                any(row["family"] == ROTATION_REFLECTION for row in rows)
            )

    def test_draw_graph_writes_image(self):
        result = main.check_parameters(4, 1, max_hamilton_check_n=4)
        graph = build_cayley_graph(
            build_dihedral_group(4),
            result["generators"],
            4,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "graph.png")
            saved_path = main.draw_graph(
                graph,
                result["generators"],
                cycle=result["cycle"],
                output_path=output_path,
                show=False,
            )

            self.assertEqual(saved_path, output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()
