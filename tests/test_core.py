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
from webgraph import cycle_to_web_route


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

    def test_cycle_to_web_route_closes_cycle(self):
        cycle = [(0, 0), (0, 1), (2, 0)]

        self.assertEqual(
            cycle_to_web_route(cycle),
            ["/page/e", "/page/r0s", "/page/r2", "/page/e"],
        )

    def test_csv_experiment_writes_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "experiment.csv")

            summary = run_csv_experiment(
                main.check_parameters,
                max_n=4,
                max_hamilton_check_n=4,
                output_path=output_path,
            )

            self.assertEqual(summary["iterations"], 4)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, "r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 4)
            row = next(row for row in rows if row["n"] == "4" and row["k"] == "1")
            self.assertEqual(row["generators"], "r0s, r2s, r1s")
            self.assertEqual(row["generator_conditions"], "True")
            self.assertEqual(row["theorem_conditions"], "True")
            self.assertEqual(row["hamiltonian"], "True")
            self.assertIn("/page/e", row["web_route"])

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
