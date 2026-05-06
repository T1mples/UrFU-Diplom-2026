import unittest

import main
from algorithm import build_dihedral_involution_generators, commutes
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


if __name__ == "__main__":
    unittest.main()
