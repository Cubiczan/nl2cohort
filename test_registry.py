"""Focused contract checks for the shared metric registry."""
import unittest
from pathlib import Path

import yaml

from gen_pairs import measures_from_registry


class MetricRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / "registry" / "cohorts.yaml"
        cls.registry = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_procurement_terms_have_named_definitions(self):
        measures = self.registry["measures"]
        for name in ("spend", "landed_cost", "gross_margin", "dso", "dio", "dpo", "ccc", "value_pool"):
            self.assertTrue(measures[name]["ambiguous"])
            self.assertGreaterEqual(len(measures[name]["definitions"]), 2)
            for definition in measures[name]["definitions"].values():
                self.assertTrue(definition["description"])
                self.assertTrue(definition["expression"])
                self.assertTrue(definition["unit"])

    def test_generator_never_emits_bare_ambiguous_metric(self):
        legal = measures_from_registry(self.registry)
        for name in ("spend", "landed_cost", "gross_margin", "dso", "dio", "dpo", "ccc", "value_pool"):
            self.assertNotIn(name, legal)
        self.assertIn("spend@cash_paid", legal)
        self.assertIn("ccc@average_balance", legal)


if __name__ == "__main__":
    unittest.main()
