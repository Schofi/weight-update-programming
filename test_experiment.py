import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from bench import Case, cases, grade, probe_score
from editor import Recipe, active, read_adapter, save_adapter


class BenchmarkTests(unittest.TestCase):
    def test_fault_hidden_until_discriminating_probe(self):
        for split in ("discovery", "validation", "test"):
            for c in cases(split):
                twin = Case(split, c.domain, 1 - c.fault, c.good_slot, c.repair_slot)
                self.assertEqual(c.probe_prompt(), twin.probe_prompt())
                for i in (0, 1):
                    x, y = probe_score(c, i), probe_score(twin, i)
                    self.assertEqual(x != y, i == c.good_slot)
                    if x == y:
                        self.assertEqual(c.repair_prompt(i, x), twin.repair_prompt(i, y))

    def test_repair_must_fix_damage_and_can_cause_regressions(self):
        for split in ("discovery", "validation", "test"):
            for c in cases(split):
                for i, column in enumerate(c.repair_columns()):
                    stats = grade(c, i)
                    self.assertEqual(stats["joint"], column == c.fault)
                    self.assertGreater(stats["broken_states"], 0)
                    self.assertGreater(stats["intact_states"], 0)
                    if column != c.fault:
                        self.assertGreater(stats["regressed_states"], 0)
                    change = c.patched(i) - c.damaged()
                    self.assertEqual(torch.linalg.matrix_rank(change).item(), 1)

    def test_split_and_family_boundaries(self):
        splits = [cases(s) for s in ("discovery", "validation", "test")]
        prompts = [{c.probe_prompt() for c in group} for group in splits]
        self.assertTrue(prompts[0].isdisjoint(prompts[1] | prompts[2]))
        self.assertTrue(prompts[1].isdisjoint(prompts[2]))
        self.assertEqual({c.damage for c in splits[0] + splits[1]}, {-3.})
        self.assertEqual({c.damage for c in splits[2]}, {-3., 3.})


class AdapterTests(unittest.TestCase):
    def test_serialized_hook_matches_weight_edit_and_rolls_back_on_error(self):
        model = nn.ModuleDict({"projection": nn.Linear(3, 2, bias=False)})
        layer = model["projection"]
        x = torch.tensor([[1., 2., 3.]])
        base = layer(x).detach()
        a, b = torch.tensor([[0., 1., 0.]]), torch.tensor([[1.], [-1.]])
        recipe = Recipe("projection", "difference", "local", 1.)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "adapter"
            save_adapter(folder, "dummy", recipe, (a, b))
            recovered, factors = read_adapter(folder)
            self.assertEqual(recipe, recovered)
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                with active(SimpleNamespace(model=model), recovered.module, factors):
                    torch.testing.assert_close(layer(x), x @ (layer.weight + b @ a).T)
                    raise RuntimeError("interrupt")
        torch.testing.assert_close(layer(x), base, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
