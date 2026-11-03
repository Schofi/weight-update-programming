import unittest

import torch

from math_edit import math_factors
from math_eval import PresencePenalty, correct, final_answer, sample_tasks


class MathTests(unittest.TestCase):
    def test_thinking_and_incomplete_answers(self):
        self.assertIsNone(final_answer(r"maybe \boxed{42}", True))
        self.assertIsNone(final_answer(r"</think> maybe \boxed{\frac{1}{2}", True))
        self.assertEqual(final_answer(r"\boxed{42}</think> answer \boxed{\frac{1}{2}}.", True),
                         r"\frac{1}{2}")
        self.assertEqual(final_answer(r"\boxed{7}", False), "7")

    def test_symbolic_and_empty_answers(self):
        self.assertTrue(correct(r"\frac{1}{2}", "0.5"))
        self.assertTrue(correct("x^2-1", "(x+1)(x-1)"))
        self.assertFalse(correct("11", "1"))
        self.assertFalse(correct("7", None))
        self.assertFalse(correct("45", "45 %"))
        self.assertTrue(correct("1.5", "2-50%"))

    def test_real_separate_splits(self):
        train, gsm, math = [sample_tasks(name, 8, 19) for name in ("gsm8k_train", "gsm8k_test", "math500")]
        self.assertEqual(gsm, sample_tasks("gsm8k_test", 8, 19))
        self.assertEqual(len({x["question"] for x in train + gsm + math}), 24)

    def test_rank_one_projection(self):
        w = torch.tensor([[1., 2., 3.], [2., 0., -1.]])
        a, b = math_factors(w, torch.tensor([2., 0.]), top=1, gain=0.5)
        torch.testing.assert_close(w + b @ a, torch.tensor([[1., 2., 1.5], [2., 0., -1.]]))
        a, b = math_factors(w, torch.zeros(2), 1, 1.)
        torch.testing.assert_close(b @ a, torch.zeros_like(w))

    def test_presence_penalty_ignores_prompt_and_counts_only_once(self):
        scores = torch.zeros(1, 5)
        out = PresencePenalty(2, 1.5)(torch.tensor([[1, 0, 3, 3, 4]]), scores)
        torch.testing.assert_close(out, torch.tensor([[0., 0., 0., -1.5, -1.5]]))

if __name__ == "__main__":
    unittest.main()
