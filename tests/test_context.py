"""Unit tests for Context Window Budgeting and Hardware Advisories."""

import unittest
import torch
from prismrunner.engine.context import ContextBudget, generate_hardware_advisory


class TestContextBudget(unittest.TestCase):

    def test_budget_calculations(self):
        budget = ContextBudget(
            model_context_limit=4096,
            max_new_tokens=512,
            system_prompt_tokens=64,
            template_tokens=128,
            safety_buffer=128,
        )

        self.assertEqual(budget.reserved_tokens, 512 + 64 + 128 + 128)
        self.assertEqual(budget.max_input_tokens, 4096 - 832)

    def test_budget_overflow_raises_error(self):
        # Exceed context window
        budget = ContextBudget(
            model_context_limit=512,
            max_new_tokens=512,
            system_prompt_tokens=64,
            template_tokens=128,
            safety_buffer=128,
        )
        with self.assertRaises(ValueError):
            _ = budget.max_input_tokens

    def test_hardware_advisory(self):
        device = torch.device("cpu")
        advisory = generate_hardware_advisory("test-model", 4096, device)
        self.assertEqual(advisory["model_name"], "test-model")
        self.assertEqual(advisory["context_limit"], 4096)
        self.assertIn("recommended_batch_size", advisory)
        self.assertIn("max_safe_input_tokens", advisory)


if __name__ == "__main__":
    unittest.main()
