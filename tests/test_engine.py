"""Unit tests for prismrunner.engine (device detection, context budgeting, and batch inference)."""

import unittest
from unittest.mock import MagicMock, patch
import torch

from prismrunner.engine import (
    BatchInferenceEngine,
    ContextBudget,
    JsonBracketStoppingCriteria,
    create_length_buckets,
    generate_hardware_advisory,
    get_default_dtype,
    get_optimal_device,
    is_json_bracket_balanced,
    resolve_device_and_dtype,
    resolve_dynamic_batch_size,
    restore_original_order,
    validate_cuda_device,
)


class TestEngine(unittest.TestCase):

    def test_device_and_dtype_resolution(self):
        self.assertTrue(validate_cuda_device(torch.device("cpu")))
        self.assertEqual(get_optimal_device(preferred="cpu").type, "cpu")
        self.assertEqual(get_default_dtype(torch.device("cpu")), torch.float32)

        device_name, dtype = resolve_device_and_dtype()
        self.assertIn(device_name, ("cuda", "mps", "cpu"))
        self.assertIn(dtype, (torch.bfloat16, torch.float16, torch.float32))

    @patch("torch.zeros")
    def test_validate_cuda_device_kernel_missing(self, mock_zeros):
        cuda_dev = torch.device("cuda:0")
        mock_tensor = MagicMock()
        mock_tensor.__add__.side_effect = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        mock_zeros.return_value = mock_tensor

        with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX Mock"), \
             patch("torch.cuda.get_device_capability", return_value=(8, 9)), \
             patch("torch.cuda.get_arch_list", return_value=["sm_70", "sm_80"]):
            self.assertFalse(validate_cuda_device(cuda_dev, raise_on_error=False))
            with self.assertRaises(RuntimeError):
                validate_cuda_device(cuda_dev, raise_on_error=True)

    def test_context_budget_and_advisory(self):
        budget = ContextBudget(model_context_limit=4096, max_new_tokens=512, system_prompt_tokens=64, template_tokens=128, safety_buffer=128)
        self.assertEqual(budget.reserved_tokens, 832)
        self.assertEqual(budget.max_input_tokens, 3264)

        overflow = ContextBudget(model_context_limit=512, max_new_tokens=512)
        with self.assertRaises(ValueError):
            _ = overflow.max_input_tokens

        advisory = generate_hardware_advisory("test-model", 4096, torch.device("cpu"))
        self.assertEqual(advisory["model_name"], "test-model")
        self.assertEqual(advisory["context_limit"], 4096)

    def test_length_bucketed_collation(self):
        items = [
            {"id": "doc1", "prompt": "short"},
            {"id": "doc2", "prompt": "a much longer prompt sequence with many words"},
            {"id": "doc3", "prompt": "medium text length"},
            {"id": "doc4", "prompt": "tiny"},
        ]
        buckets = create_length_buckets(items, batch_size=2, text_key="prompt")
        self.assertEqual([x["id"] for x in buckets[0]], ["doc4", "doc1"])
        self.assertEqual([x["id"] for x in buckets[1]], ["doc3", "doc2"])

        processed = [dict(x, output=f"out_{x['id']}") for b in buckets for x in b]
        restored = restore_original_order(processed)
        self.assertEqual([x["id"] for x in restored], ["doc1", "doc2", "doc3", "doc4"])
        self.assertNotIn("_orig_idx", restored[0])

        # Windowed length bucketing keeps items within local windows while sorting inside each window
        windowed_items = [
            {"id": "w1_long", "prompt": "a very long sequence in window one"},
            {"id": "w1_short", "prompt": "short"},
            {"id": "w2_short", "prompt": "tiny"},
            {"id": "w2_long", "prompt": "another very long sequence in window two"},
        ]
        win_buckets = create_length_buckets(windowed_items, batch_size=1, text_key="prompt", window_size=2)
        self.assertEqual([b[0]["id"] for b in win_buckets], ["w1_short", "w1_long", "w2_short", "w2_long"])

        # Dynamic batch size resolution scales down for long sequences and respects explicit int overrides
        self.assertEqual(resolve_dynamic_batch_size(torch.device("cpu"), requested_batch_size=6), 6)
        short_b = resolve_dynamic_batch_size(torch.device("mps"), requested_batch_size="auto", items=items, text_key="prompt")
        long_items = [{"prompt": " ".join(["word"] * 2200)}]
        long_b = resolve_dynamic_batch_size(torch.device("mps"), requested_batch_size="auto", items=long_items, text_key="prompt")
        self.assertGreaterEqual(short_b, long_b)

    def test_greedy_generation_and_async_batches(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.padding_side = "left"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.batch_decode.return_value = ["res_a", "res_b"]

        engine = BatchInferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
            batch_size=2,
            temperature=0.0,
        )

        engine.generate_batch(["hello"], temperature=0.0)
        call_kwargs = mock_model.generate.call_args[1]
        self.assertFalse(call_kwargs["do_sample"])
        self.assertIsNone(call_kwargs["temperature"])
        self.assertIsNone(call_kwargs["top_p"])

        items = [{"id": str(i), "prompt": f"text {i}"} for i in range(4)]
        results = engine.run_all(items, max_workers=2)
        self.assertEqual([r["id"] for r in results], ["0", "1", "2", "3"])

    def test_constrained_json_prefix_and_stopping_criteria(self):
        self.assertTrue(is_json_bracket_balanced('[{"name": "Alice"}]', "[", "]"))
        self.assertFalse(is_json_bracket_balanced('[{"name": "Alice"}', "[", "]"))
        self.assertTrue(is_json_bracket_balanced('[{"note": "contains ] inside string"}]', "[", "]"))
        self.assertFalse(is_json_bracket_balanced('[{"note": "contains ] inside string"}', "[", "]"))
        self.assertTrue(is_json_bracket_balanced('{"key": "val"}', "{", "}"))

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.batch_decode.return_value = [
            '{"name": "A"}]',
            '{"name": "B"}]',
        ]
        criteria = JsonBracketStoppingCriteria(
            tokenizer=mock_tokenizer,
            prompt_length=2,
            response_prefix="[",
        )
        input_ids = torch.tensor([[10, 11, 12, 13], [10, 11, 14, 15]])
        self.assertTrue(criteria(input_ids, torch.zeros((2, 10))))

        mock_model = MagicMock()
        mock_tokenizer_gen = MagicMock()
        mock_tokenizer_gen.pad_token_id = 0
        mock_tokenizer_gen.eos_token_id = 2
        mock_tokenizer_gen.padding_side = "left"
        mock_tokenizer_gen.apply_chat_template.return_value = "<|im_start|>assistant\n"
        mock_tokenizer_gen.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_gen.batch_decode.return_value = ['{"entity_name": "Acme"}]']

        engine = BatchInferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer_gen,
            device=torch.device("cpu"),
        )
        outputs = engine.generate_batch(["Extract entities"], response_prefix="[")
        self.assertEqual(outputs[0], '[{"entity_name": "Acme"}]')
        self.assertIn("stopping_criteria", mock_model.generate.call_args[1])


if __name__ == "__main__":
    unittest.main()

