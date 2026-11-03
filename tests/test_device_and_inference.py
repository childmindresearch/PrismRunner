"""Unit tests for device validation and inference generation parameter alignment."""

import unittest
from unittest.mock import MagicMock, patch
import torch

from prismrunner.engine.device import (
    validate_cuda_device,
    get_optimal_device,
    get_default_dtype,
)
from prismrunner.engine.inference import BatchInferenceEngine


class TestDeviceValidation(unittest.TestCase):

    def test_validate_non_cuda_device(self):
        cpu_dev = torch.device("cpu")
        self.assertTrue(validate_cuda_device(cpu_dev))

    @patch("torch.zeros")
    def test_validate_cuda_device_kernel_missing(self, mock_zeros):
        cuda_dev = torch.device("cuda:0")
        mock_tensor = MagicMock()
        mock_tensor.__add__.side_effect = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        mock_zeros.return_value = mock_tensor

        # Should return False when raise_on_error=False
        with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX Mock"), \
             patch("torch.cuda.get_device_capability", return_value=(8, 9)), \
             patch("torch.cuda.get_arch_list", return_value=["sm_70", "sm_75", "sm_80", "sm_86"]):
            valid = validate_cuda_device(cuda_dev, raise_on_error=False)
            self.assertFalse(valid)

            # Should raise informative RuntimeError when raise_on_error=True
            with self.assertRaises(RuntimeError) as ctx:
                validate_cuda_device(cuda_dev, raise_on_error=True)
            self.assertIn("no kernel image is available", str(ctx.exception).lower())
            self.assertIn("reinstall pytorch", str(ctx.exception).lower())

    def test_get_optimal_device_cpu(self):
        dev = get_optimal_device(preferred="cpu")
        self.assertEqual(dev.type, "cpu")

    def test_get_default_dtype_cpu(self):
        dtype = get_default_dtype(torch.device("cpu"))
        self.assertEqual(dtype, torch.float32)


class TestInferenceGenerationParameters(unittest.TestCase):

    def test_greedy_generation_clears_sampling_params(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.padding_side = "left"

        # Mock tokenization output
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.batch_decode.return_value = ["output test"]

        engine = BatchInferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
            temperature=0.0,
        )

        engine.generate_batch(["hello"], temperature=0.0)

        # Inspect kwargs passed to model.generate
        call_kwargs = mock_model.generate.call_args[1]
        self.assertFalse(call_kwargs["do_sample"])
        self.assertIsNone(call_kwargs["temperature"])
        self.assertIsNone(call_kwargs["top_p"])
        self.assertTrue(call_kwargs.get("use_cache"))

    def test_resolve_device_and_dtype(self):
        from prismrunner.engine.device import resolve_device_and_dtype
        device_name, dtype = resolve_device_and_dtype()
        self.assertIn(device_name, ("cuda", "mps", "cpu"))
        self.assertIn(dtype, (torch.bfloat16, torch.float16, torch.float32))

    def test_length_bucketed_collation(self):
        from prismrunner.engine.inference import create_length_buckets, restore_original_order
        items = [
            {"id": "doc1", "prompt": "short"},
            {"id": "doc2", "prompt": "a much longer prompt sequence with many words"},
            {"id": "doc3", "prompt": "medium text length"},
            {"id": "doc4", "prompt": "tiny"},
        ]
        buckets = create_length_buckets(items, batch_size=2, text_key="prompt")
        self.assertEqual(len(buckets), 2)
        # First bucket should contain "tiny" (4) and "short" (5)
        self.assertEqual([x["id"] for x in buckets[0]], ["doc4", "doc1"])
        # Second bucket should contain "medium text length" (18) and longest (43)
        self.assertEqual([x["id"] for x in buckets[1]], ["doc3", "doc2"])

        # Flatten and restore original order
        processed = [dict(x, output=f"out_{x['id']}") for b in buckets for x in b]
        restored = restore_original_order(processed)
        self.assertEqual([x["id"] for x in restored], ["doc1", "doc2", "doc3", "doc4"])
        self.assertNotIn("_orig_idx", restored[0])

    def test_run_all_async_overlapping(self):
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

        items = [
            {"id": "1", "prompt": "hello"},
            {"id": "2", "prompt": "world"},
            {"id": "3", "prompt": "long text here"},
            {"id": "4", "prompt": "longer text goes here"},
        ]

        def custom_cb(completions, batch_items):
            out = []
            for itm, comp in zip(batch_items, completions):
                d = dict(itm)
                d["custom_annotated"] = comp
                out.append(d)
            return out

        results = engine.run_all(items, on_batch_complete=custom_cb, max_workers=2)
        self.assertEqual(len(results), 4)
        self.assertEqual([r["id"] for r in results], ["1", "2", "3", "4"])
        for r in results:
            self.assertIn("custom_annotated", r)


if __name__ == "__main__":
    unittest.main()
