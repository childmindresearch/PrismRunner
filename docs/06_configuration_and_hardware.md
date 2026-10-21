# Configuration & Hardware Tuning Reference

PrismRunner is engineered for high throughput and efficient resource utilization across varied hardware—from developer laptops with Apple Silicon (M1/M2/M3/M4) to multi-GPU NVIDIA workstations and CPU-only clusters.

---

## 1. Central Configuration (`config/pipeline_config.yaml`)

All runtime defaults are declared in `config/pipeline_config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  device: "auto"             # "auto", "mps", "cuda", or "cpu"
  torch_dtype: "auto"        # "auto", "float16", "bfloat16", "float32"
  load_in_4bit: false        # Enable 4-bit quantization (BitsAndBytes)
  load_in_8bit: false        # Enable 8-bit quantization (BitsAndBytes)

inference:
  batch_size: 4
  max_new_tokens: 512
  temperature: 0.0           # 0.0 for strict deterministic extraction
  top_p: 0.95
  system_prompt: "You are PrismRunner, an exact information extraction engine. Output valid JSON array only."

chunking:
  enabled: false
  chunk_size: 1024
  chunk_overlap: 64
  strategy: "sentence"       # "sentence" or "token"

prompts:
  classify: "prompts/classify.yaml"
  extract: "prompts/extract.yaml"
  summarize: "prompts/summarize.yaml"

io:
  default_text_column: "text"
  default_id_column: "id"
  output_format: "json"
```

---

## 2. Hardware Acceleration & Device Mapping

PrismRunner automatically maps to the optimal available accelerator:

| Accelerator | Hardware Platform | Precision Default | Cache Management |
| :--- | :--- | :--- | :--- |
| **MPS** | Apple Silicon (M1/M2/M3/M4) | `float16` | `torch.mps.empty_cache()` |
| **CUDA** | NVIDIA GPUs (RTX, A100, H100) | `bfloat16` or `float16` | `torch.cuda.empty_cache()` |
| **CPU** | Intel / AMD x86 & ARM servers | `float32` | System RAM |

### Overriding Device in Code
```python
from prismrunner import PrismPipeline

# Force CPU mode
pipeline = PrismPipeline(device="cpu")

# Force CUDA device 0
pipeline = PrismPipeline(device="cuda:0")
```

---

## 3. Batch Size & VRAM Tuning

Selecting the correct batch size prevents Out-Of-Memory (OOM) errors and maximizes GPU utilization:

| Available Memory | Model Parameter Size | Recommended Batch Size | Quantization |
| :--- | :--- | :--- | :--- |
| **8 GB Unified / VRAM** | 1.5B – 3B | 2 – 4 | FP16 or 4-bit |
| **16 GB Unified / VRAM** | 3B – 7B | 4 – 8 | FP16 or 8-bit |
| **32 GB Unified / VRAM** | 7B – 14B | 8 – 16 | FP16 / BF16 |
| **64+ GB Unified / VRAM** | 14B – 32B | 16 – 32 | FP16 / BF16 |

> [!TIP]
> PrismRunner runs left-padded batch generation with causal attention masking. If you encounter memory pressure during long document passes, reduce `batch_size` or decrease `chunk_size` from 1024 to 512.

---

## 4. Context Budget & Safe Input Formula

Local LLMs have strict context window limits (e.g. 32,768 tokens for Qwen 2.5). When designing chunk sizes, PrismRunner calculates the **Safe Input Capacity**:

$$\text{Safe Input} = \text{Context Limit} - (\text{Max New Tokens} + \text{System Prompt} + \text{Template Overheads} + \text{Safety Buffer})$$

For example, with a 32k context model, 512 max new tokens, and ~300 prompt tokens, the safe input token budget is easily 28,000+ tokens. For high-density documents, a sliding window chunk size between **1,024 and 2,048 tokens** ensures optimal extraction granularity.

### Inspecting Model Limits:
```bash
prismrunner inspect-model --model Qwen/Qwen2.5-3B-Instruct
```

---

## 5. Chunking Strategies

- **`sentence` (Default & Recommended)**: Splits text at punctuation boundaries (`.`, `!`, `?`, `\n`) while respecting the token limit. Ensures semantic sentences are not cut in half, maximizing extraction accuracy.
- **`token`**: Strict sliding token window. Splits at exact token counts regardless of sentence boundaries. Suitable for dense code or tabular logs.
