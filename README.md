# PrismRunner

Lightweight, local-first LLM batch inference library for turning unstructured text, Pandas DataFrames, and datasets into validated [Pydantic v2](https://docs.pydantic.dev/) schemas on NVIDIA CUDA, Apple Silicon (MPS), or CPU.

---

## Key Capabilities & Dynamic Optimizations

Every hardware, memory, context, and batching optimization in PrismRunner is computed **dynamically and continuously relative to the active device and loaded model at runtime** (no hardcoded GPU tiers or fixed VRAM ceilings):

- **Dynamic Hardware, Precision & Kernel Detection**:
  - Automatically detects `cuda` (Ampere/Hopper/Blackwell `bfloat16` or `float16` + TF32 matrix kernels), Apple Silicon `mps` (`float16` + unified memory configuration), or `cpu` (`float32`).
  - Automatically selects `flash_attention_2` (when `flash-attn` is installed on Ampere+ CUDA) or PyTorch's native `sdpa` (Scaled Dot-Product Attention), and enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent CUDA allocator fragmentation across variable-length batches.
- **Continuous Device-Relative Batch & Context Sizing**:
  - Inspects both top-level `config` and nested `config.text_config` (`max_position_embeddings`, `seq_length`, `sliding_window`) to detect the model's native context limit automatically (`CONTEXT_WINDOW_CAP = "auto"`).
  - Queries live free device memory (`torch.cuda.mem_get_info` on CUDA, unified physical RAM minus `torch.mps.current_allocated_memory()` on MPS), the **loaded model's actual weight footprint** (`torch.cuda.memory_allocated`), and the **90th-percentile tokenized prompt length** of the queued items via [`resolve_dynamic_batch_size`](prismrunner/engine.py) to continuously scale `batch_size` (`1`–`32`) and `chunk_size` (`"auto"`) per `.run()` call.
- **Automatic OOM Batch-Splitting Recovery**:
  - If a prompt batch triggers a `torch.OutOfMemoryError` during prefill or decoding (e.g., on long sequences or hybrid linear-attention architectures), [`BatchInferenceEngine.generate_batch`](prismrunner/engine.py) catches the OOM, flushes device caches via [`clear_device_memory`](prismrunner/engine.py), splits the batch in half recursively, and completes inference without crashing or losing progress.
- **Unified Multimodal & Hybrid Architecture Support (`Qwen3` / `Qwen3.5` / `Qwen3.6`, etc.)**:
  - Automatically falls back from `AutoModelForCausalLM` to `AutoModelForImageTextToText` / `AutoModelForMultimodalLM` when loading unified vision-text checkpoints (such as `Qwen/Qwen3.5-27B` or `Qwen/Qwen3.6-27B`) for text-only batch inference.
  - Automatically passes `enable_thinking=False` to chat templates when `constrain_json=True` (`[` / `{` response prefixing) is active so reasoning models emit structured JSON immediately instead of reasoning prose inside JSON strings.
- **Windowed Length-Bucketed Batch Inference**:
  - Sorts prompts by token length within local sliding windows (`window_size = batch_size * 8`) via [`create_length_buckets`](prismrunner/engine.py) to minimize left-padding waste while keeping multi-chunk documents contiguous, runs under `torch.inference_mode()` with KV cache enabled (`use_cache=True`), and restores the exact original input order afterward.
- **Streaming Progress & Crash-Safe Checkpoint Callbacks**:
  - Supports `on_batch_start` and `on_chunk_processed` hooks on [`PrismPipeline.run`](prismrunner/pipeline.py) so long-running jobs log real-time batch progress and flush completed documents to disk incrementally as chunks finish.
- **Native Constrained JSON Decoding & Fault-Tolerant Parser**:
  - Pre-fills the assistant header with `[` or `{` to eliminate conversational preamble, and uses token-filtered [`JsonBracketStoppingCriteria`](prismrunner/engine.py) to halt `model.generate()` as soon as all sequences in the batch close their outer JSON bracket without re-decoding on non-bracket tokens.
  - Compiles Pydantic v2 schemas into compact prompt specifications and orders `reasoning` before `category` in [`ClassificationItem`](prismrunner/schema.py) to avoid the token-1 reasoning penalty.
  - Automatically strips `<think>...</think>` blocks, markdown code fences, and trailing commas, unwraps container dicts, filters all-null phantom objects, and salvages completed items from truncated JSON arrays cut off by `max_new_tokens`.

---

## Installation

Install all core, clustering, and Triton (`flash-linear-attention`) dependencies via `requirements.txt` or editable install:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### Optional C++/CUDA Source Extensions (`causal-conv1d` & `flash-attn`)

Because `causal-conv1d` and `flash-attn` import `torch` inside their `setup.py` at build time, `pip`'s default isolated build sandbox will raise `ModuleNotFoundError: No module named 'torch'` unless `--no-build-isolation` is passed **after** `torch`, `packaging`, and `ninja` are installed:

```bash
pip install --no-build-isolation causal-conv1d flash-attn
```
*(Note: PyTorch 2.4+ already includes FlashAttention-2 natively inside its `sdpa` backend, and `flash-linear-attention` is pure Python/Triton and installs automatically during `pip install -e .`.)*

---

## Python API Usage

Import `PrismPipeline` directly into your Python program to extract custom Pydantic models, classify records, or summarize documents across strings, lists, Pandas DataFrames, or dataset files (`.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, `.md`).

### 1. Structured Extraction with Custom Pydantic Schemas

```python
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field
from prismrunner import PrismPipeline

class FeedbackItem(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Customer sentiment")
    summary: str = Field(description="Brief summary of the feedback")
    urgency: int = Field(ge=1, le=5, description="Urgency score from 1 (low) to 5 (critical)")

df = pd.DataFrame({
    "ticket_id": ["T-101", "T-102"],
    "message": [
        "Payment gateway is failing at checkout! Customers cannot complete purchases.",
        "Love the new dashboard design, very smooth experience.",
    ],
})

pipeline = PrismPipeline(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="auto",       # "auto", "cuda", "mps", or "cpu"
    batch_size="auto",   # Dynamically sized from live device headroom, model size & prompt length
)

results = pipeline.run(
    task="extract",
    input_data=df,
    text_column="message",
    id_column="ticket_id",
    schema_cls=FeedbackItem,
    constrain_json=True, # Forces '[' prefix and stops on balanced ']' (default: True)
)

for row in results:
    print(row["id"], row["extracted_items"])
```

### 2. Built-in Tasks (`classify`, `extract`, `summarize`) & Auto-Chunking

```python
# Classify text (uses ClassificationItem with chain-of-thought reasoning field first)
classified = pipeline.run(
    task="classify",
    input_data=["Apple unveils new M4 chip with 16-core neural engine."],
    variables={"categories": "Technology, Finance, Healthcare"},
)

# Summarize long documents with dynamic token-aware sliding-window chunking
summaries = pipeline.run(
    task="summarize",
    input_data="path/to/long_report.txt",
    chunk="auto",        # Automatically splits only documents exceeding the context budget
    chunk_size="auto",   # Dynamically computed from model context window & prompt overhead
    chunk_overlap=64,
)
```

### 3. Streaming Progress & Incremental Checkpoint Callbacks

Use `on_batch_start` and `on_chunk_processed` to monitor concurrent batches in real time and save results incrementally as chunks complete:

```python
results = pipeline.run(
    task="extract",
    input_data=df,
    text_column="message",
    id_column="ticket_id",
    schema_cls=FeedbackItem,
    on_batch_start=lambda idx, total, items: print(f"Batch {idx}/{total} ({len(items)} items)"),
    on_chunk_processed=lambda res: print(f"Completed {res['record_id']}: {len(res['parsed_items'])} items"),
)
```

### 4. Direct Access to Engine, Compiler & Parser Primitives

```python
from prismrunner import (
    BatchInferenceEngine,
    UniversalOutputParser,
    compile_schema_to_prompt,
    init_model_and_tokenizer,
    resolve_dynamic_batch_size,
)

# Compile any Pydantic v2 model into a flat, readable prompt specification
compiled = compile_schema_to_prompt(FeedbackItem)
prompt_block = compiled.render_prompt_block(as_array=True)

# Parse and validate raw LLM output (strips <think> tags, fences, trailing commas, and repairs truncated JSON)
parsed = UniversalOutputParser.parse(raw_llm_text, model_cls=FeedbackItem)
print(parsed.items)
```

---

## Repository Structure

```text
PrismRunner/
├── prismrunner/
│   ├── __init__.py   # Public package exports
│   ├── engine.py     # Dynamic hardware/dtype/batch detection, model loading, OOM recovery, BatchInferenceEngine
│   ├── schema.py     # Built-in schemas, schema-to-prompt compiler, UniversalOutputParser
│   ├── data.py       # DatasetLoader, TokenAwareChunker, DeterministicStitcher
│   ├── prompt.py     # Default task prompts and PromptTemplate renderer
│   ├── pipeline.py   # PrismPipeline end-to-end orchestrator
│   └── cli.py        # Optional CLI (`prismrunner classify|extract|summarize|inspect-model`)
├── usertests/        # Runnable user walkthroughs (.py scripts, SemanticPrism1.py & .ipynb Notebook)
├── tests/            # Unit & mocked integration test suite
├── requirements.txt  # Core, clustering & optional CUDA kernel dependencies
└── pyproject.toml    # Package metadata, optional [cuda] & [dev] extras, and pytest config
```

---

## Running Tests & User Examples

Run the automated test suite (no model download required):

```bash
pytest -v
```

Run the interactive user tests in `usertests/`:

```bash
python usertests/01_quickstart_and_hardware.py
python usertests/02_custom_schemas_and_dataframes.py
python usertests/SemanticPrism1.py
# Or open usertests/03_interactive_walkthrough.ipynb in Jupyter / VS Code
```
