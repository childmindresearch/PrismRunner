# PrismRunner 💎

> **Turn messy text and documents into clean spreadsheets and databases—running 100% locally on your computer.**  
> Free, offline, and private. Works seamlessly on Apple Silicon (Mac), NVIDIA GPUs, and CPU.

---

```text
[ Raw Text, Notes, or CSVs ]  ──▶  [ PrismRunner (Local AI) ]  ──▶  [ Clean Excel / SQLite / JSON ]
"Patient was prescribed 50mg Metformin daily"                         {"drug": "Metformin", "dose": "50mg", "freq": "daily"}
```

---

## 🌟 Why PrismRunner?

* 🔒 **100% Private & Free:** Runs local open-source models (like Qwen, Llama, or Mistral). No OpenAI API keys, zero cloud fees, and your sensitive data never leaves your computer.
* 🎯 **Guaranteed Structure:** Uses standard [Pydantic](https://docs.pydantic.dev/) schemas to ensure the AI always returns exact numbers, dates, and categories—not random conversational filler.
* 📦 **Built for Real Datasets:** Handles long documents, auto-chunks text without losing context, repairs broken JSON outputs, and stitches multi-page results together automatically.

---

## ⚡ Quick Start (Under 60 Seconds)

### 1. Install

```bash
git clone https://github.com/childmindresearch/PrismRunner.git
cd PrismRunner
pip install -e .
```

### 2. Run your first extraction

Extract structured information directly from the terminal:

```bash
prismrunner classify \
  --input-text "Apple unveils new M4 chip with enhanced neural engine tensor cores." \
  --json
```

**Output:**
```json
{
  "category": "Technology & Hardware",
  "confidence": 0.98,
  "reasoning": "Mentions Apple's M4 processor and neural engine tensor cores."
}
```

> **Tip: Test prompts instantly with Dry-Run**  
> Add `--dry-run` to preview how your prompt and schema look without downloading or loading model weights into memory:  
> ```bash
> prismrunner classify --input-text "Clinical trial shows high efficacy." --dry-run
> ```

---

## 🛠️ The 4 Ways to Use PrismRunner

### 1. Categorize Text (`classify`)
Sort rows into labels with confidence scores (e.g. sentiment, support ticket routing, topic tags):
```bash
prismrunner classify \
  --input-path data/articles.parquet \
  --column text \
  --output-path outputs/classified.json
```

### 2. Extract Specific Fields (`extract`)
Pull out exact names, dates, dosages, or key metrics into a defined schema:
```bash
prismrunner extract \
  --input-path data/clinical_notes.csv \
  --column text \
  --schema "schemas/clinical_records.py:Medication" \
  --output-path outputs/medications.json
```

### 3. Summarize Long Documents (`summarize`)
Generate structured summaries and key takeaways across lengthy documents:
```bash
prismrunner summarize \
  --input-path data/annual_report.txt \
  --chunk \
  --output-path outputs/summary.json
```

### 4. Multi-Schema Database Mapping (`map`)
Scan documents against an entire catalog of schemas and automatically export to SQLite, normalized CSVs, or wide tables:
```bash
prismrunner map \
  --input-path data/clinical_notes.parquet \
  --schemas-dir schemas/ \
  --output-dir outputs/clinical_run/ \
  --export sqlite,csv,wide
```

---

## 🐍 Python API

PrismRunner works directly with Pandas DataFrames, lists of dictionaries, or raw text:

```python
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from prismrunner import PrismPipeline

# 1. Define what you want to extract
class FeedbackItem(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Customer sentiment")
    summary: str = Field(description="Brief summary of the feedback")
    urgency: int = Field(ge=1, le=5, description="Urgency score from 1 (low) to 5 (critical)")

# 2. Ingest your data
df = pd.DataFrame({
    "ticket_id": ["T-101", "T-102"],
    "message": [
        "Payment gateway is failing at checkout! Customers cannot complete purchases.",
        "Love the new dashboard design, very smooth experience.",
    ]
})

# 3. Run the pipeline (auto-detects Apple Silicon MPS, NVIDIA CUDA, or CPU)
pipeline = PrismPipeline()
results = pipeline.run(
    task="extract",
    input_data=df,
    text_column="message",
    id_column="ticket_id",
    schema_cls=FeedbackItem,
    batch_size=4
)

print(results)
```

---

<details>
<summary>⚙️ <b>Under the Hood (For Engineers & Researchers)</b></summary>

PrismRunner includes a high-performance batch execution engine designed for local hardware:

* **Token-Aware Chunking:** Segments documents along sentence boundaries to prevent mid-sentence cuts while strictly respecting your model's token limits.
* **Schema-to-Prompt Compiler:** Converts Pydantic models into clear bulleted rules and sample JSON—never leaking raw OpenAPI `$defs` or `$ref` blobs that confuse local open-weight models.
* **ChatML Left-Padding Batching:** Left-pads tokens and aligns ChatML generation prompts to maximize throughput on GPUs and Apple Silicon while eliminating prompt echoing.
* **Resilient Output Parsing:** Strips markdown code fences, repairs trailing commas, and filters hallucinated empty objects.
* **Deterministic Stitching:** Reassembles multi-chunk outputs into single record results using majority voting (classification) or deduplication (extraction).
* **Hardware Advisor:** Inspect context limits and recommended chunk sizes for any model with `prismrunner inspect-model --model Qwen/Qwen2.5-3B-Instruct`.

*For complete technical subsystem specifications and diagrams, see [docs/07_architecture_and_internals.md](docs/07_architecture_and_internals.md).*

</details>

---

## 📚 Guides & Documentation

Step-by-step guides and recipes are available in the [**`docs/`**](docs/README.md) directory:

| Guide | Description |
| :--- | :--- |
| [**01. Prompt Customization**](docs/01_prompt_customization.md) | Overriding templates, prompt variables, and system prompts. |
| [**02. Jupyter Notebook Guide**](docs/02_jupyter_notebooks.md) | Interactive workflows, in-cell Pydantic schemas, and DataFrame recipes. |
| [**03. Python Package API Reference**](docs/03_python_package_usage.md) | Complete SDK reference for `PrismPipeline` and `SchemaMappingPipeline`. |
| [**04. Standalone & CLI Reference**](docs/04_standalone_and_cli.md) | Command flags, recipes, and standalone `main.py` execution. |
| [**05. Custom Schemas & Catalogs**](docs/05_custom_schemas_and_catalogs.md) | Defining schemas, multi-schema catalogs, and SQLite/wide-table exports. |
| [**06. Configuration & Hardware Tuning**](docs/06_configuration_and_hardware.md) | Optimizing MPS, CUDA, CPU, VRAM batch sizing, and token budgets. |
| [**07. Architecture & Internals**](docs/07_architecture_and_internals.md) | Deep dive into batch runtime, token chunking, and parser mechanics. |

---

## ⚙️ Configuration (`config/pipeline_config.yaml`)

Customize default model and batch settings in `config/pipeline_config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  device: "auto"             # auto, mps, cuda, cpu
  torch_dtype: "auto"        # auto, float16, bfloat16, float32

inference:
  batch_size: 4
  max_new_tokens: 512
  temperature: 0.0

chunking:
  enabled: false
  chunk_size: 1024
  chunk_overlap: 64
```

---

## License

MIT
