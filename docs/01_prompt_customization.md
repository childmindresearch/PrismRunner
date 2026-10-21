# Customizing Prompts in PrismRunner

PrismRunner features a modular, centralized prompt engine that gives you full control over how instructions, guidelines, source documents, and schema specifications are formatted for local LLMs.

Whether you are working in a **Jupyter Notebook**, writing a Python pipeline, or running CLI jobs, you can customize prompts without modifying core codebase files.

---

## 1. Anatomy of a Prompt

Every prompt in PrismRunner is composed of dynamic slots:

| Slot Tag | Purpose | Injected By |
| :--- | :--- | :--- |
| `{{ text }}` or `{{ item }}` | The source text of the document or chunk. | Pipeline / Dataset Loader |
| `{schema_block}` | Human-readable field specs and a concrete example JSON object. | Dynamic Schema Compiler |
| `{instruction}` | The primary task directive for the LLM. | Template or `instruction="..."` |
| `{variable_name}` | Any custom variable provided in the template or execution dictionary. | Template or `variables={...}` |

> [!NOTE]
> `system_prompt` is sent directly to the model's native ChatML system message slot via `tokenizer.apply_chat_template()`. It is NOT prepended as user text, ensuring proper model alignment and preventing prompt echoing.

---

## 2. In-Memory Customization (Jupyter Notebook / Python)

### Method 1: Override `instruction` & `variables` (Simplest & Most Common)
You keep the default YAML prompt structure and simply customize the task instructions and specific guidelines directly in code:

```python
from pydantic import BaseModel, Field
from prismrunner import PrismPipeline

class Medication(BaseModel):
    drug_name: str = Field(..., description="Generic or trade drug name")
    dosage: str = Field(..., description="Dosage and frequency (e.g., 500mg daily)")
    route: str = Field(..., description="Oral, IV, topical, etc.")

pipeline = PrismPipeline()

results = pipeline.run(
    task="extract",
    input_data="Patient was prescribed Metformin 500mg BID orally. Discontinue Lisinopril.",
    schema_cls=Medication,
    # In-memory instruction override:
    instruction="Extract all active prescriptions. Exclude any discontinued medications.",
    # In-memory guidelines override:
    variables={
        "guidelines": "Extract exact verbatim drug names and route. Do not extrapolate dosages."
    },
)

print(results[0]["parsed_items"])
```

---

### Method 2: Custom Multi-Line String Template (`prompt_template="..."`)
If you want to restructure the prompt layout entirely (e.g. to use clinical notes headers, specific markdown sections, or specialized delimiters), pass a multi-line string containing `{{ text }}` and `{schema_block}`:

```python
custom_prompt = """\
### Clinical Progress Note:
{{ text }}

### Extraction Objective:
Extract every medication mentioned according to the specification below.

### Output Directive:
{schema_block}

### Important Guidelines:
Verify that all dosages contain both strength and frequency.
"""

results = pipeline.run(
    task="extract",
    input_data="Patient was prescribed Metformin 500mg daily.",
    schema_cls=Medication,
    prompt_template=custom_prompt,
    system_prompt="You are a clinical pharmacologist. Output strict JSON only.",
)
```

---

### Method 3: In-Memory Dictionary (`prompt_template={...}`)
To supply a complete prompt specification—including system prompt, instructions, and custom variables—pass a dictionary:

```python
custom_dict = {
    "system_prompt": "You are a specialized clinical trial annotator.",
    "instruction": "Identify all adverse drug reactions and associated treatments.",
    "variables": {
        "severity_rule": "Include mild, moderate, and severe events.",
    },
    "prompt": """\
Task: {instruction}
Severity Rule: {severity_rule}

Source Document:
\"\"\"
{{ text }}
\"\"\"

{schema_block}
"""
}

results = pipeline.run(
    task="extract",
    input_data="Sample document text...",
    schema_cls=Medication,
    prompt_template=custom_dict,
)
```

---

### Method 4: Custom System Prompt Override (`system_prompt="..."`)
You can override the model's system prompt at runtime without altering the template body:

```python
results = pipeline.run(
    task="extract",
    input_data="Sample document text...",
    schema_cls=Medication,
    system_prompt="You are an exact data extraction engine. Output valid JSON array only.",
)
```

---

## 3. External YAML Files (`prompt_file="..."`)

For reproducible, version-controlled experiments, store prompt configurations in `.yaml` files.

### Example: `custom_prompts/oncology_extract.yaml`
```yaml
version: "1.0"
task: extract

system_prompt: "You are an oncology NLP extractor. Output valid JSON array only."

instruction: "Extract all chemotherapy regimens and dosages."

variables:
  guidelines: "Only extract regimens initiated within the last 6 months. Exclude past resolved treatments."
  target: "Chemotherapy agents, immunotherapy, and targeted therapies."

prompt: |
  {instruction}

  Target Elements:
  {target}

  Guidelines:
  {guidelines}

  Source Content:
  """
  {{ text }}
  """

  {schema_block}
```

### Loading in Python:
```python
results = pipeline.run(
    task="extract",
    input_data="data/clinical_notes.parquet",
    schema_cls=Medication,
    prompt_file="custom_prompts/oncology_extract.yaml",
)
```

### Loading in CLI:
```bash
prismrunner extract \
  --input-path data/clinical_notes.parquet \
  --prompt-file custom_prompts/oncology_extract.yaml \
  --output-path outputs/oncology_results.json
```

---

## 4. Multi-Schema Mapping Prompts ([`SchemaMappingPipeline`](file:///Users/david.lobue/Desktop/Agents/PrismRunner/prismrunner/mapping_pipeline.py#L25))

When using the multi-schema catalog (with triage pre-screening and SQLite normalization), you can pass custom prompt templates for both triage and schema extraction:

```python
from prismrunner import SchemaMappingPipeline, SchemaCatalog
from schemas.clinical_records import Diagnosis, Medication, LabResult

catalog = SchemaCatalog()
catalog.register(Diagnosis, description="Clinical diagnoses and conditions")
catalog.register(Medication, description="Prescribed medications, dosages, routes")
catalog.register(LabResult, description="Laboratory test names, values, and flags")

mapper = SchemaMappingPipeline(catalog=catalog)

outputs = mapper.run(
    input_data="data/clinical_notes.parquet",
    output_dir="./outputs/clinical_run",
    # In-memory single-schema extraction prompt template:
    extract_prompt_template="""\
Extract all factual instances of {schema_name}.
Scope: {schema_description}

Source Content:
\"\"\"
{{ text }}
\"\"\"

{schema_block}
""",
    # Custom system prompt:
    system_prompt="Extract exact verbatim clinical facts into strict JSON arrays.",
    # Or point to a custom triage prompt YAML:
    # triage_prompt_file="custom_prompts/my_triage.yaml",
)
```

---

## 5. Interactive Prompt Inspection (`dry_run=True`)

Always test and verify your rendered prompt before generating tokens or running on GPU:

```python
payloads = pipeline.run(
    task="extract",
    input_data="Patient was prescribed Metformin 500mg daily.",
    schema_cls=Medication,
    instruction="Extract only oral medications.",
    dry_run=True,  # Does not load model weights or call GPU
)

# Inspect the exact rendered prompt sent to the LLM:
print(payloads[0]["prompt"])
```

In the CLI, the `--dry-run` flag performs the same preview:
```bash
prismrunner extract \
  --input-text "Patient was prescribed Metformin 500mg daily." \
  --instruction "Extract only oral medications." \
  --dry-run
```
