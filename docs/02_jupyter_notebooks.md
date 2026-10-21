# Using PrismRunner in Jupyter Notebooks

This guide provides end-to-end workflows and copy-paste recipes for running PrismRunner interactively inside Jupyter Notebooks, Google Colab, and VS Code Notebooks.

---

## 1. Quickstart Notebook Setup

In your first notebook cell, import the pipeline components:

```python
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from prismrunner import PrismPipeline, Record
from prismrunner.schemas.compiler import compile_schema_to_prompt
```

Initialize the pipeline:
```python
# Automatically selects Apple Silicon MPS, NVIDIA CUDA, or CPU
pipeline = PrismPipeline()
```

---

## 2. In-Cell Pydantic Schema Definition

You can define any Pydantic model directly in a notebook cell. The Dynamic Schema Compiler inspects field types, `Literal` values, constraints, and descriptions:

```python
class AdverseEvent(BaseModel):
    event_name: str = Field(..., description="Name of adverse symptom or reaction")
    severity: Literal["mild", "moderate", "severe"] = Field(..., description="Clinical severity level")
    action_taken: str = Field(..., description="Action taken (e.g. dose reduction, discontinuation)")
    resolved: bool = Field(..., description="True if symptom resolved, False otherwise")
```

### Inspect the Compiled Schema Directive:
```python
compiled = compile_schema_to_prompt(AdverseEvent)
print(compiled.render_prompt_block(as_array=True))
```

---

## 3. Passing Data from Pandas DataFrames

You can run PrismRunner directly on Pandas DataFrames or Parquet/CSV files.

### Option A: From an Existing DataFrame
```python
df = pd.DataFrame({
    "patient_id": ["PT-101", "PT-102", "PT-103"],
    "clinical_note": [
        "Patient experienced moderate rash after second dose of Amoxicillin. Medication was discontinued.",
        "Mild nausea noted following chemotherapy infusion. Antiemetic administered with resolution.",
        "Severe anaphylaxis noted 10 minutes post IV penicillin. Epinephrine administered.",
    ]
})

# Convert to PrismRunner Record objects
records = [
    Record(id=row["patient_id"], index=i, text=row["clinical_note"])
    for i, row in df.iterrows()
]
```

### Option B: Directly from File Path
```python
# Parquet, CSV, JSON, TXT, or Markdown
input_path = "data/patient_notes.parquet"
```

---

## 4. Previewing Before Running (`dry_run=True`)

To verify how your data, prompt template, guidelines, and schema format together before loading the model:

```python
payloads = pipeline.run(
    task="extract",
    input_data=records,
    schema_cls=AdverseEvent,
    instruction="Extract all adverse reactions mentioned in the note.",
    variables={"guidelines": "Only capture symptoms that occurred after drug administration."},
    dry_run=True,
)

# Display rendered prompt for the first record:
print(payloads[0]["prompt"])
```

---

## 5. Running Batch Inference & Exploring Results with Pandas

Execute batch inference and convert extracted Pydantic models directly into a Pandas DataFrame:

```python
results = pipeline.run(
    task="extract",
    input_data=records,
    schema_cls=AdverseEvent,
    batch_size=4,
    temperature=0.0,
    max_new_tokens=512,
)

# Flatten extracted items into a DataFrame
extracted_rows = []
for res in results:
    rec_id = res["record_id"]
    for item in res.get("parsed_items", []):
        row = item if isinstance(item, dict) else item.model_dump()
        row["record_id"] = rec_id
        extracted_rows.append(row)

results_df = pd.DataFrame(extracted_rows)
display(results_df)
```

---

## 6. Long Documents: Token-Budgeted Chunking in Notebooks

When processing long clinical reports, legal contracts, or multi-page documents:

```python
long_notes = [
    Record(id="doc_999", index=0, text="Very long document content..." * 50)
]

results = pipeline.run(
    task="extract",
    input_data=long_notes,
    schema_cls=AdverseEvent,
    chunk=True,             # Enable sliding window chunker
    chunk_size=1024,        # Token budget per chunk
    chunk_overlap=64,       # Token overlap between adjacent chunks
    chunk_strategy="sentence",  # Preserve sentence boundaries
)
```

The **Deterministic Stitcher** will automatically:
- Segment the text into token-budgeted chunks.
- Run inference across all chunks.
- Merge, deduplicate, and assemble extracted entities across chunks back into a single record.

---

## 7. Multi-Schema Relational Extraction in Notebooks

To extract across multiple schemas simultaneously (e.g. Diagnoses, Medications, and Lab Results) and export relational tables directly into SQLite and CSVs:

```python
from prismrunner import SchemaMappingPipeline, SchemaCatalog
from schemas.clinical_records import Diagnosis, Medication, LabResult

# 1. Build catalog
catalog = SchemaCatalog()
catalog.register(Diagnosis, description="Clinical diagnoses and conditions")
catalog.register(Medication, description="Prescribed medications, dosages, routes")
catalog.register(LabResult, description="Laboratory test names, values, and flags")

# 2. Run multi-schema mapper with triage
mapper = SchemaMappingPipeline(catalog=catalog)

summary = mapper.run(
    input_data=records,
    output_dir="./notebook_outputs",
    use_triage=True,     # Fast pre-screening ChatML filter skips irrelevant schemas
    export_formats=["sqlite", "csv", "wide"],
)

print(f"Extraction complete! Relational tables saved to: {summary['output_directory']}")
```

Query the generated SQLite database directly in a notebook cell:
```python
import sqlite3

conn = sqlite3.connect("./notebook_outputs/extracted_database.db")
diagnoses_df = pd.read_sql_query("SELECT * FROM Diagnosis", conn)
display(diagnoses_df)
```
