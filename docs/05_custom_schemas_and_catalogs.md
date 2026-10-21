# Custom Schemas & Catalog Reference

PrismRunner uses Pydantic v2 to define target data structures. The Dynamic Schema Compiler inspects model definitions at runtime to build concise, human-readable formatting specifications for local models without leaking raw OpenAPI `$defs` or `$ref` blobs.

---

## 1. Authoring Custom Schemas

### Best Practices for Pydantic Models

1. **Provide Clear Descriptions**: Use `Field(description="...")`. The compiler converts these directly into bulleted guidelines for the LLM.
2. **Constrain Values with `Literal` or `Enum`**: When a field must belong to a fixed set of choices, use `Literal` or `Enum`. PrismRunner renders these as `one of ['Val1', 'Val2']`.
3. **Set Numeric Bounds**: Use `ge`, `le`, `gt`, `lt` on numeric fields (e.g. `Field(ge=0.0, le=1.0)`).
4. **Use Optional Types Wisely**: Fields can be `Optional[T] = None`. PrismRunner's parser includes phantom object detection that automatically discards hallucinations where all fields are `None`.

### Example Schema
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class SeverityLevel(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

class LabFinding(BaseModel):
    test_name: str = Field(..., description="Standardized name of the laboratory assay")
    measurement: float = Field(..., description="Numeric measurement value")
    units: str = Field(..., description="Units of measurement (e.g. mg/dL, mmol/L)")
    flag: Optional[Literal["LOW", "NORMAL", "HIGH", "CRITICAL"]] = Field(
        None, description="Reference range interpretation flag"
    )
    severity: Optional[SeverityLevel] = Field(
        None, description="Clinical severity assessment"
    )
```

---

## 2. Dynamic Runtime Schemas (`load_schema_from_string`)

You can compile schemas dynamically from Python code strings at runtime without saving a `.py` file to disk:

```python
from prismrunner.schemas.compiler import load_schema_from_string

schema_code = """
from pydantic import BaseModel, Field

class HardwareComponent(BaseModel):
    name: str = Field(description="Component name (CPU, GPU, RAM, etc.)")
    architecture: str = Field(description="Architecture or fabrication node")
    tdp_watts: int = Field(description="Thermal design power in watts")
"""

DynamicModel = load_schema_from_string(schema_code, "HardwareComponent")
```

---

## 3. Schema Catalogs ([`SchemaCatalog`](file:///Users/david.lobue/Desktop/Agents/PrismRunner/prismrunner/catalog/registry.py#L23))

When mapping unstructured text against multiple candidate schemas (e.g., medical records that may contain Diagnoses, Medications, Lab Tests, and Allergies), register them in a `SchemaCatalog`:

```python
from prismrunner import SchemaCatalog
from schemas.clinical_records import Diagnosis, Medication, LabResult, Allergy

# 1. Initialize catalog
catalog = SchemaCatalog()

# 2. Register schemas with domain descriptions (used by the triage pre-screener)
catalog.register(Diagnosis, description="Clinical diagnoses, conditions, and active problems.")
catalog.register(Medication, description="Prescribed pharmaceuticals, dosages, frequencies, and administration routes.")
catalog.register(LabResult, description="Laboratory test names, quantitative values, units, and abnormal flags.")
catalog.register(Allergy, description="Known patient drug or environmental allergies and allergic reactions.")
```

### Auto-Discovery from Directory
You can also auto-load all Pydantic models from a directory of `.py` files:
```python
catalog = SchemaCatalog(schemas_dir="schemas/clinical/")
print(f"Discovered {len(catalog)} schemas: {[s.name for s in catalog.list_schemas()]}")
```

---

## 4. Vectorized ChatML Triage

When extracting across multiple schemas, running every schema against every chunk causes combinatorial explosion ($N \times M$ LLM passes).

PrismRunner's **Vectorized ChatML Triage** uses a fast single-pass pre-screener to evaluate which schemas are actually relevant to a document chunk, skipping 80–90% of redundant extraction passes:

```mermaid
flowchart TD
    Chunk["Document Chunk"] --> Triage["Schema Triage Filter (Single Pass)"]
    Triage --> Decision{"Relevant Schemas?"}
    Decision -->|Diagnosis| ExtDiag["Extract Diagnosis"]
    Decision -->|Medication| ExtMed["Extract Medication"]
    Decision -->|LabResult (None)| SkipLab["[SKIPPED]"]
    Decision -->|Allergy (None)| SkipAllergy["[SKIPPED]"]
```

---

## 5. Relational Aggregation & Exports

Outputs from multi-schema extraction are automatically normalized into relational tables with deterministic foreign keys (`source_id`, `chunk_index`, `entity_index`).

PrismRunner exports these into three formats simultaneously:

### 1. SQLite Database (`extracted_database.db`)
- Every schema becomes a typed SQL table.
- Primary and foreign keys are created automatically.
- Ready for SQL queries, BI dashboards, and joins.

### 2. Normalized CSVs (`relational_csv/`)
- Generates a separate CSV per schema (`Diagnosis.csv`, `Medication.csv`, etc.).
- Includes `source_id` foreign key for easy joins in Pandas or R.

### 3. Consolidated Wide Table (`consolidated_wide_table.parquet`)
- Exactly 1 row per input document.
- Extracted entities are stored as structured JSON arrays in dedicated columns (`Diagnosis_extracted`, `Medication_extracted`, etc.).
- Accompanied by boolean presence indicators (`Diagnosis_present`, `Medication_present`) for fast cohort filtering.
