#!/usr/bin/env python3
"""User Test 02: DataFrame Extraction, Classification & Long-Text Chunking.

Run directly from the project root:
    python usertests/02_custom_schemas_and_dataframes.py           # Dry-run preview
    python usertests/02_custom_schemas_and_dataframes.py --live    # Live local LLM inference
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import List, Literal, Optional

# Ensure prismrunner is importable when running from repo root or usertests/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from pydantic import BaseModel, Field

from prismrunner import DEFAULT_MODEL_NAME, PrismPipeline, Record, TokenAwareChunker


class MedicationOrder(BaseModel):
    """Prescribed medication extracted from clinical notes."""

    drug_name: str = Field(description="Medication brand or generic name")
    dosage: str = Field(description="Dosage amount (e.g. 500mg, 10mg)")
    frequency: str = Field(description="Administration schedule (e.g. daily, BID, PRN)")
    route: Optional[Literal["oral", "IV", "inhaled", "topical"]] = Field(
        default="oral", description="Route of administration"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PrismRunner DataFrame & Chunking User Test")
    parser.add_argument("--live", action="store_true", help="Load local LLM weights and run live inference")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model ID")
    args = parser.parse_args()

    pipeline = PrismPipeline(model_name=args.model)

    print("=" * 70)
    print("1. PANDAS DATAFRAME EXTRACTION WITH METADATA PRESERVATION")
    print("=" * 70)
    df = pd.DataFrame({
        "patient_id": ["PT-101", "PT-102"],
        "department": ["Endocrinology", "Cardiology"],
        "clinical_note": [
            "Patient started on Metformin 500mg twice daily orally and Lisinopril 10mg daily.",
            "Continue Atorvastatin 40mg oral nightly; added Nitroglycerin 0.4mg sublingual PRN chest pain.",
        ],
    })

    extraction_results = pipeline.run(
        task="extract",
        input_data=df,
        text_column="clinical_note",
        id_column="patient_id",
        schema_cls=MedicationOrder,
        instruction="Extract all prescribed medications and dosages from the clinical note.",
        dry_run=not args.live,
    )
    print(json.dumps(extraction_results, indent=2, default=str))

    print("\n" + "=" * 70)
    print("2. CUSTOM PROMPT VARIABLES IN CLASSIFICATION")
    print("=" * 70)
    headlines = [
        "Federal Reserve signals potential interest rate adjustment in Q4.",
        "Phase 3 clinical trial demonstrates 84% reduction in relapse rates.",
    ]
    classify_results = pipeline.run(
        task="classify",
        input_data=headlines,
        variables={"categories": "Macroeconomics, Biotechnology, Hardware, Other"},
        dry_run=not args.live,
    )
    print(json.dumps(classify_results, indent=2, default=str))

    print("\n" + "=" * 70)
    print("3. SLIDING-WINDOW SENTENCE CHUNKING & SUMMARIZATION")
    print("=" * 70)
    long_report = (
        "Q1 Architecture Review: Migrated inference workloads to native PyTorch SDPA attention. "
        "Padding overhead dropped by 62% after enabling length-bucketed batch collation. "
        "Q2 Reliability Report: Introduced pre-validation JSON sanitization and trailing-comma repair. "
        "Schema validation pass rates reached 99.4% across 10,000 clinical notes. "
        "Q3 Hardware Expansion: Verified zero-copy weight loading across Apple Silicon MPS and NVIDIA Ampere GPUs."
    )

    chunker = TokenAwareChunker(tokenizer=None, chunk_size=25, chunk_overlap=5, strategy="sentence")
    preview_chunks = chunker.chunk_record(Record(id="report_01", index=0, text=long_report))
    print(f"Chunker split long report into {len(preview_chunks)} overlapping sentence chunks:")
    for ch in preview_chunks:
        print(f"  [Chunk {ch.chunk_index + 1}/{ch.total_chunks} | ~{ch.token_count} tokens]: {ch.text}")

    summary_results = pipeline.run(
        task="summarize",
        input_data=long_report,
        chunk=True,
        chunk_size=25,
        chunk_overlap=5,
        dry_run=not args.live,
    )
    print("\nSummarization Pipeline Output:")
    print(json.dumps(summary_results, indent=2, default=str))


if __name__ == "__main__":
    main()

