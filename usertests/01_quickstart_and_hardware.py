#!/usr/bin/env python3
"""User Test 01: Hardware Detection, Schema Compilation, Parser & Quickstart.

Run directly from the project root:
    python usertests/01_quickstart_and_hardware.py           # Instant dry-run & component checks
    python usertests/01_quickstart_and_hardware.py --live    # Runs live local LLM inference
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

# Ensure prismrunner is importable when running from repo root or usertests/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Literal, Optional
from pydantic import BaseModel, Field

from prismrunner import (
    DEFAULT_MODEL_NAME,
    PrismPipeline,
    UniversalOutputParser,
    compile_schema_to_prompt,
    generate_hardware_advisory,
    get_memory_info,
    get_optimal_device,
    resolve_device_and_dtype,
)


class SupportTicket(BaseModel):
    """Structured schema for triaging incoming customer support tickets."""

    category: Literal["billing", "technical", "account", "feature_request"] = Field(
        description="Primary support ticket routing category"
    )
    urgency: int = Field(ge=1, le=5, description="Urgency score from 1 (low) to 5 (critical)")
    summary: str = Field(description="One-sentence summary of the issue")
    requires_escalation: bool = Field(description="True if production is blocked or payment failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="PrismRunner Quickstart & Hardware User Test")
    parser.add_argument("--live", action="store_true", help="Load local LLM weights and run live inference")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model ID")
    args = parser.parse_args()

    print("=" * 70)
    print("1. HARDWARE & PRECISION AUTO-DETECTION")
    print("=" * 70)
    device_name, dtype = resolve_device_and_dtype()
    device = get_optimal_device()
    mem_info = get_memory_info(device)
    advisory = generate_hardware_advisory(args.model, context_limit=32768, device=device)

    print(f"Detected Device      : {device_name} ({mem_info.get('device_name')})")
    print(f"Optimal Torch Dtype  : {dtype}")
    print(f"Recommended Batch    : {advisory['recommended_batch_size']}")
    print(f"Recommended Chunk    : {advisory['recommended_chunk_size']} tokens")
    print(f"Max Safe Input Budget: {advisory['max_safe_input_tokens']} tokens")

    print("\n" + "=" * 70)
    print("2. DYNAMIC SCHEMA-TO-PROMPT COMPILER")
    print("=" * 70)
    compiled = compile_schema_to_prompt(SupportTicket)
    print(compiled.render_prompt_block(as_array=True))

    print("\n" + "=" * 70)
    print("3. ROBUST OUTPUT PARSER (FENCE STRIPPING + TRAILING COMMA REPAIR)")
    print("=" * 70)
    messy_llm_output = """Here is the extracted JSON:
```json
[
  {"category": null, "urgency": null, "summary": null, "requires_escalation": null},
  {
    "category": "billing",
    "urgency": 5,
    "summary": "Customer charged twice for annual enterprise subscription.",
    "requires_escalation": true,
  }
]
```"""
    parsed = UniversalOutputParser.parse(messy_llm_output, model_cls=SupportTicket)
    print(f"Parse Success: {parsed.success} | Validated Items (null phantom dropped): {len(parsed.items)}")
    print(json.dumps([item.model_dump() for item in parsed.items], indent=2))

    print("\n" + "=" * 70)
    print(f"4. PRISMPIPELINE EXECUTION ({'LIVE MODEL' if args.live else 'DRY RUN'})")
    print("=" * 70)
    pipeline = PrismPipeline(model_name=args.model)
    sample_tickets = [
        "Our checkout API is returning HTTP 502 errors for all users since 9am!",
        "Could you add a dark mode toggle to the analytics export page?",
    ]

    results = pipeline.run(
        task="extract",
        input_data=sample_tickets,
        schema_cls=SupportTicket,
        dry_run=not args.live,
    )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

