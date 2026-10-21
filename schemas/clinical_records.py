"""Sample domain schemas for clinical and medical document extraction."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    """Clinical diagnoses, medical conditions, and clinical impressions."""

    condition: str = Field(description="Name of the diagnosis or clinical condition")
    status: Literal["active", "resolved", "suspected", "rule_out"] = Field(
        default="active",
        description="Current clinical status of the condition",
    )
    icd10_code: Optional[str] = Field(
        default=None,
        description="Standard ICD-10 diagnostic code if mentioned",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Clinical context, onset timing, or severity remarks",
    )


class Medication(BaseModel):
    """Prescribed drugs, pharmaceuticals, dosages, and administration schedules."""

    drug_name: str = Field(description="Generic or brand pharmaceutical name")
    dosage: str = Field(description="Dosage strength or amount (e.g. 10mg, 2 puffs)")
    frequency: str = Field(description="Dosing schedule (e.g. daily, BID, Q4H, as needed)")
    route: Optional[str] = Field(
        default="oral",
        description="Administration route (e.g. oral, IV, topical, inhaler)",
    )
    prn: bool = Field(
        default=False,
        description="True if medication is taken as-needed (PRN)",
    )


class LabResult(BaseModel):
    """Laboratory blood tests, diagnostics, measured values, and reference flags."""

    test_name: str = Field(description="Name of the laboratory test (e.g. Hemoglobin, Glucose, ALT)")
    value: float = Field(description="Quantitative numeric test value")
    unit: str = Field(description="Unit of measurement (e.g. mg/dL, g/L, mmol/L)")
    flag: Optional[Literal["normal", "high", "low", "critical"]] = Field(
        default=None,
        description="Laboratory reference flag",
    )


class Allergy(BaseModel):
    """Known drug allergies, adverse reactions, and environmental sensitivities."""

    allergen: str = Field(description="Substance, medication, or material causing reaction")
    reaction: str = Field(description="Reported adverse reaction (e.g. hives, anaphylaxis, rash)")
    severity: Literal["mild", "moderate", "severe"] = Field(
        default="moderate",
        description="Severity classification of the allergic reaction",
    )
