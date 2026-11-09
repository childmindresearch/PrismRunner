from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class FamilyHistoryCondition(str, Enum):
    GENERALIZED_ANXIETY_DISORDER = "generalized anxiety disorder"
    DEPRESSION = "depression"
    SUBSTANCE_ABUSE = "substance abuse"
    POSTTRAUMATIC_STRESS_DISORDER = "posttraumatic stress disorder"
    PSYCHOSIS = "psychosis"
    SUICIDALITY = "suicidality"
    EATING_DISORDER = "eating disorder"
    OBSESSIVE_COMPULSIVE_DISORDER = "obsessive compulsive disorder"
    BIPOLAR_DISORDER = "bipolar disorder"
    SCHIZOPHRENIA = "schizophrenia"
    OTHER = "other"


class DevelopmentalHistory(BaseModel):
    """Represents longitudinal biopsychosocial developmental history including family background, onset of symptoms, and environmental factors."""

    family_history_conditions: Optional[List[FamilyHistoryCondition]] = Field(
        default=None,
        description="List of psychiatric conditions in first, second, and third-degree relatives.",
    )
    symptom_onset_age: Optional[int] = Field(
        default=None, description="Age when symptoms were first noted."
    )
    symptom_duration: Optional[str] = Field(
        default=None,
        description="Duration of symptoms (e.g., since childhood, since age X).",
    )
    symptom_impairment_settings: Optional[List[str]] = Field(
        default=None,
        description="Settings where symptoms cause functional impairment (e.g., home, school, family).",
    )
    family_structure: Optional[str] = Field(
        default=None,
        description="Description of family structure (e.g., intact, bilingual household).",
    )
    family_functioning: Optional[str] = Field(
        default=None, description="Overall family functioning description."
    )
    environmental_stressors: Optional[str] = Field(
        default=None,
        description="Environmental stressors experienced during development (e.g., pandemic, trauma).",
    )
    medical_conditions: Optional[List[str]] = Field(
        default=None,
        description="Medical conditions relevant to developmental history (e.g., KBG syndrome).",
    )
    language_background: Optional[List[str]] = Field(
        default=None, description="Languages spoken in the household."
    )
    referral_source: Optional[str] = Field(
        default=None,
        description="How the family became involved in the study or care process.",
    )
    caregiver_concerns: Optional[str] = Field(
        default=None,
        description="Primary concerns expressed by caregivers regarding the individual's development.",
    )
    behavioral_patterns: Optional[List[str]] = Field(
        default=None,
        description="Noted behavioral patterns or incidents (e.g., wetting accidents, outbursts).",
    )
    symptom_context: Optional[str] = Field(
        default=None,
        description="Contextual information about when symptoms typically occur.",
    )
    impact_on_daily_life: Optional[str] = Field(
        default=None, description="Impact of symptoms on daily life activities."
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional relevant notes about developmental history.",
    )
