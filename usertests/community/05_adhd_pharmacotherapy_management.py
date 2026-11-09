from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class MedicationType(str, Enum):
    STIMULANT = "stimulant"
    ALPHA_AGONIST = "alpha_agonist"
    ATOMOXETINE = "atomoxetine"
    SSRI = "ssri"
    OTHER = "other"


class SideEffect(str, Enum):
    DECREASED_APPETITE = "decreased appetite"
    INSOMNIA = "insomnia"
    LOW_BLOOD_PRESSURE = "low blood pressure"
    INCREASED_HEART_RATE = "increased heart rate"
    MILD_SLEEPINESS = "mild sleepiness"
    OTHER = "other"


class MedicationHolidayStatus(str, Enum):
    RECOMMENDED = "recommended"
    NOT_RECOMMENDED = "not recommended"
    OPTIONAL = "optional"
    OTHER = "other"


class ADHDPharmacotherapy(BaseModel):
    """
    Represents the clinical management protocol for ADHD pharmacotherapy including
    medication types, dosing considerations, monitoring requirements, and side effect profiles.
    """

    medication_type: Optional[MedicationType] = Field(
        default=None, description="Type of ADHD medication prescribed"
    )
    dosage_mg: Optional[float] = Field(
        default=None, description="Dosage of the medication in milligrams"
    )
    side_effects: Optional[List[SideEffect]] = Field(
        default=None, description="List of reported side effects"
    )
    monitoring_requirements: Optional[List[str]] = Field(
        default=None, description="Vital signs or health metrics to monitor"
    )
    onset_time_hours: Optional[float] = Field(
        default=None, description="Time in hours for medication to take effect"
    )
    duration_weeks: Optional[float] = Field(
        default=None, description="Duration in weeks for full therapeutic effect"
    )
    medication_holiday_status: Optional[MedicationHolidayStatus] = Field(
        default=None, description="Recommendation status for medication holidays"
    )
    effectiveness_rating: Optional[str] = Field(
        default=None, description="Rating of medication effectiveness"
    )
    discontinuation_reason: Optional[str] = Field(
        default=None, description="Reason for discontinuing the medication"
    )
    alternative_treatments: Optional[List[str]] = Field(
        default=None, description="Alternative or adjunct treatments considered"
    )
    patient_response_notes: Optional[str] = Field(
        default=None, description="Notes on patient's response to treatment"
    )
    contraindications: Optional[List[str]] = Field(
        default=None, description="Known contraindications or drug interactions"
    )
    special_considerations: Optional[List[str]] = Field(
        default=None,
        description="Special considerations for administration or monitoring",
    )
    follow_up_timeline_days: Optional[int] = Field(
        default=None, description="Days until next follow-up appointment"
    )
    additional_comments: Optional[str] = Field(
        default=None, description="Additional clinical comments or notes"
    )
