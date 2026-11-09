from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class PhonologicalProcessingDomain(str, Enum):
    PHONOLOGICAL_AWARENESS = "phonological awareness"
    PHONOLOGICAL_MEMORY = "phonological memory"
    ORTHOGRAPHIC_PROCESSING = "orthographic processing"
    OTHER = "other"


class SpeechAssessmentResult(str, Enum):
    NORMAL_SPEECH = "normal speech"
    ARTICULATION_DIFFICULTY = "articulation difficulty"
    LISPS = "lisps"
    OTHER = "other"


class EducationalSupportProgram(BaseModel):
    """Represents an educational support program designed to assist students with learning challenges."""

    name: Optional[str] = Field(
        default=None, description="Name of the educational support program."
    )
    description: Optional[str] = Field(
        default=None, description="Description of the program's purpose and components."
    )
    target_skills: Optional[List[str]] = Field(
        default=None,
        description="Skills targeted by the program such as phonological awareness or spelling.",
    )
    accommodations: Optional[List[str]] = Field(
        default=None, description="Specific accommodations provided by the program."
    )
    assistive_technologies: Optional[List[str]] = Field(
        default=None,
        description="Technologies used in the program such as text-to-speech apps.",
    )
    intervention_approach: Optional[str] = Field(
        default=None,
        description="General approach taken by the program (e.g., multi-modal, structured).",
    )
    focus_areas: Optional[List[str]] = Field(
        default=None, description="Areas of focus like reading, writing, or vocabulary."
    )
    recommended_for: Optional[List[str]] = Field(
        default=None,
        description="Student profiles or conditions the program is recommended for.",
    )
    duration: Optional[str] = Field(
        default=None, description="Duration or time frame of the program."
    )
    materials_needed: Optional[List[str]] = Field(
        default=None, description="Materials required to implement the program."
    )
    assessment_tools: Optional[List[str]] = Field(
        default=None, description="Tools used for assessing student progress."
    )
    progress_checkpoints: Optional[List[str]] = Field(
        default=None, description="Methods or intervals for checking student progress."
    )
    customization_options: Optional[List[str]] = Field(
        default=None, description="Options available for customizing the program."
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional notes regarding implementation or effectiveness.",
    )


class ClinicalAssessment(BaseModel):
    """Captures clinical psychometric assessments related to phonological processing and speech sound disorders."""

    assessment_type: Optional[str] = Field(
        default=None, description="Type of assessment performed (e.g., CTOPP-2, RAS)."
    )
    domains_assessed: Optional[List[PhonologicalProcessingDomain]] = Field(
        default=None,
        description="Domains assessed in the phonological processing evaluation.",
    )
    results_summary: Optional[str] = Field(
        default=None, description="Summary of key findings from the assessment."
    )
    speech_sound_disorder: Optional[SpeechAssessmentResult] = Field(
        default=None,
        description="Result indicating presence or absence of speech sound disorder.",
    )
    phonological_processing_score: Optional[str] = Field(
        default=None,
        description="Score or level of phonological processing proficiency.",
    )
    reading_related_findings: Optional[str] = Field(
        default=None,
        description="Findings specifically related to reading performance.",
    )
    recommendations: Optional[List[str]] = Field(
        default=None, description="Recommendations based on the assessment results."
    )
    next_steps: Optional[List[str]] = Field(
        default=None, description="Next steps suggested after the assessment."
    )
    assessment_date: Optional[str] = Field(
        default=None, description="Date when the assessment was conducted."
    )
    evaluator: Optional[str] = Field(
        default=None,
        description="Evaluator or organization responsible for the assessment.",
    )
    supporting_evidence: Optional[List[str]] = Field(
        default=None, description="Evidence supporting the assessment conclusions."
    )
    special_considerations: Optional[str] = Field(
        default=None, description="Special considerations noted during the assessment."
    )
    follow_up_required: Optional[bool] = Field(
        default=None, description="Indicates if a follow-up assessment is needed."
    )
