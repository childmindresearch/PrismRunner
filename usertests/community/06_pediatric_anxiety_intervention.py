from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class InterventionType(str, Enum):
    """
    Types of interventions for pediatric anxiety.
    """

    COGNITIVE_BEHAVIORAL_THERAPY = "Cognitive Behavioral Therapy"
    ACCEPTANCE_AND_COMMITMENT_THERAPY = "Acceptance and Commitment Therapy"
    PARENT_BASED_TREATMENT = "Parent-Based Treatment"
    PSYCHIATRIC_PHARMACOTHERAPY = "Psychiatric Pharmacotherapy"
    EDUCATIONAL_ACCOMMODATIONS = "Educational Accommodations"
    EXECUTIVE_FUNCTION_SUPPORT = "Executive Function Support"
    OTHER = "Other"


class AnxietyDisorder(str, Enum):
    """
    Categories of anxiety disorders in children.
    """

    GENERALIZED_ANXIETY_DISORDER = "Generalized Anxiety Disorder"
    SEPARATION_ANXIETY_DISORDER = "Separation Anxiety Disorder"
    SOCIAL_ANXIETY_DISORDER = "Social Anxiety Disorder"
    PANIC_DISORDER = "Panic Disorder"
    SPECIFIC_phobia = "Specific Phobia"
    OTHER = "Other"


class AssessmentTool(str, Enum):
    """
    Standardized tools used for diagnosing pediatric anxiety.
    """

    SCARED = "SCARED"
    DISS = "DISS"
    KIDDIE_SADS = "KIDDIE-SADS"
    OTHER = "Other"


class PediatricAnxietyIntervention(BaseModel):
    """
    A structured model representing the components of pediatric anxiety intervention including therapies, assessments, and support systems.
    """

    intervention_type: Optional[InterventionType] = Field(
        default=None, description="The type of intervention being applied."
    )
    anxiety_disorders: Optional[List[AnxietyDisorder]] = Field(
        default=None, description="List of diagnosed anxiety disorders."
    )
    assessment_tools_used: Optional[List[AssessmentTool]] = Field(
        default=None, description="Tools used for clinical assessment."
    )
    caregiver_support_provided: Optional[bool] = Field(
        default=None, description="Indicates if caregiver support was offered."
    )
    educational_accommodations: Optional[bool] = Field(
        default=None, description="Whether educational accommodations were implemented."
    )
    executive_function_scaffolding: Optional[bool] = Field(
        default=None, description="If scaffolding for executive functions was provided."
    )
    pharmacotherapy_considered: Optional[bool] = Field(
        default=None, description="Whether medication decisions were discussed."
    )
    parent_treatment_program: Optional[str] = Field(
        default=None,
        description="Name or reference to parent-based treatment programs.",
    )
    recommended_strategies: Optional[List[str]] = Field(
        default=None, description="Strategies recommended for managing anxiety."
    )
    key_resources: Optional[Dict[str, str]] = Field(
        default=None, description="Key resources such as books or websites."
    )
    clinical_interview_summary: Optional[str] = Field(
        default=None, description="Summary of clinical interview findings."
    )
    diagnostic_classification: Optional[str] = Field(
        default=None,
        description="Diagnostic classification from structured interviews.",
    )
    treatment_outcome_notes: Optional[str] = Field(
        default=None, description="Notes on treatment outcomes or progress."
    )
    additional_comments: Optional[str] = Field(
        default=None, description="Any other relevant comments or observations."
    )
