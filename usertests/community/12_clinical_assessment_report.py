from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class TestEnvironmentSetup(str, Enum):
    QUIET_ENVIRONMENT = "quiet environment"
    MINIMAL_DISTRACTIONS = "minimal distractions"
    COMFORTABLE_SETTING = "comfortable setting"
    OTHER = "other"


class TestAdministrationProcedure(str, Enum):
    STANDARDIZED_PROCEDURES = "standardized procedures"
    MODIFIED_PROCEDURES = "modified procedures"
    OTHER = "other"


class ExaminerTrainingLevel(str, Enum):
    BASIC_TRAINING = "basic training"
    ADVANCED_TRAINING = "advanced training"
    SPECIALIZED_TRAINING = "specialized training"
    OTHER = "other"


class TestMaterialProtection(str, Enum):
    PLASTIC_SHEET_PROTECTORS = "plastic sheet protectors"
    HAND_SANITIZER_PROTOCOL = "hand sanitizer protocol"
    GLOVE_USAGE = "glove usage"
    OTHER = "other"


class SessionDuration(str, Enum):
    TWO_SESSIONS = "two sessions"
    THREE_SESSIONS = "three sessions"
    OTHER = "other"


class MedicalHistoryStatus(str, Enum):
    UNREMARKABLE = "unremarkable"
    REMARKABLE = "remarkable"
    OTHER = "other"


class AccommodationType(str, Enum):
    FLEXIBLE_SETTING = "flexible setting"
    CLEAR_DESK = "clear desk"
    SEPARATE_LOCATION = "separate location"
    OTHER = "other"


class ExecutiveFunctionSupport(str, Enum):
    TIME_ALLOCATION = "time allocation"
    ATTENTION_MAINTENANCE = "attention maintenance"
    COGNITIVE_SCAFFOLDING = "cognitive scaffolding"
    OTHER = "other"


class ClinicalAssessmentReport(BaseModel):
    """A comprehensive report detailing clinical psychometric assessment procedures, environmental setup, examiner qualifications, and developmental history for diagnostic classification."""

    test_environment_setup: Optional[List[TestEnvironmentSetup]] = Field(
        default=None,
        description="List of environmental setup elements such as spacing of furniture and noise control.",
    )
    test_administration_procedure: Optional[TestAdministrationProcedure] = Field(
        default=None, description="Type of procedure used during test administration."
    )
    examiner_training_level: Optional[ExaminerTrainingLevel] = Field(
        default=None,
        description="Level of examiner training applied during assessment.",
    )
    test_material_protection: Optional[List[TestMaterialProtection]] = Field(
        default=None,
        description="Measures taken to sanitize and protect test materials.",
    )
    session_duration: Optional[SessionDuration] = Field(
        default=None,
        description="Number of sessions required to complete the assessment.",
    )
    medical_history_status: Optional[MedicalHistoryStatus] = Field(
        default=None,
        description="Overall status of the patient's medical history relevant to assessment.",
    )
    accommodations_and_technologies: Optional[List[AccommodationType]] = Field(
        default=None,
        description="Types of educational accommodations or assistive technologies recommended.",
    )
    executive_function_support: Optional[List[ExecutiveFunctionSupport]] = Field(
        default=None,
        description="Cognitive supports provided to aid in executive function performance.",
    )
    diagnosis_criteria_met: Optional[bool] = Field(
        default=None,
        description="Indicates whether diagnostic criteria for ADHD or similar disorders were met.",
    )
    attention_monitoring_required: Optional[bool] = Field(
        default=None,
        description="Indicates if ongoing monitoring of attention is needed.",
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any other relevant information or observations from the assessment process.",
    )
    assessment_sessions_completed: Optional[int] = Field(
        default=None,
        description="Total number of sessions actually completed for the assessment.",
    )
    test_completion_context: Optional[str] = Field(
        default=None,
        description="Contextual details about where and how the tests were administered.",
    )
    key_findings_summary: Optional[str] = Field(
        default=None,
        description="Summary of key findings regarding cognitive abilities and academic skills.",
    )
