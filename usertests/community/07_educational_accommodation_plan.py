from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class AccommodationType(str, Enum):
    EXTENDED_TIME = "extended_time"
    WORKSHEET_ACCOMMODATION = "worksheet_accommodation"
    BREAK_SCHEDULE = "break_schedule"
    TASK_BREAKDOWN = "task_breakdown"
    STRUCTURED_CLASSROOM = "structured_classroom"
    ASSISTIVE_TECHNOLOGY = "assistive_technology"
    OTHER = "other"


class ExecutiveFunctionSupport(str, Enum):
    EXTERNAL_PROMPTING = "external_prompting"
    FREQUENT_BREAKS = "frequent_breaks"
    ENVIRONMENTAL_STRUCTURE = "environmental_structure"
    BEHAVIORAL_REGULATION = "behavioral_regulation"
    OTHER = "other"


class EducationalAccommodationPlan(BaseModel):
    """A comprehensive plan outlining educational accommodations and support strategies tailored for students with executive functioning challenges."""

    student_profile: Optional[str] = Field(
        default=None,
        description="Description of the student's learning profile and identified difficulties.",
    )
    accommodation_types: Optional[List[AccommodationType]] = Field(
        default=None,
        description="Types of accommodations implemented to support the student.",
    )
    executive_function_supports: Optional[List[ExecutiveFunctionSupport]] = Field(
        default=None,
        description="Strategies aimed at scaffolding executive function skills.",
    )
    assistive_technology_tools: Optional[List[str]] = Field(
        default=None,
        description="Specific assistive technology tools used, e.g., SnapType Pro.",
    )
    break_schedule: Optional[str] = Field(
        default=None,
        description="Details regarding scheduled breaks during tasks or tests.",
    )
    task_management_strategies: Optional[str] = Field(
        default=None,
        description="Methods for breaking down large tasks into smaller, manageable units.",
    )
    environmental_structures: Optional[str] = Field(
        default=None,
        description="Structural elements in the classroom or learning environment that aid focus.",
    )
    external_support_mechanisms: Optional[str] = Field(
        default=None,
        description="Mechanisms such as prompts or peer modeling to initiate tasks.",
    )
    behavioral_regulation_approaches: Optional[str] = Field(
        default=None,
        description="Approaches for managing behavior and emotional regulation.",
    )
    caregiver_advocacy_resources: Optional[str] = Field(
        default=None,
        description="Resources or programs designed to support caregivers and advocates.",
    )
    assessment_findings: Optional[str] = Field(
        default=None,
        description="Summary of clinical psychometric assessments related to attention and executive functioning.",
    )
    support_plan_context: Optional[str] = Field(
        default=None,
        description="Contextual information about where and how the support plan is applied.",
    )
    recommended_interventions: Optional[List[str]] = Field(
        default=None,
        description="Recommended interventions based on observed symptoms and needs.",
    )
    strengths_recognition: Optional[str] = Field(
        default=None,
        description="How student strengths are acknowledged and reinforced in the curriculum.",
    )
    implementation_notes: Optional[str] = Field(
        default=None,
        description="Additional notes on implementation or modifications needed.",
    )
