from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class CopingStrategyType(Enum):
    """Types of coping strategies used in behavioral regulation."""

    RELAXATION_TECHNIQUE = "Relaxation Technique"
    PHYSICAL_EXERCISE = "Physical Exercise"
    SPIRITUAL_MANTRA = "Spiritual Mantra"
    JOURNALING = "Journaling"
    ROUTINE_MAINTENANCE = "Routine Maintenance"
    OTHER = "Other"


class BehavioralRegulationPlan(BaseModel):
    """A comprehensive plan for managing emotional regulation and behavioral responses through structured interventions."""

    person_name: Optional[str] = Field(
        default=None, description="Name of the individual involved in the plan."
    )

    coping_strategies: Optional[List[CopingStrategyType]] = Field(
        default=None,
        description="List of coping strategies tailored to the individual's needs.",
    )

    relaxation_techniques: Optional[List[str]] = Field(
        default=None,
        description="Specific relaxation techniques recommended for stress reduction.",
    )

    physical_exercises: Optional[List[str]] = Field(
        default=None,
        description="Recommended physical exercises to alleviate anxiety and promote calmness.",
    )

    spiritual_mantras: Optional[List[str]] = Field(
        default=None,
        description="Mantras or affirmations intended to focus the mind and reduce anxiety.",
    )

    journaling_prompts: Optional[List[str]] = Field(
        default=None,
        description="Guided prompts for journaling to process thoughts and emotions.",
    )

    safe_spaces: Optional[List[str]] = Field(
        default=None,
        description="Locations where the individual can retreat to calm down or escape overwhelming situations.",
    )

    routine_elements: Optional[List[str]] = Field(
        default=None,
        description="Essential elements of daily routines that foster a sense of security.",
    )

    assessment_notes: Optional[str] = Field(
        default=None,
        description="Notes regarding clinical assessments or diagnostic classifications relevant to the plan.",
    )

    intervention_goals: Optional[List[str]] = Field(
        default=None,
        description="Primary objectives of the behavioral intervention approach.",
    )

    additional_supports: Optional[List[str]] = Field(
        default=None,
        description="Supplementary supports such as assistive technologies or accommodations.",
    )

    stress_reduction_methods: Optional[List[str]] = Field(
        default=None, description="Methods aimed at reducing overall stress levels."
    )

    muscle_tension_relief: Optional[bool] = Field(
        default=None,
        description="Indicates whether progressive muscle relaxation is part of the regimen.",
    )

    self_soothing_skills: Optional[bool] = Field(
        default=None,
        description="Whether self-soothing skills training is included in the plan.",
    )

    reward_consequence_system: Optional[bool] = Field(
        default=None,
        description="Indicates implementation of reward and consequence systems for behavior management.",
    )
