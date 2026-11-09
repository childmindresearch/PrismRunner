from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class InstructionalApproach(str, Enum):
    """
    Types of instructional approaches used in educational support.
    """

    DIRECT_EXPLICIT = "direct_explicit"
    MULTISENSORY = "multisensory"
    VISUAL = "visual"
    KINESTHETIC = "kinesthetic"
    AUDITORY = "auditory"
    OTHER = "other"


class StrategyType(str, Enum):
    """
    Categories of metacognitive and behavioral strategies.
    """

    METACOGNITIVE = "metacognitive"
    BEHAVIORAL = "behavioral"
    EXECUTIVE_FUNCTION = "executive_function"
    OTHER = "other"


class EducationalSupportPlan(BaseModel):
    """
    A comprehensive plan outlining educational accommodations, instructional strategies,
    and cognitive scaffolding techniques tailored for students requiring additional support.
    """

    instructional_approaches: Optional[List[InstructionalApproach]] = Field(
        default=None,
        description="List of instructional methodologies employed such as direct explicit, multisensory, etc.",
    )
    strategy_types: Optional[List[StrategyType]] = Field(
        default=None,
        description="Types of strategies implemented including metacognitive, behavioral, executive function.",
    )
    modified_assignments: Optional[List[str]] = Field(
        default=None,
        description="Examples of adjusted tasks like reduced length reports or fewer test items.",
    )
    visual_cues: Optional[List[str]] = Field(
        default=None,
        description="Visual indicators or aids used to draw attention or convey information.",
    )
    auditory_methods: Optional[List[str]] = Field(
        default=None,
        description="Methods involving sound or speech, e.g., reading aloud, taped books.",
    )
    kinesthetic_activities: Optional[List[str]] = Field(
        default=None,
        description="Physical movement-based learning activities such as writing motor patterns.",
    )
    goal_setting: Optional[str] = Field(
        default=None,
        description="Description of how goals are framed, e.g., accuracy vs. speed.",
    )
    error_correction_techniques: Optional[List[str]] = Field(
        default=None,
        description="Techniques aimed at increasing error recognition and correction.",
    )
    prediction_practices: Optional[List[str]] = Field(
        default=None,
        description="Activities where students predict performance outcomes and compare with results.",
    )
    behavioral_instruction: Optional[str] = Field(
        default=None,
        description="Guidelines for delivering clear directives to students.",
    )
    material_checklists: Optional[bool] = Field(
        default=None,
        description="Whether daily material checklists are provided to students.",
    )
    oral_directions_limit: Optional[int] = Field(
        default=None,
        description="Maximum number of steps recommended in oral instructions.",
    )
    assessment_category: Optional[str] = Field(
        default=None,
        description="Category of clinical psychometric assessment if applicable.",
    )
