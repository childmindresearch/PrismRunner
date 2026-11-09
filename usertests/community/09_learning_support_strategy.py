from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum


class StrategyType(str, Enum):
    """
    Types of learning support strategies.
    """

    IMPULSE_CONTROL = "impulse_control"
    ARITHMETIC_COMPUTATION = "arithmetic_computation"
    VISUAL_LEARNING_AID = "visual_learning_aid"
    INSTRUCTIONAL_AID = "instructional_aid"
    OTHER = "other"


class LearningSupportStrategy(BaseModel):
    """
    A model representing educational accommodations, assistive technologies, and cognitive support strategies for learners.
    """

    strategy_type: StrategyType = Field(
        ..., description="The category of the learning support strategy"
    )
    description: Optional[str] = Field(
        None, description="Detailed description of the strategy"
    )
    key_words: Optional[List[str]] = Field(
        None, description="Important keywords associated with the strategy"
    )
    visual_aid: Optional[str] = Field(
        None, description="Description of visual aids used"
    )
    step_sequence: Optional[List[str]] = Field(
        None, description="Sequence of steps for procedural strategies"
    )
    memory_technique: Optional[str] = Field(
        None, description="Memory strategy for algorithm recall"
    )
    color_coding: Optional[str] = Field(
        None, description="Color coding instructions if applicable"
    )
    socialization_opportunity: Optional[str] = Field(
        None, description="Details about social interaction support"
    )
    emotional_regulation_tactic: Optional[str] = Field(
        None, description="Techniques for managing emotions and impulses"
    )
    peer_collaboration_activity: Optional[str] = Field(
        None, description="Peer-based collaborative learning activity"
    )
    conceptual_reference: Optional[str] = Field(
        None, description="Reference material or concept being supported"
    )
    repetition_encouragement: Optional[str] = Field(
        None, description="Encouragement for repeating successful behaviors"
    )
    consequence_evaluation: Optional[str] = Field(
        None, description="Guidance on evaluating outcomes"
    )
    alternative_representation: Optional[str] = Field(
        None, description="Alternative ways to represent information"
    )
    example_application: Optional[str] = Field(
        None, description="Example usage of the strategy"
    )
