from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class PsychometricPerformanceLevel(str, Enum):
    """Categories representing performance levels in psychometric assessments."""

    EXTREMELY_LOW = "Extremely Low"
    LOW_AVERAGE = "Low Average"
    AVERAGE = "Average"
    HIGH_AVERAGE = "High Average"
    VERY_HIGH = "Very High"
    EXTREMELY_HIGH = "Extremely High"
    OTHER = "Other"


class ExecutiveFunctionSymptom(str, Enum):
    """Types of executive dysfunction symptoms observed."""

    IMPAIRED_WORKING_MEMORY = "Impaired Working Memory"
    DEFICIT_IN_ATTENTION = "Deficit in Attention"
    REDUCED_MENTAL_FLEXIBILITY = "Reduced Mental Flexibility"
    SLOW_PROCESSING_SPEED = "Slow Processing Speed"
    OTHER = "Other"


class PsychometricIndex(str, Enum):
    """Standardized indices used in psychometric evaluations."""

    VERBAL_COMPREHENSION_INDEX = "Verbal Comprehension Index"
    VISUAL_SPATIAL_INDEX = "Visual Spatial Index"
    FLUID_REASONING_INDEX = "Fluid Reasoning Index"
    WORKING_MEMORY_INDEX = "Working Memory Index"
    PROCESSING_SPEED_INDEX = "Processing Speed Index"
    GENERAL_ABILITY_INDEX = "General Ability Index"
    FULL_SCALE_IQ = "Full Scale IQ"
    PERFORMANCE_IQ = "Performance IQ"
    VERBAL_IQ = "Verbal IQ"
    OTHER = "Other"


class PsychometricAssessment(BaseModel):
    """Represents a comprehensive psychometric assessment report focusing on cognitive abilities, executive functions, and academic performance."""

    patient_id: Optional[str] = Field(
        default=None, description="Unique identifier for the patient."
    )
    indexs: Optional[List[str]] = Field(default=None, description="Associated indexs.")
    iqs: Optional[List[str]] = Field(default=None, description="Associated iqs.")
    performance_level: Optional[PsychometricPerformanceLevel] = Field(
        default=None,
        description="Overall performance level according to standardized ranges.",
    )
    executive_function_symptoms: Optional[List[ExecutiveFunctionSymptom]] = Field(
        default=None, description="List of identified executive dysfunction symptoms."
    )
    academic_achievement_subtests: Optional[List[str]] = Field(
        default=None, description="Subtests used to evaluate academic progress."
    )
    assessment_tools_used: Optional[List[str]] = Field(
        default=None,
        description="Tools and instruments utilized during the assessment.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional comments or observations from the assessment.",
    )
