from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum


class SupportResourceType(str, Enum):
    TIME_TIMER = "Time Timer"
    BEHAVIOR_CHART = "Behavior Chart"
    WORKSHOP = "Workshop"
    OTHER = "Other"


class CaregiverSupportPlan(BaseModel):
    """
    A comprehensive plan outlining psychoeducational caregiver support and advocacy strategies,
    executive function scaffolding techniques, and behavioral regulation therapies.
    """

    support_resource_type: Optional[SupportResourceType] = Field(
        default=None,
        description="The type of clinical support resource being utilized.",
    )

    resource_description: Optional[str] = Field(
        default=None,
        description="A brief description of the clinical support resource.",
    )

    temporal_measurement_method: Optional[str] = Field(
        default=None,
        description="Method used to measure or track time, such as visual timers or stopwatches.",
    )

    executive_function_strategy: Optional[str] = Field(
        default=None,
        description="Strategy employed to scaffold executive functions, e.g., using clocks or timers.",
    )

    behavioral_regulation_technique: Optional[str] = Field(
        default=None,
        description="Technique used for emotional and behavioral regulation, such as reinforcing effort.",
    )

    family_support_involvement: Optional[str] = Field(
        default=None,
        description="Description of how family support is integrated into the care plan.",
    )

    homework_monitoring_approach: Optional[str] = Field(
        default=None,
        description="Approach taken to monitor the quantity and completion of homework assignments.",
    )

    positive_clinical_outcome: Optional[str] = Field(
        default=None,
        description="Expected or observed positive clinical outcome from the intervention.",
    )

    educational_accommodation: Optional[str] = Field(
        default=None,
        description="Specific educational accommodation implemented for the individual.",
    )

    assistive_technology_used: Optional[str] = Field(
        default=None,
        description="Assistive technology tool used to support learning or behavior.",
    )

    caregiver_training_component: Optional[str] = Field(
        default=None,
        description="Component of training provided to caregivers or educators.",
    )

    reinforcement_strategy: Optional[str] = Field(
        default=None,
        description="Strategy for reinforcing effort and accepting imperfections.",
    )

    communication_tool: Optional[str] = Field(
        default=None,
        description="Tool used for communication between school and home, such as notes or charts.",
    )

    target_population: Optional[List[str]] = Field(
        default=None,
        description="List of target population groups involved in the support plan.",
    )

    additional_notes: Optional[str] = Field(
        default=None,
        description="Any other relevant information about the support plan.",
    )
