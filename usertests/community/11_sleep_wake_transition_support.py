from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class SleepWakeTransitionSupport(BaseModel):
    """
    A comprehensive model for supporting healthy sleep-wake transitions through behavioral, environmental, and therapeutic interventions.
    """

    person_name: Optional[str] = Field(
        default=None, description="Name of the individual receiving support"
    )
    consistency_practice: Optional[str] = Field(
        default=None, description="Reinforces body's sleep-wake cycle"
    )
    natural_light_exposure: Optional[bool] = Field(
        default=None, description="Exposure to natural light and fresh air"
    )
    routine_activities: Optional[List[str]] = Field(
        default=None, description="Activities done nightly to signal bedtime"
    )
    leisure_activity_buffer: Optional[bool] = Field(
        default=None, description="Minutes of downtime between activities"
    )
    relaxation_activities: Optional[List[str]] = Field(
        default=None, description="Activities promoting sleep transition"
    )
    electronic_device_removal: Optional[bool] = Field(
        default=None, description="Removal of electronic devices during relaxation"
    )
    sensory_interventions: Optional[List[str]] = Field(
        default=None, description="Interventions affecting sensory input"
    )
    environmental_conditions: Optional[Dict[str, str]] = Field(
        default=None, description="Room conditions for optimal sleep"
    )
    medication_consideration: Optional[str] = Field(
        default=None, description="Potential pharmacotherapy options if needed"
    )
    caregiver_support: Optional[bool] = Field(
        default=None, description="Support provided to caregivers"
    )
    educational_accommodations: Optional[List[str]] = Field(
        default=None, description="Accommodations for academic settings"
    )
    assistive_devices: Optional[List[str]] = Field(
        default=None, description="Devices aiding independence and performance"
    )
    extended_time_accommodation: Optional[bool] = Field(
        default=None, description="Extended time for assessments or tasks"
    )
    other_support_strategies: Optional[List[str]] = Field(
        default=None, description="Additional strategies not covered above"
    )
