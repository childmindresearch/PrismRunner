from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class MoodState(str, Enum):
    EUTHYMIC = "euthymic"
    OTHER = "other"


class ThoughtProcess(str, Enum):
    LOGICAL = "logical"
    FLIGHT_OF_IDEAS = "flight_of_ideas"
    LOOSE_ASSOCIATIONS = "loose_associations"
    OTHER = "other"


class DietaryHabit(str, Enum):
    HEALTHY_CHOICES = "healthy_choices"
    LIMITED_FRIED_FOODS = "limited_fried_foods"
    LIMITED_ADDED_SUGARS = "limited_added_sugars"
    LIMITED_SUGARY_DRINKS = "limited_sugary_drinks"
    OTHER = "other"


class CaregiverSupportStatus(str, Enum):
    ENCOURAGED_CONTACT = "encouraged_contact"
    NO_ACTION_NEEDED = "no_action_needed"
    OTHER = "other"


class SubscriptionPlan(str, Enum):
    ALLEY_MEMBERSHIP = "alley_membership"
    UNLIMITED_BOOKS = "unlimited_books"
    OTHER = "other"


class AssessmentSummary(BaseModel):
    """A comprehensive psychosocial assessment summary capturing clinical findings, dietary behaviors, caregiver support status, and educational accommodations."""

    mood_state: Optional[MoodState] = Field(
        default=None, description="The patient's current mood state."
    )
    thought_process: Optional[ThoughtProcess] = Field(
        default=None, description="Description of the patient's thought process."
    )
    dietary_habits: Optional[List[DietaryHabit]] = Field(
        default=None, description="List of dietary habits observed or recommended."
    )
    caregiver_support_status: Optional[CaregiverSupportStatus] = Field(
        default=None, description="Indicates if caregiver contact is encouraged."
    )
    subscription_plan: Optional[SubscriptionPlan] = Field(
        default=None,
        description="Type of subscription plan associated with assistive tools.",
    )
    contact_information: Optional[str] = Field(
        default=None, description="Contact details for the care provider."
    )
    ally_program_access: Optional[bool] = Field(
        default=None, description="Whether the user has access to the ally program."
    )
    educational_accommodations: Optional[str] = Field(
        default=None,
        description="Details about educational accommodations or assistive technologies used.",
    )
    biopsychosocial_history: Optional[str] = Field(
        default=None, description="Summary of longitudinal developmental history."
    )
    executive_function_support: Optional[str] = Field(
        default=None,
        description="Notes on scaffolding or cognitive support strategies.",
    )
    behavioral_regulation_therapy: Optional[str] = Field(
        default=None,
        description="Information on therapies related to emotional regulation.",
    )
    nutrition_importance: Optional[str] = Field(
        default=None,
        description="Statement on the importance of nutrition in treatment.",
    )
    diagnostic_classification: Optional[str] = Field(
        default=None,
        description="Current diagnostic classification or presumptive diagnosis.",
    )
    target_outcomes: Optional[List[str]] = Field(
        default=None, description="List of targeted outcomes based on assessment."
    )
    additional_notes: Optional[str] = Field(
        default=None, description="Any other relevant notes from the assessment."
    )
