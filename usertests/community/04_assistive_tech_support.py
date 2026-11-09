from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class TechnologyType(str, Enum):
    SOFTWARE = "software"
    VIDEOGAME = "videogame"
    APP = "app"
    WEBSITE = "website"
    OTHER = "other"


class FeatureCategory(str, Enum):
    READING_SUPPORT = "reading_support"
    WRITING_ASSISTANCE = "writing_assistance"
    TASK_MANAGEMENT = "task_management"
    TEXT_TO_SPEECH = "text_to_speech"
    HIGHLIGHTING = "highlighting"
    ORGANIZERS = "organizers"
    OTHER = "other"


class PlatformType(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    WEB = "web"
    OTHER = "other"


class AssistiveTechnology(BaseModel):
    """Represents an assistive technology product or service designed to support educational accommodations and cognitive scaffolding."""

    name: Optional[str] = Field(
        default=None, description="Name of the assistive technology"
    )
    description: Optional[str] = Field(
        default=None, description="Description of the technology's purpose and features"
    )
    technology_type: Optional[TechnologyType] = Field(
        default=None, description="Type of technology (e.g., software, videogame)"
    )
    platform_types: Optional[List[PlatformType]] = Field(
        default=None, description="Supported platforms (e.g., mobile, web)"
    )
    price: Optional[str] = Field(
        default=None, description="Pricing information if applicable"
    )
    features: Optional[List[FeatureCategory]] = Field(
        default=None, description="Key functional categories supported"
    )
    url: Optional[str] = Field(
        default=None, description="URL to the official website or app store"
    )
    membership_required: Optional[bool] = Field(
        default=None,
        description="Whether a membership is required to access the service",
    )
    free_version_limit: Optional[str] = Field(
        default=None,
        description="Limitations of the free version, e.g., number of documents",
    )
    accessibility_features: Optional[List[str]] = Field(
        default=None, description="Specific accessibility features offered"
    )
    target_audience: Optional[str] = Field(
        default=None, description="Target demographic or user group"
    )
    review_summary: Optional[str] = Field(
        default=None, description="Summary of professional reviews or ratings"
    )
    additional_resources: Optional[List[str]] = Field(
        default=None,
        description="Links to additional resources like articles or guides",
    )
    supported_languages: Optional[List[str]] = Field(
        default=None, description="Languages supported by the technology"
    )
    developer_organization: Optional[str] = Field(
        default=None,
        description="Organization responsible for developing the technology",
    )
