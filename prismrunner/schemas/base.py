"""Default built-in Pydantic v2 schemas for classification, extraction, and summarization."""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class ClassificationItem(BaseModel):
    """Classification record representing an assigned category and confidence."""

    category: str = Field(description="The primary label or category assigned to the text.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief justification or evidence for the assigned category.",
    )


class EntityItem(BaseModel):
    """Information extraction record representing a named entity or action item."""

    entity: str = Field(description="The extracted entity name, key term, or action item.")
    category: str = Field(description="Classification type of entity (e.g. Person, Org, Action, Date).")
    details: Optional[str] = Field(
        default=None,
        description="Associated metadata, owner, deadline, or descriptive context.",
    )


class SummaryItem(BaseModel):
    """Document synthesis record representing key points and cohesive summary."""

    summary: str = Field(description="A concise narrative summary capturing the main message.")
    key_points: List[str] = Field(
        default_factory=list,
        description="Key takeaways or bullet points highlighting critical findings.",
    )
    sentiment: Optional[str] = Field(
        default=None,
        description="Overall tone or sentiment if applicable (e.g. positive, neutral, negative).",
    )
