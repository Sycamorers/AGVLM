"""Reward-oriented schemas."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RewardBreakdown(BaseModel):
    """Per-module reward values."""

    model_config = ConfigDict(extra="forbid")

    total: float
    by_module: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class RewardInput(BaseModel):
    """Minimal information required to score a completion."""

    model_config = ConfigDict(extra="forbid")

    prediction: str
    task_type: str
    target_text: Optional[str] = None
    target_label: Optional[str] = None
    target_labels: List[str] = Field(default_factory=list)
    expected_decision: Optional[str] = None
    required_sections: List[str] = Field(default_factory=list)
    management_keywords: List[str] = Field(default_factory=list)
    forbidden_claims: List[str] = Field(default_factory=list)
    acceptable_answers: List[str] = Field(default_factory=list)
    synonym_groups: Dict[str, List[str]] = Field(default_factory=dict)
    uncertainty_required: bool = False
    weights: Dict[str, float] = Field(default_factory=dict)
