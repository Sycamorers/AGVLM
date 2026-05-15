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
    accepted_labels: List[str] = Field(default_factory=list)
    synonym_groups: Dict[str, List[str]] = Field(default_factory=dict)
    uncertainty_required: bool = False
    expected_uncertainty: Optional[bool] = None
    crop: Optional[str] = None
    disease: Optional[str] = None
    known_facts: List[str] = Field(default_factory=list)
    allowed_claims: List[str] = Field(default_factory=list)
    visual_evidence: List[str] = Field(default_factory=list)
    unsafe_recommendations: List[str] = Field(default_factory=list)
    preference_score: Optional[float] = None
    preference_rationale: Optional[str] = None
    chosen_response: Optional[str] = None
    rejected_response: Optional[str] = None
    expert_quality_score: Optional[float] = None
    agronomic_correctness_score: Optional[float] = None
    management_usefulness_score: Optional[float] = None
    uncertainty_calibration_score: Optional[float] = None
    safety_score: Optional[float] = None
    weights: Dict[str, float] = Field(default_factory=dict)
