"""Pydantic schema for normalized multimodal examples."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agri_vlm.constants import SUPPORTED_SPLITS, SUPPORTED_TASK_TYPES


class MessageContent(BaseModel):
    """One multimodal content block inside a chat message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image"]
    text: Optional[str] = None
    image: Optional[str] = None

    @model_validator(mode="after")
    def validate_payload(self) -> "MessageContent":
        if self.type == "text" and not self.text:
            raise ValueError("Text message content requires `text`.")
        if self.type == "image" and not self.image:
            raise ValueError("Image message content requires `image`.")
        return self


class Message(BaseModel):
    """A single chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: List[MessageContent]

    @field_validator("content")
    @classmethod
    def validate_non_empty_content(cls, value: List[MessageContent]) -> List[MessageContent]:
        if not value:
            raise ValueError("Message content cannot be empty.")
        return value


class Target(BaseModel):
    """Canonical supervised target."""

    model_config = ConfigDict(extra="forbid")

    answer_text: Optional[str] = None
    canonical_label: Optional[str] = None
    canonical_labels: List[str] = Field(default_factory=list)
    decision: Optional[Literal["clarify", "respond"]] = None
    structured: Dict[str, Any] = Field(default_factory=dict)
    acceptable_answers: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target_not_empty(self) -> "Target":
        populated = [
            bool(self.answer_text),
            bool(self.canonical_label),
            bool(self.canonical_labels),
            bool(self.decision),
            bool(self.structured),
            bool(self.acceptable_answers),
        ]
        if not any(populated):
            raise ValueError("Target must contain at least one supervised field.")
        return self


class Verifier(BaseModel):
    """Verifier payload for deterministic or semi-deterministic rewards."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "exact_match", "label", "synonym", "structured", "clarify"] = "none"
    accepted_answers: List[str] = Field(default_factory=list)
    accepted_labels: List[str] = Field(default_factory=list)
    synonyms: Dict[str, List[str]] = Field(default_factory=dict)
    required_sections: List[str] = Field(default_factory=list)
    expected_decision: Optional[Literal["clarify", "respond"]] = None
    management_keywords: List[str] = Field(default_factory=list)
    forbidden_claims: List[str] = Field(default_factory=list)
    uncertainty_required: bool = False


class RewardMeta(BaseModel):
    """Reward routing and weighting metadata."""

    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(default_factory=dict)
    task_difficulty: Optional[str] = None
    allow_clarification: bool = False
    structured_output_required: bool = False


class UnifiedSample(BaseModel):
    """Unified JSONL schema row."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source_dataset: str
    task_type: str
    split: str
    images: List[str]
    messages: List[Message]
    target: Target
    metadata: Dict[str, Any] = Field(default_factory=dict)
    verifier: Verifier = Field(default_factory=Verifier)
    reward_meta: RewardMeta = Field(default_factory=RewardMeta)

    @field_validator("sample_id", "source_dataset")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Required string field cannot be empty.")
        return value.strip()

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value: str) -> str:
        if value not in SUPPORTED_TASK_TYPES:
            raise ValueError("Unsupported task type: %s" % value)
        return value

    @field_validator("split")
    @classmethod
    def validate_split(cls, value: str) -> str:
        if value not in SUPPORTED_SPLITS:
            raise ValueError("Unsupported split: %s" % value)
        return value

    @field_validator("images")
    @classmethod
    def validate_images(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Each sample must include at least one image path.")
        if any(not item for item in value):
            raise ValueError("Image paths must be non-empty strings.")
        return value

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: List[Message]) -> List[Message]:
        if not value:
            raise ValueError("Each sample must include at least one message.")
        return value
