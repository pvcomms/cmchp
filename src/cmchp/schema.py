from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Goal(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    description: str
    priority: float = Field(ge=0.0, le=1.0)
    status: Literal["active", "completed", "blocked"]
    subgoals: list[Goal] = Field(default_factory=list)


Goal.model_rebuild()


class MemoryEntry(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["tool_result", "inference", "user_provided", "observation"]
    ttl_tokens: int | None = None


class OpenToolCall(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    tool_name: str
    arguments: dict
    status: Literal["pending", "in_progress", "awaiting_result"]
    reasoning: str


class DecisionStep(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    step: int
    decision: str
    reasoning: str
    outcome: str | None = None


class AgentState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    persona: str
    constraints: list[str] = Field(default_factory=list)
    behavioral_priors: list[str] = Field(default_factory=list)


class CompressionMeta(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    method: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    model_used: str


class CMCHPPacket(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    cmchp_version: str = "0.1"
    session_id: str
    timestamp: datetime
    source_model: str
    target_model: str
    compression: CompressionMeta
    goals: list[Goal] = Field(default_factory=list)
    working_memory: list[MemoryEntry] = Field(default_factory=list)
    open_tool_calls: list[OpenToolCall] = Field(default_factory=list)
    agent_state: AgentState
    decision_trace: list[DecisionStep] = Field(default_factory=list)
    continuation_hint: str
    raw_token_budget: int = 4000

    def to_expansion_prompt(self, target_model: str) -> str:
        from .expander import Expander
        return Expander().expand(self, target_model)
