"""
app/schemas/models.py
─────────────────────
Pydantic data models and LangGraph state definitions for ScholarAgent AI.

All runtime state flows through `AgentState`; every external API surface is
validated by the Pydantic models below, giving us a single source of truth for
both the HTTP layer and the graph execution layer.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# Domain models (HTTP request / response layer)
# ─────────────────────────────────────────────


class StudentProfile(BaseModel):
    """
    Structured representation of an applicant's academic profile.

    Every field is validated strictly so downstream scoring and eligibility
    checks can trust the data they receive without defensive null-checks.
    """

    gpa: float = Field(
        ...,
        ge=0.0,
        le=4.0,
        description="Cumulative GPA on a 4.0 scale.",
        examples=[3.7],
    )
    major: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Declared academic major or field of study.",
        examples=["Computer Science"],
    )
    residency: str = Field(
        ...,
        min_length=2,
        max_length=80,
        description="US state or country of legal residency.",
        examples=["California"],
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Student's academic or extracurricular interests used for semantic matching.",
        examples=[["machine learning", "open-source", "education equity"]],
    )

    @field_validator("major", "residency", mode="before")
    @classmethod
    def strip_and_title(cls, v: str) -> str:
        """Normalise free-text fields to stripped, title-cased strings."""
        return v.strip().title()

    @field_validator("interests", mode="before")
    @classmethod
    def deduplicate_interests(cls, v: list[str]) -> list[str]:
        """Remove duplicates while preserving insertion order."""
        seen: set[str] = set()
        deduped: list[str] = []
        for item in v:
            normalised = item.strip().lower()
            if normalised not in seen:
                seen.add(normalised)
                deduped.append(item.strip())
        return deduped


class ChatRequest(BaseModel):
    """
    Payload accepted by the POST /api/chat endpoint.

    The `prompt` is the free-text user question; `student_profile` carries the
    structured applicant data that will be injected into every agent prompt.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Natural-language question or instruction from the user.",
    )
    student_profile: StudentProfile = Field(
        ...,
        description="Applicant context used by the agent to personalise scholarship discovery.",
    )
    search_mode: str = Field(
        default="general",
        description="Search scope: 'university_only' restricts to university/government sources, 'general' includes aggregators and nonprofits.",
    )


class ScholarshipMatch(BaseModel):
    """
    A single scholarship candidate returned by the orchestration pipeline.

    Fields mirror the raw metadata stored in the vector database so the
    front-end can render cards without additional parsing.
    """

    scholarship_id: str
    name: str
    provider: str
    url: str | None = None
    match_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite score (0–100 %) computed by the vector-scoring engine.",
    )
    min_gpa: float
    required_residency: str
    award_amount: int = Field(default=0, description="Award amount in USD.")
    eligible: bool
    summary: str | None = None
    why_relevant: str | None = None
    eligibility_explanation: str | None = None
    details: str | None = None
    raw_reason: str | None = None
    evidence: str | None = None


class ChatResponse(BaseModel):
    """
    Envelope returned by POST /api/chat once the LangGraph loop resolves.
    """

    answer: str = Field(..., description="Final analyst answer produced by the agent.")
    matches: list[ScholarshipMatch] = Field(
        default_factory=list,
        description="Ranked scholarship candidates surfaced during the agent run.",
    )
    tool_calls_made: int = Field(
        default=0,
        description="Number of tool-execution cycles the supervisor performed.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary diagnostic data (latency, model used, etc.).",
    )


# ─────────────────────────────────────────────
# LangGraph agent state
# ─────────────────────────────────────────────


class AgentState(TypedDict):
    """
    Mutable execution context threaded through every node in the LangGraph DAG.

    LangGraph uses structural merging on `TypedDict` keys between graph steps.
    The `messages` field uses the `add_messages` reducer so that each node can
    *append* to the conversation without needing to read-then-rewrite the full
    history — this is essential for multi-turn tool-call loops.

    Attributes
    ----------
    messages:
        Annotated sequence of LangChain `AnyMessage` objects.  New messages
        from each node are appended automatically by the `add_messages` reducer.
    student_profile:
        Serialised dict of `StudentProfile` data.  Stored as a plain dict so
        LangGraph can deep-copy and snapshot it across graph checkpoints without
        needing Pydantic installed in every worker.
    tool_call_count:
        Running counter incremented by `call_tool_node` on each invocation.
        Used by the supervisor to enforce a hard cap and prevent infinite loops.
    session_id:
        Optional correlation ID propagated from the HTTP request for tracing.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    student_profile: dict[str, Any]
    tool_call_count: int
    session_id: str | None
    search_mode: str
