"""Scenario 01: a stateless structured workflow with explicit confidence routing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator


class IntakeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)
    source: str = Field(default="api", min_length=1, max_length=40)
    minimum_confidence: float = Field(default=0.75, ge=0, le=1)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Intake text must not be blank")
        return value


class Analysis(BaseModel):
    category: Literal["support", "sales", "risk"]
    summary: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)
    entities: list[str] = Field(default_factory=list, max_length=10)


class WorkflowResult(BaseModel):
    status: Literal["accepted", "needs_review", "rejected", "invalid"]
    category: str | None = None
    summary: str
    entities: list[str] = Field(default_factory=list)


class StructuredState(TypedDict, total=False):
    text: str
    source: str
    minimum_confidence: float
    normalized_text: str
    analysis: dict[str, object]
    validation_error: str
    status: str
    result: dict[str, object]


Analyzer = Callable[[str], Analysis]


def build_graph(analyzer: Analyzer):
    def validate(state: StructuredState) -> dict[str, object]:
        try:
            request = IntakeRequest.model_validate(state)
        except ValidationError as exc:
            return {"validation_error": type(exc).__name__, "status": "invalid"}
        return {
            "normalized_text": " ".join(request.text.split()),
            "source": request.source,
            "minimum_confidence": request.minimum_confidence,
        }

    def route_validation(state: StructuredState) -> Literal["analyze", "invalid"]:
        return "invalid" if state.get("validation_error") else "analyze"

    def analyze(state: StructuredState) -> dict[str, object]:
        decision = analyzer(state["normalized_text"])
        return {"analysis": decision.model_dump()}

    def route_analysis(state: StructuredState) -> Literal["transform", "review", "reject"]:
        analysis = Analysis.model_validate(state["analysis"])
        if analysis.confidence < state["minimum_confidence"]:
            return "review"
        return "reject" if analysis.category == "risk" else "transform"

    def transform(state: StructuredState) -> dict[str, object]:
        analysis = Analysis.model_validate(state["analysis"])
        return {
            "status": "accepted",
            "result": WorkflowResult(
                status="accepted",
                category=analysis.category,
                summary=analysis.summary,
                entities=analysis.entities,
            ).model_dump(),
        }

    def review(state: StructuredState) -> dict[str, object]:
        analysis = Analysis.model_validate(state["analysis"])
        return {
            "status": "needs_review",
            "result": WorkflowResult(
                status="needs_review",
                category=analysis.category,
                summary=analysis.summary,
                entities=analysis.entities,
            ).model_dump(),
        }

    def reject(state: StructuredState) -> dict[str, object]:
        analysis = Analysis.model_validate(state["analysis"])
        return {
            "status": "rejected",
            "result": WorkflowResult(
                status="rejected",
                category=analysis.category,
                summary=analysis.summary,
                entities=analysis.entities,
            ).model_dump(),
        }

    def invalid(state: StructuredState) -> dict[str, object]:
        return {
            "result": WorkflowResult(
                status="invalid", summary=f"Invalid intake: {state['validation_error']}"
            ).model_dump()
        }

    def verify(state: StructuredState) -> dict[str, object]:
        WorkflowResult.model_validate(state["result"])
        return {}

    builder = StateGraph(StructuredState)
    builder.add_node("validate", validate)
    builder.add_node("analyze", analyze)
    builder.add_node("transform", transform)
    builder.add_node("review", review)
    builder.add_node("reject", reject)
    builder.add_node("invalid", invalid)
    builder.add_node("verify", verify)
    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", route_validation)
    builder.add_conditional_edges("analyze", route_analysis)
    builder.add_edge("transform", "verify")
    builder.add_edge("review", "verify")
    builder.add_edge("reject", "verify")
    builder.add_edge("invalid", "verify")
    builder.add_edge("verify", END)
    return builder.compile()
