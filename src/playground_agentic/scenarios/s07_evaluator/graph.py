"""Scenario 07: a bounded evaluator-optimizer workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class OptimizationRequest(BaseModel):
    task: str = Field(min_length=1, max_length=5_000)
    instructions: str = Field(default="", max_length=20_000)
    resources: list[str] = Field(default_factory=list, max_length=20)
    minimum_score: float = Field(default=0.8, ge=0, le=1)
    max_iterations: int = Field(default=3, ge=1, le=6)


class Evaluation(BaseModel):
    score: float = Field(ge=0, le=1)
    feedback: str = Field(max_length=1_000)


class OptimizationResult(BaseModel):
    status: Literal["accepted", "partial"]
    content: str
    score: float
    feedback: str
    iterations: int


class OptimizerState(TypedDict, total=False):
    task: str
    instructions: str
    resources: list[str]
    minimum_score: float
    max_iterations: int
    content: str
    score: float
    feedback: str
    iterations: int
    status: str


Generator = Callable[[str, str, list[str], str, int], str]
Evaluator = Callable[[str, str, str, list[str]], Evaluation]


def offline_generator(
    task: str, instructions: str, resources: list[str], feedback: str, iteration: int
) -> str:
    del instructions, resources
    if iteration == 1:
        return f"Added: {task}"
    return (
        f"Outcome: {task}\nCompatibility: no breaking changes.\n"
        "Validation: focused scenario tests passed.\n"
        f"Reviewer feedback addressed: {feedback}"
    )


def offline_evaluator(
    _task: str, content: str, _instructions: str, _resources: list[str]
) -> Evaluation:
    required = ("Outcome:", "Compatibility:", "Validation:")
    missing = [item for item in required if item not in content]
    if missing:
        return Evaluation(score=0.45, feedback=f"Add {', '.join(missing)}")
    return Evaluation(score=0.95, feedback="The artifact satisfies the supplied rubric.")


def build_graph(
    generator: Generator = offline_generator,
    evaluator: Evaluator = offline_evaluator,
):
    def initialize(state: OptimizerState) -> dict[str, object]:
        request = OptimizationRequest.model_validate(state)
        return {**request.model_dump(), "iterations": 0, "feedback": ""}

    def generate(state: OptimizerState) -> dict[str, object]:
        iteration = state["iterations"] + 1
        content = generator(
            state["task"],
            state.get("instructions", ""),
            state.get("resources", []),
            state.get("feedback", ""),
            iteration,
        )
        return {"content": content, "iterations": iteration}

    def evaluate(state: OptimizerState) -> dict[str, object]:
        decision = evaluator(
            state["task"],
            state["content"],
            state.get("instructions", ""),
            state.get("resources", []),
        )
        return {"score": decision.score, "feedback": decision.feedback}

    def route_evaluation(state: OptimizerState) -> Literal["accepted", "generate", "partial"]:
        if state["score"] >= state["minimum_score"]:
            return "accepted"
        if state["iterations"] >= state["max_iterations"]:
            return "partial"
        return "generate"

    def accepted(state: OptimizerState) -> dict[str, object]:
        del state
        return {"status": "accepted"}

    def partial(state: OptimizerState) -> dict[str, object]:
        del state
        return {"status": "partial"}

    builder = StateGraph(OptimizerState)
    builder.add_node("initialize", initialize)
    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)
    builder.add_node("accepted", accepted)
    builder.add_node("partial", partial)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges("evaluate", route_evaluation)
    builder.add_edge("accepted", END)
    builder.add_edge("partial", END)
    return builder.compile()


def run(request: OptimizationRequest, *, compiled_graph) -> OptimizationResult:
    state = compiled_graph.invoke(request.model_dump())
    return OptimizationResult(
        status=state["status"],
        content=state["content"],
        score=state["score"],
        feedback=state["feedback"],
        iterations=state["iterations"],
    )
