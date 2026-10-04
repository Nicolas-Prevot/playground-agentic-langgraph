"""Scenario 05: dynamic planner/orchestrator-worker workflow with bounded replanning."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field, model_validator


class OrchestrationRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=5_000)
    max_tasks: int = Field(default=5, ge=1, le=12)
    max_iterations: int = Field(default=2, ge=1, le=5)


class PlannedTask(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    objective: str = Field(min_length=1, max_length=500)


class Plan(BaseModel):
    tasks: list[PlannedTask] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def unique_task_ids(self) -> Plan:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Planned task IDs must be unique")
        return self


class Review(BaseModel):
    approved: bool
    feedback: str = Field(max_length=500)


class WorkerArtifact(BaseModel):
    task_id: str
    iteration: int
    content: str


class OrchestrationResult(BaseModel):
    status: Literal["completed", "partial"]
    iterations: int
    task_count: int
    answer: str
    reason: str = ""


class OrchestratorState(TypedDict, total=False):
    goal: str
    max_tasks: int
    max_iterations: int
    iteration: int
    tasks: list[dict[str, str]]
    task: dict[str, str]
    worker_iteration: int
    artifacts: Annotated[list[dict[str, object]], operator.add]
    synthesis: str
    approved: bool
    feedback: str
    status: str
    answer: str
    reason: str


Planner = Callable[[str, str, int], Plan]
Worker = Callable[[PlannedTask], str]
Reviewer = Callable[[str, list[WorkerArtifact]], Review]


def build_graph(
    planner: Planner,
    worker: Worker,
    reviewer: Reviewer,
):
    def initialize(state: OrchestratorState) -> dict[str, object]:
        request = OrchestrationRequest.model_validate(state)
        return {
            "goal": request.goal,
            "max_tasks": request.max_tasks,
            "max_iterations": request.max_iterations,
            "iteration": 0,
            "artifacts": [],
            "feedback": "",
        }

    def plan(state: OrchestratorState) -> dict[str, object]:
        iteration = state["iteration"] + 1
        proposed = planner(state["goal"], state.get("feedback", ""), iteration)
        tasks = proposed.tasks[: state["max_tasks"]]
        if not tasks:
            raise ValueError("Planner returned no tasks")
        return {"iteration": iteration, "tasks": [item.model_dump() for item in tasks]}

    def dispatch(state: OrchestratorState) -> list[Send]:
        return [
            Send(
                "worker",
                {
                    "task": task,
                    "worker_iteration": state["iteration"],
                },
            )
            for task in state["tasks"]
        ]

    def run_worker(state: OrchestratorState) -> dict[str, object]:
        task = PlannedTask.model_validate(state["task"])
        content = worker(task)
        artifact = WorkerArtifact(
            task_id=task.id,
            iteration=state["worker_iteration"],
            content=content,
        )
        return {"artifacts": [artifact.model_dump()]}

    def synthesize(state: OrchestratorState) -> dict[str, object]:
        current = [
            WorkerArtifact.model_validate(item)
            for item in state["artifacts"]
            if item["iteration"] == state["iteration"]
        ]
        content = "\n".join(f"## {item.task_id}\n{item.content}" for item in current)
        return {"synthesis": content}

    def review(state: OrchestratorState) -> dict[str, object]:
        current = [
            WorkerArtifact.model_validate(item)
            for item in state["artifacts"]
            if item["iteration"] == state["iteration"]
        ]
        decision = reviewer(state["goal"], current)
        return {"approved": decision.approved, "feedback": decision.feedback}

    def route_review(state: OrchestratorState) -> Literal["finish", "plan", "partial"]:
        if state["approved"]:
            return "finish"
        if state["iteration"] >= state["max_iterations"]:
            return "partial"
        return "plan"

    def finish(state: OrchestratorState) -> dict[str, object]:
        return {"status": "completed", "answer": state["synthesis"], "reason": ""}

    def partial(state: OrchestratorState) -> dict[str, object]:
        return {
            "status": "partial",
            "answer": state["synthesis"],
            "reason": f"Review not satisfied: {state['feedback']}",
        }

    builder = StateGraph(OrchestratorState)
    builder.add_node("initialize", initialize)
    builder.add_node("plan", plan)
    builder.add_node("worker", run_worker)
    builder.add_node("synthesize", synthesize)
    builder.add_node("review", review)
    builder.add_node("finish", finish)
    builder.add_node("partial", partial)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "plan")
    builder.add_conditional_edges("plan", dispatch, ["worker"])
    builder.add_edge("worker", "synthesize")
    builder.add_edge("synthesize", "review")
    builder.add_conditional_edges("review", route_review)
    builder.add_edge("finish", END)
    builder.add_edge("partial", END)
    return builder.compile()


def run(request: OrchestrationRequest, *, compiled_graph) -> OrchestrationResult:
    state = compiled_graph.invoke(request.model_dump())
    current_tasks = len(state["tasks"])
    return OrchestrationResult(
        status=state["status"],
        iterations=state["iteration"],
        task_count=current_tasks,
        answer=state["answer"],
        reason=state.get("reason", ""),
    )
