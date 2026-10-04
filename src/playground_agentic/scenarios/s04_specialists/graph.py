"""Scenario 04: structured routing to isolated specialists executed with Send."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field


class SpecialistRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5_000)
    specialists: list[str] = Field(default_factory=list, max_length=5)


class SpecialistReport(BaseModel):
    specialist: str
    conclusion: str
    recommendation: Literal["approve", "revise", "reject"]


class SpecialistResult(BaseModel):
    status: Literal["consensus", "needs_resolution"]
    reports: list[SpecialistReport]
    answer: str


class SpecialistState(TypedDict, total=False):
    question: str
    requested_specialists: list[str]
    selected_specialists: list[str]
    specialist: str
    reports: Annotated[list[dict[str, str]], operator.add]
    conflict: bool
    status: str
    answer: str


class SpecialistWorkerState(TypedDict, total=False):
    """Private state of one per-invocation specialist subgraph."""

    question: str
    specialist: str
    conclusion: str
    reports: list[dict[str, str]]


Router = Callable[[str], list[str]]


def _recommendation(content: str) -> Literal["approve", "revise", "reject"]:
    prefix = content.strip().lower()
    if prefix.startswith("reject"):
        return "reject"
    if prefix.startswith("approve"):
        return "approve"
    return "revise"


def build_graph(
    model: BaseChatModel,
    router: Router,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    specialist_agent = create_agent(
        model,
        tools=[],
        system_prompt=(
            "You are a narrowly scoped specialist. Start with APPROVE, REVISE, or REJECT, "
            "then provide one concise reason. Do not solve another specialist's scope."
        ),
        middleware=[ModelCallLimitMiddleware(run_limit=2, exit_behavior="error")],
        name="parallel_specialist",
    )

    def route(state: SpecialistState) -> dict[str, object]:
        request = SpecialistRequest(
            question=state["question"], specialists=state.get("requested_specialists", [])
        )
        selected = request.specialists or router(request.question)
        allowed = [
            name for name in dict.fromkeys(selected) if name in {"product", "delivery", "security"}
        ]
        if not allowed:
            raise ValueError("Router selected no allowed specialist")
        return {"selected_specialists": allowed, "reports": []}

    def dispatch(state: SpecialistState) -> list[Send]:
        return [
            Send("specialist", {"question": state["question"], "specialist": name})
            for name in state["selected_specialists"]
        ]

    def call_specialist(state: SpecialistWorkerState) -> dict[str, object]:
        role = state["specialist"]
        result = specialist_agent.invoke(
            {"messages": [HumanMessage(content=f"SPECIALIST={role}\nQUESTION={state['question']}")]}
        )
        message = next(item for item in reversed(result["messages"]) if isinstance(item, AIMessage))
        return {"conclusion": str(message.content)}

    def format_report(state: SpecialistWorkerState) -> dict[str, object]:
        content = state["conclusion"]
        report = SpecialistReport(
            specialist=state["specialist"],
            conclusion=content,
            recommendation=_recommendation(content),
        )
        return {"reports": [report.model_dump()]}

    def aggregate(state: SpecialistState) -> dict[str, object]:
        reports = [SpecialistReport.model_validate(item) for item in state["reports"]]
        conflict = len({item.recommendation for item in reports}) > 1
        return {"conflict": conflict}

    def route_conflict(state: SpecialistState) -> Literal["resolve", "finalize"]:
        return "resolve" if state["conflict"] else "finalize"

    def resolve(state: SpecialistState) -> dict[str, object]:
        reports = [SpecialistReport.model_validate(item) for item in state["reports"]]
        answer = "Disagreement preserved: " + "; ".join(
            f"{item.specialist}={item.recommendation}" for item in reports
        )
        return {"status": "needs_resolution", "answer": answer}

    def finalize(state: SpecialistState) -> dict[str, object]:
        reports = [SpecialistReport.model_validate(item) for item in state["reports"]]
        recommendation = reports[0].recommendation
        return {
            "status": "consensus",
            "answer": f"Specialists agree: {recommendation}.",
        }

    specialist_builder = StateGraph(SpecialistWorkerState)
    specialist_builder.add_node("call_agent", call_specialist)
    specialist_builder.add_node("format_report", format_report)
    specialist_builder.add_edge(START, "call_agent")
    specialist_builder.add_edge("call_agent", "format_report")
    specialist_builder.add_edge("format_report", END)
    # The default per-invocation mode inherits the parent's checkpointer while keeping
    # independent parallel specialist calls isolated.
    specialist_workflow = specialist_builder.compile()

    def run_specialist_subgraph(state: SpecialistState) -> dict[str, object]:
        """Map parent state into the private schema and expose only the report."""

        result = specialist_workflow.invoke(
            {"question": state["question"], "specialist": state["specialist"]}
        )
        return {"reports": result["reports"]}

    builder = StateGraph(SpecialistState)
    builder.add_node("route", route)
    builder.add_node("specialist", run_specialist_subgraph)
    builder.add_node("aggregate", aggregate)
    builder.add_node("resolve", resolve)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", dispatch, ["specialist"])
    builder.add_edge("specialist", "aggregate")
    builder.add_conditional_edges("aggregate", route_conflict)
    builder.add_edge("resolve", END)
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


def run(request: SpecialistRequest, *, compiled_graph) -> SpecialistResult:
    state = compiled_graph.invoke(
        {"question": request.question, "requested_specialists": request.specialists, "reports": []}
    )
    return SpecialistResult(
        status=state["status"],
        reports=[SpecialistReport.model_validate(item) for item in state["reports"]],
        answer=state["answer"],
    )
