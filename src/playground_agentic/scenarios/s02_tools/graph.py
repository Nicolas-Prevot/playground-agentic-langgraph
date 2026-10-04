"""Scenario 02: an explicit model/tool loop with a hard step budget."""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    max_steps: int = Field(default=4, ge=1, le=20)


class ChatResult(BaseModel):
    answer: str
    status: Literal["completed", "budget_exhausted"]
    steps: int
    tool_errors: list[str] = Field(default_factory=list)


class ToolPolicy(BaseModel):
    """Limits owned by this tool loop, not by a shared playground runtime."""

    allowed_tools: frozenset[str]
    max_steps: int = Field(default=6, ge=1, le=100)
    max_tool_calls: int = Field(default=8, ge=1, le=100)


class ToolLoopState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int
    tool_calls: int
    max_steps: int
    status: Literal["running", "completed", "budget_exhausted"]
    tool_errors: Annotated[list[str], operator.add]


def build_graph(
    model: Any,
    *,
    tools: Sequence[BaseTool],
    policy: ToolPolicy | None = None,
):
    policy = policy or ToolPolicy(allowed_tools=frozenset(tool.name for tool in tools), max_steps=6)
    allowed = [candidate for candidate in tools if candidate.name in policy.allowed_tools]
    if not allowed:
        raise ValueError("At least one tool must be allowed")
    bound_model = model.bind_tools(allowed)
    tool_node = ToolNode(allowed, handle_tool_errors=True)

    def call_model(state: ToolLoopState) -> dict[str, object]:
        chunks = list(bound_model.stream(state["messages"]))
        if not chunks:
            raise RuntimeError("model stream returned no chunks")
        combined = chunks[0]
        for chunk in chunks[1:]:
            combined += chunk
        response = message_chunk_to_message(combined)
        if not isinstance(response, AIMessage):
            raise TypeError("tool loop requires an AIMessage response")
        return {
            "messages": [response],
            "steps": state.get("steps", 0) + 1,
            "status": "running",
        }

    def call_tools(state: ToolLoopState) -> dict[str, object]:
        result = tool_node.invoke(state)
        messages = result.get("messages", [])
        errors = [
            str(message.content)
            for message in messages
            if isinstance(message, ToolMessage) and message.status == "error"
        ]
        return {
            **result,
            "tool_errors": errors,
            "tool_calls": state.get("tool_calls", 0) + len(messages),
        }

    def route_model(state: ToolLoopState) -> Literal["tools", "done", "budget"]:
        last = state["messages"][-1]
        has_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
        if not has_calls:
            return "done"
        maximum = min(state.get("max_steps", policy.max_steps), policy.max_steps)
        calls = len(last.tool_calls) if isinstance(last, AIMessage) else 0
        exhausted = (
            state.get("steps", 0) >= maximum
            or state.get("tool_calls", 0) + calls > policy.max_tool_calls
        )
        return "budget" if exhausted else "tools"

    def done(state: ToolLoopState) -> dict[str, object]:
        return {"status": "completed"}

    def budget(state: ToolLoopState) -> dict[str, object]:
        last = state["messages"][-1]
        # Preserve the provider's assistant-tool message pairing even when calls are denied.
        denied = [
            ToolMessage(
                content="Not executed: model-step or tool-call budget exhausted.",
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
            for call in (last.tool_calls if isinstance(last, AIMessage) else [])
        ]
        return {
            "status": "budget_exhausted",
            "messages": [*denied, AIMessage(content="Tool loop stopped: budget exhausted.")],
        }

    builder = StateGraph(ToolLoopState)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_node("done", done)
    builder.add_node("budget", budget)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_model)
    builder.add_edge("tools", "model")
    builder.add_edge("done", END)
    builder.add_edge("budget", END)
    return builder.compile()


def run(request: ChatRequest, *, compiled_graph) -> ChatResult:
    state = compiled_graph.invoke(
        {
            "messages": [
                SystemMessage(
                    content=(
                        "Use only the bound tools when they are needed. Never invent a tool "
                        "result or call an unbound tool."
                    )
                ),
                HumanMessage(content=request.message),
            ],
            "steps": 0,
            "max_steps": request.max_steps,
            "status": "running",
            "tool_errors": [],
        },
        config={"recursion_limit": 2 * request.max_steps + 5},
    )
    final = state["messages"][-1]
    return ChatResult(
        answer=str(final.content),
        status=state["status"],
        steps=state["steps"],
        tool_errors=state.get("tool_errors", []),
    )


def stream_text(request: ChatRequest, *, compiled_graph) -> Iterator[str]:
    """Yield provider token chunks while the explicit graph keeps routing control."""

    inputs: ToolLoopState = {
        "messages": [
            SystemMessage(
                content=(
                    "Use only the bound tools when they are needed. Never invent a tool "
                    "result or call an unbound tool."
                )
            ),
            HumanMessage(content=request.message),
        ],
        "steps": 0,
        "max_steps": request.max_steps,
        "status": "running",
        "tool_errors": [],
    }
    for message, _metadata in compiled_graph.stream(
        inputs, config={"recursion_limit": 2 * request.max_steps + 5}, stream_mode="messages"
    ):
        if isinstance(message, AIMessageChunk) and message.content:
            yield str(message.content)
