"""Scenario 03: bounded conversational agent with separate short/long-term memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

_DEFAULT = object()


@dataclass(frozen=True)
class MemoryContext:
    user_id: str


class MemoryState(TypedDict, total=False):
    owner_user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    profile: dict[str, str]
    summary: str


class ChatTurn(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    thread_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=10_000)


class TurnResult(BaseModel):
    answer: str
    profile: dict[str, str]
    message_count: int
    summary: str = ""


class MemoryPolicy(BaseModel):
    max_messages: int = Field(default=8, ge=2, le=100)
    max_summary_chars: int = Field(default=4_000, ge=100, le=50_000)


def extract_profile(messages: list[BaseMessage], current: dict[str, str]) -> dict[str, str]:
    """Extract a deliberately small, application-owned allowlist of durable facts."""

    updated = dict(current)
    human = next((item for item in reversed(messages) if isinstance(item, HumanMessage)), None)
    if human:
        text = str(human.content)
        patterns = {
            "name": r"\bmy name is\s+([\w-]+)",
            "preferred_language": r"\bI prefer\s+([A-Za-z][\w+#.-]*)\s+examples\b",
            "timezone": r"\bmy timezone is\s+([A-Za-z][\w./+-]+)",
            "update_channel": r"\bI prefer\s+(email|SMS|Slack)\s+updates\b",
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                updated[field] = match.group(1).rstrip(".,;")
        style = re.search(r"\b(?:keep\s+)?(?:your\s+)?answers\s+(concise|detailed)\b", text, re.I)
        if style:
            updated["response_style"] = style.group(1).lower()
    return updated


def build_graph(
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver[Any] | object | None = _DEFAULT,
    store: BaseStore | object | None = _DEFAULT,
    policy: MemoryPolicy | None = None,
):
    resolved_checkpointer = InMemorySaver() if checkpointer is _DEFAULT else checkpointer
    resolved_store = InMemoryStore() if store is _DEFAULT else store
    policy = policy or MemoryPolicy()
    chat_agent = create_agent(
        model,
        tools=[],
        system_prompt="You are a concise assistant. Respect the supplied user profile.",
        middleware=[ModelCallLimitMiddleware(run_limit=2, exit_behavior="error")],
        name="memory_chat_agent",
    )

    def load_memory(state: MemoryState, runtime: Runtime[MemoryContext]) -> dict[str, object]:
        owner = state.get("owner_user_id")
        if owner is not None and owner != runtime.context.user_id:
            raise PermissionError("Conversation belongs to a different user")
        if runtime.store is None:
            raise RuntimeError("Chat with Memories requires a LangGraph Store")
        item = runtime.store.get(("users", runtime.context.user_id), "profile")
        return {
            "profile": dict(item.value) if item else {},
            "owner_user_id": runtime.context.user_id,
        }

    def call_chat_agent(state: MemoryState) -> dict[str, object]:
        summary = state.get("summary", "")
        context_message = HumanMessage(
            content=(
                "READ_ONLY_MEMORY_CONTEXT\n"
                f"PROFILE={json.dumps(state.get('profile', {}))}\n"
                f"OLDER_CONVERSATION_SUMMARY={summary}\n"
                "END_READ_ONLY_MEMORY_CONTEXT"
            )
        )
        result = chat_agent.invoke(
            cast(
                Any,
                {"messages": [context_message, *state["messages"]]},
            )
        )
        answer = next(item for item in reversed(result["messages"]) if isinstance(item, AIMessage))
        return {"messages": [answer]}

    def extract_memory(state: MemoryState) -> dict[str, object]:
        return {"profile": extract_profile(state["messages"], state.get("profile", {}))}

    def compact(state: MemoryState) -> dict[str, object]:
        messages = state["messages"]
        if len(messages) <= policy.max_messages:
            return {}
        discarded = messages[: -policy.max_messages]
        fragments = [f"{item.type}: {str(item.content)[:500]}" for item in discarded]
        summary = "\n".join(filter(None, [state.get("summary", ""), *fragments]))
        removals = [RemoveMessage(id=item.id) for item in discarded if item.id]
        return {"messages": removals, "summary": summary[-policy.max_summary_chars :]}

    def persist_memory(state: MemoryState, runtime: Runtime[MemoryContext]) -> dict[str, object]:
        if runtime.store is None:
            raise RuntimeError("Chat with Memories requires a LangGraph Store")
        runtime.store.put(("users", runtime.context.user_id), "profile", state.get("profile", {}))
        return {}

    builder = StateGraph(MemoryState, context_schema=MemoryContext)
    builder.add_node("load_memory", load_memory)
    builder.add_node("chat_agent", call_chat_agent)
    builder.add_node("extract_memory", extract_memory)
    builder.add_node("compact", compact)
    builder.add_node("persist_memory", persist_memory)
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "chat_agent")
    builder.add_edge("chat_agent", "extract_memory")
    builder.add_edge("extract_memory", "compact")
    builder.add_edge("compact", "persist_memory")
    builder.add_edge("persist_memory", END)
    return builder.compile(
        checkpointer=resolved_checkpointer,  # type: ignore[arg-type]
        store=resolved_store,  # type: ignore[arg-type]
    )


def run_turn(turn: ChatTurn, *, compiled_graph) -> TurnResult:
    config = {"configurable": {"thread_id": turn.thread_id}}
    if compiled_graph.checkpointer is not None:
        saved = compiled_graph.get_state(config)
        owner = saved.values.get("owner_user_id")
        if owner is not None and owner != turn.user_id:
            raise PermissionError("Conversation belongs to a different user")
    state = compiled_graph.invoke(
        {"messages": [HumanMessage(content=turn.message)]},
        config=config,
        context=MemoryContext(user_id=turn.user_id),
    )
    final = next(item for item in reversed(state["messages"]) if isinstance(item, AIMessage))
    return TurnResult(
        answer=str(final.content),
        profile=state.get("profile", {}),
        message_count=len(state["messages"]),
        summary=state.get("summary", ""),
    )


def delete_user_memory(store: BaseStore, user_id: str) -> None:
    store.delete(("users", user_id), "profile")


def delete_thread_memory(checkpointer: BaseCheckpointSaver[Any], thread_id: str) -> None:
    checkpointer.delete_thread(thread_id)
