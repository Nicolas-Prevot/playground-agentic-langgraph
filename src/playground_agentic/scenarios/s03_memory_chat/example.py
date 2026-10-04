"""Run two useful memory cases across a SQLite close/reopen boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from playground_agentic.scenarios.s03_memory_chat.graph import (
    ChatTurn,
    build_graph,
    run_turn,
)
from playground_agentic.testing import ScriptedChatModel


def _chat_responder(messages: list[BaseMessage], _tools: object) -> AIMessage:
    profile: dict[str, str] = {}
    for message in messages:
        content = str(message.content)
        if isinstance(message, HumanMessage) and content.startswith("READ_ONLY_MEMORY_CONTEXT"):
            profile_line = next(
                line for line in content.splitlines() if line.startswith("PROFILE=")
            )
            profile = json.loads(profile_line.removeprefix("PROFILE="))
    human = next(message for message in reversed(messages) if isinstance(message, HumanMessage))
    text = str(human.content)
    match = re.search(r"\bmy name is\s+([\w-]+)", text, re.IGNORECASE)
    if match:
        return AIMessage(content=f"Nice to meet you, {match.group(1)}. I will remember that.")
    if "language" in text.lower() or "answer style" in text.lower():
        language = profile.get("preferred_language", "unspecified")
        style = profile.get("response_style", "unspecified")
        return AIMessage(content=f"Use {language} examples and keep answers {style}.")
    if "incident updates" in text.lower():
        channel = profile.get("update_channel", "unspecified")
        timezone = profile.get("timezone", "unspecified")
        return AIMessage(content=f"Send incident updates by {channel} using {timezone} time.")
    if "prefer" in text.lower() or "timezone" in text.lower():
        return AIMessage(content="I will keep those preferences for future conversations.")
    if "name" in text.lower() or "who am i" in text.lower():
        return AIMessage(content=f"Your remembered name is {profile.get('name', 'unknown')}.")
    return AIMessage(content=f"Hello {profile.get('name', 'there')}. How can I help?")


def example_model() -> ScriptedChatModel:
    return ScriptedChatModel(responder=_chat_responder)


def _persist_and_recall(
    *,
    user_id: str,
    first_thread: str,
    first_message: str,
    second_thread: str,
    second_message: str,
) -> dict[str, object]:
    model = example_model()
    with TemporaryDirectory(prefix="memory-chat-") as temporary:
        root = Path(temporary)
        checkpoint_path = root / "checkpoints.sqlite"
        store_path = root / "store.sqlite"
        with (
            SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer,
            SqliteStore.from_conn_string(str(store_path)) as store,
        ):
            checkpointer.setup()
            store.setup()
            chat = build_graph(model, checkpointer=checkpointer, store=store)
            saved = run_turn(
                ChatTurn(
                    user_id=user_id,
                    thread_id=first_thread,
                    message=first_message,
                ),
                compiled_graph=chat,
            )
        with (
            SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer,
            SqliteStore.from_conn_string(str(store_path)) as store,
        ):
            checkpointer.setup()
            store.setup()
            reopened = build_graph(model, checkpointer=checkpointer, store=store)
            recalled = run_turn(
                ChatTurn(
                    user_id=user_id,
                    thread_id=second_thread,
                    message=second_message,
                ),
                compiled_graph=reopened,
            )
    return {
        "saved": saved.model_dump(),
        "recalled_in_new_thread": recalled.model_dump(),
        "resumed_after_reopen": True,
    }


# Example 1: developer preferences across project threads
def example_1_developer_preferences() -> dict[str, object]:
    """Recall coding and response preferences in a different project thread."""

    return _persist_and_recall(
        user_id="developer-7",
        first_thread="project-a",
        first_message="I prefer Python examples and keep your answers concise.",
        second_thread="project-b",
        second_message="Which language and answer style should you use for me?",
    )


# Example 2: support preferences across incident threads
def example_2_support_contact_preferences() -> dict[str, object]:
    """Recall a customer's incident channel and timezone in a new support thread."""

    return _persist_and_recall(
        user_id="customer-9",
        first_thread="incident-a",
        first_message="I prefer email updates and my timezone is Europe/Paris.",
        second_thread="incident-b",
        second_message="How should you send incident updates, and in which timezone?",
    )


def run_example() -> dict[str, object]:
    return {
        "developer_preferences": example_1_developer_preferences(),
        "support_contact_preferences": example_2_support_contact_preferences(),
    }


if __name__ == "__main__":
    print(run_example())
