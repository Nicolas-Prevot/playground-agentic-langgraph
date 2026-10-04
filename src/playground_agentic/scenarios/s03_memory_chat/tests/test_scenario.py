from typing import Any, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from playground_agentic.scenarios.s03_memory_chat.example import (
    example_1_developer_preferences,
    example_2_support_contact_preferences,
    example_model,
)
from playground_agentic.scenarios.s03_memory_chat.graph import (
    ChatTurn,
    MemoryPolicy,
    build_graph,
    delete_thread_memory,
    delete_user_memory,
    run_turn,
)


def test_example_1_recalls_developer_preferences_across_threads() -> None:
    developer = example_1_developer_preferences()
    developer_recall = cast(dict[str, Any], developer["recalled_in_new_thread"])
    assert developer_recall["answer"] == "Use Python examples and keep answers concise."
    assert developer_recall["profile"] == {
        "preferred_language": "Python",
        "response_style": "concise",
    }
    assert developer["resumed_after_reopen"] is True


def test_example_2_recalls_support_preferences_across_threads() -> None:
    support = example_2_support_contact_preferences()
    support_recall = cast(dict[str, Any], support["recalled_in_new_thread"])
    assert support_recall["answer"] == "Send incident updates by email using Europe/Paris time."
    assert support_recall["profile"] == {
        "timezone": "Europe/Paris",
        "update_channel": "email",
    }
    assert support["resumed_after_reopen"] is True


def test_store_isolation_and_independent_deletion() -> None:
    store = InMemoryStore()
    saver = InMemorySaver()
    chat = build_graph(example_model(), checkpointer=saver, store=store)
    run_turn(ChatTurn(user_id="a", thread_id="a-1", message="My name is Lin"), compiled_graph=chat)
    unknown = run_turn(
        ChatTurn(user_id="b", thread_id="b-1", message="Who am I?"), compiled_graph=chat
    )
    assert unknown.answer.endswith("unknown.")
    delete_thread_memory(saver, "a-1")
    remembered = run_turn(
        ChatTurn(user_id="a", thread_id="a-1", message="Who am I?"), compiled_graph=chat
    )
    assert remembered.answer.endswith("Lin.")
    delete_user_memory(store, "a")
    forgotten = run_turn(
        ChatTurn(user_id="a", thread_id="a-2", message="Who am I?"), compiled_graph=chat
    )
    assert forgotten.answer.endswith("unknown.")


def test_compaction_bounds_checkpoint_messages() -> None:
    chat = build_graph(
        example_model(),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        policy=MemoryPolicy(max_messages=4, max_summary_chars=500),
    )
    result = None
    for index in range(5):
        result = run_turn(
            ChatTurn(user_id="compact", thread_id="compact", message=f"Turn {index}"),
            compiled_graph=chat,
        )
    assert result is not None
    assert result.message_count == 4
    assert result.summary
