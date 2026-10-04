"""Regression tests for copyable contracts, budgets, and replay behavior."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from playground_agentic.scenarios.s01_structured.graph import build_graph as structured_graph
from playground_agentic.scenarios.s02_tools.example import lookup_shipment
from playground_agentic.scenarios.s02_tools.graph import ChatRequest, ToolPolicy
from playground_agentic.scenarios.s02_tools.graph import build_graph as tool_graph
from playground_agentic.scenarios.s02_tools.graph import run as run_tools
from playground_agentic.scenarios.s05_orchestrator.graph import Plan, PlannedTask
from playground_agentic.scenarios.s06_rag.graph import InMemoryKnowledgeClient, RagRequest
from playground_agentic.scenarios.s06_rag.graph import build_graph as rag_graph
from playground_agentic.scenarios.s06_rag.graph import run as run_rag
from playground_agentic.scenarios.s08_hitl.dummy_executor import SqliteDummyExecutor
from playground_agentic.scenarios.s08_hitl.graph import (
    IdempotentExecutor,
    PreparedAction,
    RefundPayload,
    SendEmailPayload,
)
from playground_agentic.scenarios.s08_hitl.graph import build_graph as approval_graph
from playground_agentic.testing import ScriptedChatModel


def refund(amount: float = 25) -> PreparedAction:
    return PreparedAction(
        action="refund", payload=RefundPayload(order_id="order-42", amount=amount), risk="high"
    )


def test_whitespace_intake_does_not_call_analyzer() -> None:
    def unexpected(_text):
        raise AssertionError("Invalid input must not reach a model")

    assert structured_graph(unexpected).invoke({"text": " \n\t "})["status"] == "invalid"


def calling_model(count=1) -> ScriptedChatModel:
    return ScriptedChatModel(
        responder=lambda _messages, _tools: AIMessage(
            content="",
            tool_calls=[
                {"name": "lookup_shipment", "args": {"tracking_id": "TRK-2048"}, "id": f"call-{i}"}
                for i in range(count)
            ],
        )
    )


def test_parallel_tool_calls_cannot_bypass_budget_and_all_get_responses() -> None:
    graph = tool_graph(
        calling_model(3),
        tools=[lookup_shipment],
        policy=ToolPolicy(
            allowed_tools=frozenset({"lookup_shipment"}),
            max_tool_calls=2,
        ),
    )
    state = graph.invoke({"messages": [HumanMessage(content="lookup")], "steps": 0})
    assert state["status"] == "budget_exhausted"
    assert state.get("tool_calls", 0) == 0
    responses = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    assert {message.tool_call_id for message in responses} == {"call-0", "call-1", "call-2"}
    assert all(message.status == "error" for message in responses)


def test_large_valid_step_budget_reaches_domain_terminal_not_recursion_error() -> None:
    result = run_tools(
        ChatRequest(message="lookup", max_steps=20),
        compiled_graph=tool_graph(
            calling_model(),
            tools=[lookup_shipment],
            policy=ToolPolicy(
                allowed_tools=frozenset({"lookup_shipment"}),
                max_steps=20,
                max_tool_calls=30,
            ),
        ),
    )
    assert result.status == "budget_exhausted"
    assert result.steps == 20


def test_plan_ids_are_unique() -> None:
    task = PlannedTask(id="migration", objective="Draft migration plan")
    with pytest.raises(ValidationError, match="unique"):
        Plan(tasks=[task, task])


@pytest.mark.asyncio
async def test_empty_corpus_cannot_answer_from_hidden_fixtures() -> None:
    result = await run_rag(
        RagRequest(question="What is LangGraph persistence?"),
        compiled_graph=rag_graph(InMemoryKnowledgeClient([])),
    )
    assert result.status == "not_found"
    assert result.citations == []


@pytest.mark.asyncio
async def test_domain_rewriter_receives_previous_query_and_budget() -> None:
    calls = []

    def rewrite(question, previous, attempt):
        calls.append((question, previous, attempt))
        return "invoice retention"

    result = await run_rag(
        RagRequest(question="archive", max_rewrites=2),
        compiled_graph=rag_graph(InMemoryKnowledgeClient([]), rewriter=rewrite),
    )
    assert result.rewrites == 2
    assert calls == [("archive", "archive", 1), ("archive", "invoice retention", 2)]


def test_action_payload_must_match_action_kind() -> None:
    with pytest.raises(ValidationError, match="match"):
        PreparedAction(
            action="refund", payload=SendEmailPayload(to="a@example.test", subject="X"), risk="high"
        )


def test_reviewed_edit_is_rechecked_against_policy() -> None:
    executor = IdempotentExecutor()
    graph = approval_graph(
        lambda _: refund(),
        executor=executor,
        checkpointer=InMemorySaver(),
        action_policy=lambda action: action.payload.amount <= 25,
    )
    config = {"configurable": {"thread_id": "edit-policy", "actor_id": "fixture-reviewer"}}
    graph.invoke({"instruction": "refund", "idempotency_key": "refund-42"}, config=config)
    state = graph.invoke(
        Command(
            resume={
                "decision": "edit",
                "payload": {
                    "order_id": "order-42",
                    "amount": 100,
                },
            }
        ),
        config=config,
    )
    assert state["status"] == "rejected"
    assert executor.calls == 0
    assert state["audit"][-1]["event"] == "policy_denied_after_review"


def test_uncertain_execution_is_not_reported_as_compensated() -> None:
    class DisconnectedExecutor(IdempotentExecutor):
        def execute(self, key, action):
            raise TimeoutError("Response lost; effect is unknown")

    executor = DisconnectedExecutor()
    graph = approval_graph(lambda _: refund(), executor=executor, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "uncertain", "actor_id": "fixture-reviewer"}}
    graph.invoke({"instruction": "refund", "idempotency_key": "refund-42"}, config=config)
    state = graph.invoke(Command(resume={"decision": "approve"}), config=config)
    assert state["status"] == "error"
    assert executor.compensations == 0
    assert state["audit"][-1]["event"] == "execution_requires_reconciliation"


@pytest.mark.parametrize("durable", [False, True])
def test_effect_keys_bind_payloads_and_compensation_keeps_tombstone(
    tmp_path: Path, durable
) -> None:
    executor = SqliteDummyExecutor(tmp_path / "effects.sqlite") if durable else IdempotentExecutor()
    receipt = executor.execute("key-42", refund())
    if durable:
        executor = SqliteDummyExecutor(tmp_path / "effects.sqlite")
    assert executor.execute("key-42", refund()) == receipt
    with pytest.raises(ValueError):
        executor.execute("key-42", refund(50))
    executor.compensate("key-42")
    executor.compensate("key-42")
    with pytest.raises(ValueError):
        executor.execute("key-42", refund())


def test_approval_without_checkpointer_fails_at_build() -> None:
    with pytest.raises(ValueError, match="checkpointer"):
        approval_graph(lambda _: refund())


def test_memory_owner_mismatch_does_not_append_to_checkpoint() -> None:
    from playground_agentic.scenarios.s03_memory_chat.graph import ChatTurn, build_graph, run_turn

    model = ScriptedChatModel(responder=lambda _m, _t: AIMessage(content="Welcome"))
    graph = build_graph(model)
    run_turn(
        ChatTurn(user_id="alice", thread_id="shared", message="My name is Alice"),
        compiled_graph=graph,
    )
    config = {"configurable": {"thread_id": "shared"}}
    before = graph.get_state(config)
    with pytest.raises(PermissionError):
        run_turn(
            ChatTurn(user_id="bob", thread_id="shared", message="Overwrite Alice"),
            compiled_graph=graph,
        )
    after = graph.get_state(config)
    assert after.config == before.config
    assert after.values == before.values
    assert model.counter.value == 1


def test_deep_worker_blocks_second_delegation() -> None:
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError

    from playground_agentic.scenarios.s09_deep_worker.graph import DeepTaskRequest, build_graph, run

    parent = ScriptedChatModel(
        responder=lambda _m, _t: AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {
                        "description": "Review supplied evidence",
                        "subagent_type": "researcher",
                    },
                    "id": "delegate",
                }
            ],
        )
    )
    child = ScriptedChatModel(responder=lambda _m, _t: AIMessage(content="Evidence reviewed"))
    with pytest.raises(ToolCallLimitExceededError):
        run(
            DeepTaskRequest(task="Keep delegating"),
            compiled_graph=build_graph(parent, researcher_model=child),
        )
    assert child.counter.value == 1


def test_deep_worker_denies_write_outside_artifact() -> None:
    from playground_agentic.scenarios.s09_deep_worker.graph import DeepTaskRequest, build_graph, run

    observed = []

    def respond(messages, _tools):
        if isinstance(messages[-1], ToolMessage):
            observed.append(str(messages[-1].content))
            return AIMessage(content="Stopped")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/outside.txt", "content": "forbidden"},
                    "id": "outside",
                }
            ],
        )

    parent = ScriptedChatModel(responder=respond)
    child = ScriptedChatModel(responder=lambda _m, _t: AIMessage(content="Unused"))
    result = run(
        DeepTaskRequest(task="Write outside the permitted path"),
        compiled_graph=build_graph(parent, researcher_model=child),
    )
    assert result.status == "invalid_artifact"
    assert result.artifact == ""
    assert observed and any("denied" in message.lower() for message in observed)
    assert child.counter.value == 0
