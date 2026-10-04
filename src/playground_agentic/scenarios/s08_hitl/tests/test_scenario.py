from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from playground_agentic.scenarios.s08_hitl.example import (
    example_1_duplicate_refund,
    example_2_edited_incident_email,
)
from playground_agentic.scenarios.s08_hitl.graph import (
    IdempotentExecutor,
    PreparedAction,
    RefundPayload,
    SendEmailPayload,
    build_graph,
)


def _config(thread: str) -> dict[str, object]:
    return {"configurable": {"thread_id": thread, "actor_id": "reviewer"}}


def _test_preparer(instruction: str) -> PreparedAction:
    if "refund" in instruction.lower():
        return PreparedAction(
            action="refund",
            payload=RefundPayload(order_id="order-test", amount=25),
            risk="high",
        )
    return PreparedAction(
        action="send_email",
        payload=SendEmailPayload(to="draft@example.test", subject="Draft"),
        risk="medium",
    )


def test_example_1_resumes_refund_once_after_reopen() -> None:
    refund = example_1_duplicate_refund()
    assert refund["interrupted"] is True
    assert refund["result"]["status"] == "executed"
    assert refund["result"]["reviewer"] == "finance-reviewer-7"
    assert refund["replay_status"] == "executed"
    assert refund["executor_calls"] == 1


def test_example_2_executes_the_reviewed_email_payload() -> None:
    email = example_2_edited_incident_email()
    assert email["interrupted"] is True
    assert email["result"]["status"] == "executed"
    assert email["result"]["action"]["payload"] == {
        "to": "status-page@example.test",
        "subject": "Checkout latency mitigated - monitoring continues",
    }
    assert email["result"]["reviewer"] == "incident-commander-3"
    assert email["executor_calls"] == 1


def test_reject_never_executes_action() -> None:
    executor = IdempotentExecutor()
    graph = build_graph(_test_preparer, executor=executor, checkpointer=InMemorySaver())
    config = _config("reject")
    graph.invoke(
        {"instruction": "Refund the order", "idempotency_key": "refund-reject"}, config=config
    )
    state = graph.invoke(Command(resume={"decision": "reject", "reason": "No"}), config=config)
    assert state["status"] == "rejected"
    assert executor.calls == 0


def test_edit_changes_payload_before_execution() -> None:
    executor = IdempotentExecutor()
    graph = build_graph(_test_preparer, executor=executor, checkpointer=InMemorySaver())
    config = _config("edit")
    graph.invoke({"instruction": "Send the update", "idempotency_key": "email-edit"}, config=config)
    state = graph.invoke(
        Command(
            resume={
                "decision": "edit",
                "payload": {"to": "owner@example.test", "subject": "Edited"},
            }
        ),
        config=config,
    )
    assert state["action"]["payload"]["subject"] == "Edited"
    assert executor.calls == 1


def test_failed_verification_compensates() -> None:
    executor = IdempotentExecutor()
    graph = build_graph(
        _test_preparer,
        executor=executor,
        verifier=lambda _receipt: False,
        checkpointer=InMemorySaver(),
    )
    config = _config("compensate")
    graph.invoke(
        {"instruction": "Refund the order", "idempotency_key": "refund-comp"}, config=config
    )
    state = graph.invoke(Command(resume={"decision": "approve"}), config=config)
    assert state["status"] == "compensated"
    assert executor.calls == 1
    assert executor.compensations == 1
