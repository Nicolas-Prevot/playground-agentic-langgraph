"""Reopen approval checkpoints and demonstrate single-process effect deduplication."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from playground_agentic.scenarios.s08_hitl.graph import (
    ActionIntent,
    HitlState,
    IdempotentExecutor,
    PreparedAction,
    RefundPayload,
    SendEmailPayload,
    build_graph,
    result_from_state,
)


def _example_preparer(instruction: str) -> PreparedAction:
    if "refund" in instruction.lower():
        return PreparedAction(
            action="refund",
            payload=RefundPayload(order_id="order-42", amount=25),
            risk="high",
        )
    return PreparedAction(
        action="send_email",
        payload=SendEmailPayload(
            to="incident-review@example.test",
            subject="Draft checkout incident update",
        ),
        risk="medium",
    )


# Example 1: approve and replay an idempotent refund
def example_1_duplicate_refund() -> dict[str, object]:
    """Approve after restart and prove replay does not duplicate the refund."""

    executor = IdempotentExecutor()
    config: RunnableConfig = {
        "configurable": {"thread_id": "refund-approval-42", "actor_id": "finance-reviewer-7"}
    }
    intent = ActionIntent(
        instruction="Refund order 42 for 25 EUR after duplicate billing.",
        idempotency_key="refund-order-42",
    )
    with TemporaryDirectory(prefix="durable-hitl-") as temporary:
        database = Path(temporary) / "checkpoint.sqlite"
        with SqliteSaver.from_conn_string(str(database)) as saver:
            saver.setup()
            graph = build_graph(_example_preparer, executor=executor, checkpointer=saver)
            first = graph.invoke(cast(HitlState, intent.model_dump()), config=config)
            interrupted = bool(first.get("__interrupt__"))
        with SqliteSaver.from_conn_string(str(database)) as saver:
            saver.setup()
            resumed_graph = build_graph(_example_preparer, executor=executor, checkpointer=saver)
            completed = cast(
                HitlState,
                resumed_graph.invoke(
                    Command(resume={"decision": "approve", "reason": "Validated duplicate"}),
                    config=config,
                ),
            )
            replay = cast(HitlState, resumed_graph.invoke(None, config=config))
    return {
        "interrupted": interrupted,
        "result": result_from_state(completed).model_dump(),
        "replay_status": replay["status"],
        "executor_calls": executor.calls,
    }


# Example 2: edit a prepared action before execution
def example_2_edited_incident_email() -> dict[str, object]:
    """Edit after restart and execute only the reviewed email payload."""

    executor = IdempotentExecutor()
    config: RunnableConfig = {
        "configurable": {"thread_id": "email-approval-9", "actor_id": "incident-commander-3"}
    }
    intent = ActionIntent(
        instruction="Send the scheduled checkout incident status update.",
        idempotency_key="checkout-status-2026-08-29-1530",
    )
    with TemporaryDirectory(prefix="durable-hitl-") as temporary:
        database = Path(temporary) / "checkpoint.sqlite"
        with SqliteSaver.from_conn_string(str(database)) as saver:
            saver.setup()
            graph = build_graph(_example_preparer, executor=executor, checkpointer=saver)
            first = graph.invoke(cast(HitlState, intent.model_dump()), config=config)
            interrupted = bool(first.get("__interrupt__"))
        with SqliteSaver.from_conn_string(str(database)) as saver:
            saver.setup()
            resumed_graph = build_graph(_example_preparer, executor=executor, checkpointer=saver)
            completed = cast(
                HitlState,
                resumed_graph.invoke(
                    Command(
                        resume={
                            "decision": "edit",
                            "payload": {
                                "to": "status-page@example.test",
                                "subject": "Checkout latency mitigated - monitoring continues",
                            },
                            "reason": "Use the approved status-page wording and recipient.",
                        }
                    ),
                    config=config,
                ),
            )
    return {
        "interrupted": interrupted,
        "result": result_from_state(completed).model_dump(),
        "executor_calls": executor.calls,
    }


def run_example() -> dict[str, object]:
    """Run both independent examples for a compact command-line demonstration."""

    return {
        "duplicate_refund": example_1_duplicate_refund(),
        "edited_incident_email": example_2_edited_incident_email(),
    }


if __name__ == "__main__":
    print(run_example())
