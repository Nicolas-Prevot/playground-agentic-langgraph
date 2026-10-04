"""Prepare in one process; inspect and approve/reject a simulated refund in another."""

import argparse
from pathlib import Path
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from playground_agentic.example_support import model_from_env, show, structured
from playground_agentic.scenarios.s08_hitl.dummy_executor import SqliteDummyExecutor
from playground_agentic.scenarios.s08_hitl.graph import (
    HitlState,
    PreparedAction,
    RefundPayload,
    build_graph,
    result_from_state,
)

CONFIG: RunnableConfig = {"configurable": {"thread_id": "demo-refund", "actor_id": "demo-reviewer"}}


def run_example(
    model: BaseChatModel | None, directory: Path, *, decision: str | None = None
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    executor = SqliteDummyExecutor(directory / "effects.sqlite")

    def prepare(instruction: str) -> PreparedAction:
        if model is None:
            raise RuntimeError("Resume must use the saved proposal, never regenerate it")
        payload = structured(
            model,
            RefundPayload,
            "Extract the order ID and refund amount. "
            "Amounts are EUR; do not invent missing information.",
            {"instruction": instruction},
        )
        return PreparedAction(action="refund", payload=payload, risk="high")

    def policy(action: PreparedAction) -> bool:
        # Fixture service says order-42 was overcharged by exactly EUR 25.
        return (
            isinstance(action.payload, RefundPayload)
            and action.payload.order_id == "order-42"
            and action.payload.amount <= 25
        )

    with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        saver.setup()
        graph = build_graph(prepare, executor=executor, checkpointer=saver, action_policy=policy)
        snapshot = graph.get_state(CONFIG)
        if decision is None:
            if snapshot.values:
                raise ValueError("This demo already exists. Resume it or choose a new --directory.")
            state = graph.invoke(
                {
                    "instruction": "Refund EUR 25 to order-42 for a confirmed duplicate charge.",
                    "idempotency_key": "duplicate-order-42",
                },
                config=CONFIG,
                durability="sync",
            )
            return {
                "status": state["status"],
                "proposal": state["action"],
                "interrupted": bool(state.get("__interrupt__")),
                "effects": executor.count(),
            }
        if decision not in {"approve", "reject"}:
            raise ValueError("Demo decision must be approve or reject")
        if not snapshot.next or not any(task.interrupts for task in snapshot.tasks):
            raise ValueError("No pending approval. Run prepare first with this directory.")
        state = graph.invoke(
            Command(resume={"decision": decision}), config=CONFIG, durability="sync"
        )
        return {
            "result": result_from_state(cast(HitlState, state)).model_dump(),
            "effects": executor.count(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "resume"])
    parser.add_argument("--directory", type=Path, default=Path(".demo-state/approval"))
    parser.add_argument("--decision", choices=["approve", "reject"], default="reject")
    args = parser.parse_args()
    show(
        run_example(
            model_from_env() if args.phase == "prepare" else None,
            args.directory,
            decision=args.decision if args.phase == "resume" else None,
        )
    )
