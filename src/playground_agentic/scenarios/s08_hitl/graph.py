"""Scenario 08: durable approve/edit/reject workflow with idempotent effects."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionIntent(BaseModel):
    instruction: str = Field(min_length=1, max_length=2_000)
    idempotency_key: str = Field(min_length=4, max_length=128)


class SendEmailPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=1, max_length=200)


class RefundPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0, le=10_000)


class PreparedAction(BaseModel):
    action: Literal["send_email", "refund"]
    payload: SendEmailPayload | RefundPayload
    risk: Literal["medium", "high"]

    @model_validator(mode="after")
    def matching_payload(self) -> PreparedAction:
        expected = SendEmailPayload if self.action == "send_email" else RefundPayload
        if not isinstance(self.payload, expected):
            raise ValueError("Payload must match the selected action")
        return self


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    payload: dict[str, Any] | None = None
    reason: str = Field(default="", max_length=500)


class ActionResult(BaseModel):
    status: Literal["executed", "rejected", "compensated", "error"]
    action: PreparedAction
    receipt: str | None = None
    reviewer: str | None = None
    audit: list[dict[str, str]]


class HitlState(TypedDict, total=False):
    instruction: str
    idempotency_key: str
    action: dict[str, Any]
    decision: dict[str, Any]
    audit: Annotated[list[dict[str, str]], operator.add]
    receipt: str
    execution_error: str
    verified: bool
    reviewer: str
    status: str


Preparer = Callable[[str], PreparedAction]
Verifier = Callable[[str], bool]
ActionPolicy = Callable[[PreparedAction], bool]


class Executor(Protocol):
    def execute(self, key: str, action: PreparedAction) -> str: ...

    def compensate(self, key: str) -> None: ...


class IdempotentExecutor:
    """Single-process test double; use durable atomic records for real effects."""

    def __init__(self) -> None:
        self.receipts: dict[str, str] = {}
        self.actions: dict[str, str] = {}
        self.compensated: set[str] = set()
        self.calls = 0
        self.compensations = 0

    def execute(self, key: str, action: PreparedAction) -> str:
        if key in self.receipts:
            if self.actions[key] != action.model_dump_json():
                raise ValueError("Idempotency key reused for a different action")
            if key in self.compensated:
                raise ValueError("Compensated actions require a new idempotency key")
            return self.receipts[key]
        self.calls += 1
        receipt = f"{action.action}:{key}:ok"
        self.receipts[key] = receipt
        self.actions[key] = action.model_dump_json()
        return receipt

    def compensate(self, key: str) -> None:
        if key in self.receipts and key not in self.compensated:
            self.compensations += 1
            self.compensated.add(key)


def build_graph(
    preparer: Preparer,
    *,
    executor: Executor | None = None,
    verifier: Verifier = lambda _receipt: True,
    action_policy: ActionPolicy = lambda _action: True,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    if checkpointer is None:
        raise ValueError("Approval requires a checkpointer and a thread_id at invocation")
    executor = executor or IdempotentExecutor()

    def prepare(state: HitlState) -> dict[str, object]:
        intent = ActionIntent.model_validate(state)
        action = preparer(intent.instruction)
        return {"action": action.model_dump(), "audit": [], "status": "prepared"}

    def policy(state: HitlState) -> dict[str, object]:
        action = PreparedAction.model_validate(state["action"])
        if not action_policy(action):
            return {"status": "rejected", "audit": [{"event": "policy_denied"}]}
        return {"status": "awaiting_approval"}

    def route_policy(state: HitlState) -> Literal["review", "rejected"]:
        return "rejected" if state["status"] == "rejected" else "review"

    def review(state: HitlState, config: RunnableConfig) -> dict[str, object]:
        actor = str(config.get("configurable", {}).get("actor_id", ""))
        if not actor:
            raise PermissionError("Authenticated actor_id is required to resume approval")
        raw = interrupt(
            {
                "action": state["action"],
                "allowed_decisions": ["approve", "edit", "reject"],
            }
        )
        decision = ApprovalDecision.model_validate(raw)
        action = PreparedAction.model_validate(state["action"])
        if decision.decision == "edit":
            action = PreparedAction.model_validate(
                {**action.model_dump(), "payload": decision.payload or {}}
            )
        if decision.decision != "reject" and not action_policy(action):
            return {
                "status": "rejected",
                "reviewer": actor,
                "audit": [{"event": "policy_denied_after_review", "actor": actor}],
            }
        return {
            "action": action.model_dump(),
            "decision": decision.model_dump(),
            "reviewer": actor,
            "status": "rejected" if decision.decision == "reject" else "approved",
            "audit": [{"event": decision.decision, "actor": actor}],
        }

    def route_review(state: HitlState) -> Literal["execute", "rejected"]:
        return "rejected" if state["status"] == "rejected" else "execute"

    def execute(state: HitlState) -> dict[str, object]:
        action = PreparedAction.model_validate(state["action"])
        try:
            receipt = executor.execute(state["idempotency_key"], action)
        except Exception as exc:
            return {"execution_error": type(exc).__name__, "status": "error"}
        return {"receipt": receipt, "status": "executed"}

    def route_execute(state: HitlState) -> Literal["verify", "error"]:
        return "error" if state.get("execution_error") else "verify"

    def error(state: HitlState) -> dict[str, object]:
        # A timeout may follow a committed external effect. Reconcile at the executor boundary;
        # do not claim that an unknown effect has been successfully compensated.
        del state
        return {"status": "error", "audit": [{"event": "execution_requires_reconciliation"}]}

    def verify(state: HitlState) -> dict[str, object]:
        verified = verifier(state["receipt"])
        return {"verified": verified}

    def route_verify(state: HitlState) -> Literal["finish", "compensate"]:
        return "finish" if state["verified"] else "compensate"

    def compensate(state: HitlState) -> dict[str, object]:
        executor.compensate(state["idempotency_key"])
        return {"status": "compensated", "audit": [{"event": "compensated"}]}

    def finish(state: HitlState) -> dict[str, object]:
        del state
        return {"status": "executed", "audit": [{"event": "verified"}]}

    def rejected(state: HitlState) -> dict[str, object]:
        del state
        return {"status": "rejected"}

    builder = StateGraph(HitlState)
    builder.add_node("prepare", prepare)
    builder.add_node("policy", policy)
    builder.add_node("review", review)
    builder.add_node("execute", execute)
    builder.add_node("error", error)
    builder.add_node("verify", verify)
    builder.add_node("compensate", compensate)
    builder.add_node("finish", finish)
    builder.add_node("rejected", rejected)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "policy")
    builder.add_conditional_edges("policy", route_policy)
    builder.add_conditional_edges("review", route_review)
    builder.add_conditional_edges("execute", route_execute)
    builder.add_conditional_edges("verify", route_verify)
    builder.add_edge("compensate", END)
    builder.add_edge("finish", END)
    builder.add_edge("rejected", END)
    builder.add_edge("error", END)
    return builder.compile(checkpointer=checkpointer)


def result_from_state(state: HitlState) -> ActionResult:
    return ActionResult.model_validate(
        {
            "status": state["status"],
            "action": state["action"],
            "receipt": state.get("receipt"),
            "reviewer": state.get("reviewer"),
            "audit": state.get("audit", []),
        }
    )
