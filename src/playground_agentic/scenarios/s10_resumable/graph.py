"""Scenario 10: durable fan-out work with retries and failure isolation."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field, model_validator


class WorkItem(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", max_length=80)
    payload: str = Field(min_length=1, max_length=10_000)


class JobRequest(BaseModel):
    items: list[WorkItem] = Field(min_length=1, max_length=100)
    max_attempts: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def unique_item_ids(self) -> JobRequest:
        identifiers = [item.id for item in self.items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Work item IDs must be unique")
        return self


class WorkOutcome(BaseModel):
    status: Literal["completed", "retryable", "failed"]
    output: str = ""
    error: str = ""


class AttemptRecord(BaseModel):
    item_id: str
    attempt: int
    status: Literal["completed", "retryable", "failed"]
    output: str = ""
    error: str = ""


class WorkRecord(BaseModel):
    item_id: str
    status: Literal["completed", "failed"]
    attempts: int
    output: str = ""
    error: str = ""


class JobResult(BaseModel):
    status: Literal["completed", "partial", "failed"]
    records: list[WorkRecord]


class JobState(TypedDict, total=False):
    items: list[dict[str, str]]
    max_attempts: int
    item: dict[str, str]
    attempt: int
    attempts: Annotated[list[dict[str, object]], operator.add]
    status: str
    records: list[dict[str, object]]


Processor = Callable[[WorkItem, int], WorkOutcome]


def offline_processor(item: WorkItem, attempt: int) -> WorkOutcome:
    """Exercise success, retry, and permanent failure without external services."""

    lowered = item.payload.lower()
    if "retry-once" in lowered and attempt == 1:
        return WorkOutcome(status="retryable", error="temporary dependency failure")
    if "fail" in lowered:
        return WorkOutcome(status="failed", error="item is not processable")
    return WorkOutcome(status="completed", output=item.payload.upper())


def _latest_attempts(state: JobState) -> dict[str, AttemptRecord]:
    latest: dict[str, AttemptRecord] = {}
    for raw in state.get("attempts", []):
        record = AttemptRecord.model_validate(raw)
        previous = latest.get(record.item_id)
        if previous is None or record.attempt >= previous.attempt:
            latest[record.item_id] = record
    return latest


def build_graph(
    processor: Processor = offline_processor,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    def initialize(state: JobState) -> dict[str, object]:
        request = JobRequest.model_validate(state)
        return {
            "items": [item.model_dump() for item in request.items],
            "max_attempts": request.max_attempts,
            "attempts": [],
        }

    def dispatch_initial(state: JobState) -> list[Send]:
        return [Send("worker", {"item": item, "attempt": 1}) for item in state["items"]]

    def worker(state: JobState) -> dict[str, object]:
        item = WorkItem.model_validate(state["item"])
        attempt = state["attempt"]
        outcome = processor(item, attempt)
        record = AttemptRecord(
            item_id=item.id,
            attempt=attempt,
            status=outcome.status,
            output=outcome.output,
            error=outcome.error,
        )
        return {"attempts": [record.model_dump()]}

    def assess(state: JobState) -> dict[str, object]:
        # This synchronization node intentionally writes nothing. It exists so the
        # retry route sees all parallel worker writes from the completed superstep.
        del state
        return {}

    def retry_or_finalize(state: JobState) -> str | list[Send]:
        latest = _latest_attempts(state)
        items = {item["id"]: item for item in state["items"]}
        retries = [
            Send(
                "worker",
                {"item": items[item_id], "attempt": record.attempt + 1},
            )
            for item_id, record in latest.items()
            if record.status == "retryable" and record.attempt < state["max_attempts"]
        ]
        return retries or "finalize"

    def finalize(state: JobState) -> dict[str, object]:
        latest = _latest_attempts(state)
        records: list[WorkRecord] = []
        for raw_item in state["items"]:
            item = WorkItem.model_validate(raw_item)
            attempt = latest[item.id]
            completed = attempt.status == "completed"
            records.append(
                WorkRecord(
                    item_id=item.id,
                    status="completed" if completed else "failed",
                    attempts=attempt.attempt,
                    output=attempt.output,
                    error=attempt.error,
                )
            )
        completed_count = sum(record.status == "completed" for record in records)
        if completed_count == len(records):
            status = "completed"
        elif completed_count:
            status = "partial"
        else:
            status = "failed"
        return {"status": status, "records": [record.model_dump() for record in records]}

    builder = StateGraph(JobState)
    builder.add_node("initialize", initialize)
    builder.add_node("worker", worker)
    builder.add_node("assess", assess)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", dispatch_initial, ["worker"])
    builder.add_edge("worker", "assess")
    builder.add_conditional_edges("assess", retry_or_finalize, ["worker", "finalize"])
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


def run(request: JobRequest, *, compiled_graph, config: RunnableConfig | None = None) -> JobResult:
    state = compiled_graph.invoke(request.model_dump(), config=config)
    return JobResult.model_validate({"status": state["status"], "records": state["records"]})
