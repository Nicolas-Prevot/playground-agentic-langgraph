"""Demonstrate crash recovery and mixed-outcome batch processing."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from playground_agentic.scenarios.s10_resumable.graph import (
    JobRequest,
    JobResult,
    JobState,
    WorkItem,
    WorkOutcome,
    build_graph,
    run,
)


# Example 1: process-crash recovery
def example_1_resumed_document_batch() -> dict[str, object]:
    """Resume after a process crash without repeating the successful sibling item."""

    calls: dict[str, int] = {}
    crash_once = {"enabled": True}

    def processor(item: WorkItem, _attempt: int) -> WorkOutcome:
        calls[item.id] = calls.get(item.id, 0) + 1
        if item.id == "unstable" and crash_once["enabled"]:
            crash_once["enabled"] = False
            raise RuntimeError("simulated process failure")
        return WorkOutcome(status="completed", output=item.payload.upper())

    request = JobRequest(
        items=[
            WorkItem(id="stable", payload="first document"),
            WorkItem(id="unstable", payload="second document"),
        ]
    )
    config: RunnableConfig = {"configurable": {"thread_id": "resumable-job-example"}}
    with TemporaryDirectory(prefix="resumable-job-") as temporary:
        database = Path(temporary) / "checkpoints.sqlite"
        with SqliteSaver.from_conn_string(str(database)) as saver:
            saver.setup()
            with suppress(RuntimeError):
                build_graph(processor, checkpointer=saver).invoke(
                    cast(JobState, request.model_dump()), config=config
                )
        with SqliteSaver.from_conn_string(str(database)) as saver:
            saver.setup()
            resumed = cast(
                JobState,
                build_graph(processor, checkpointer=saver).invoke(None, config=config),
            )

    result = JobResult.model_validate({"status": resumed["status"], "records": resumed["records"]})
    return {
        "result": result.model_dump(),
        "processor_calls": calls,
    }


# Example 2: retryable and permanent item failures
def example_2_mixed_import_batch() -> dict[str, object]:
    """Retry a transient item while isolating a permanently unreadable document."""

    result = run(
        JobRequest(
            items=[
                WorkItem(id="product_manual", payload="normalize product manual"),
                WorkItem(id="vendor_datasheet", payload="retry-once vendor datasheet"),
                WorkItem(id="corrupt_scan", payload="fail unreadable scan"),
            ],
            max_attempts=2,
        ),
        compiled_graph=build_graph(),
    )
    return result.model_dump()


def run_example() -> dict[str, object]:
    return {
        "resumed_document_batch": example_1_resumed_document_batch(),
        "mixed_import_batch": example_2_mixed_import_batch(),
    }


if __name__ == "__main__":
    print(run_example())
