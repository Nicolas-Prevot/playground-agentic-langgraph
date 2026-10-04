"""Summarize a batch, simulate a worker crash, then resume in a fresh process."""

import argparse
from pathlib import Path
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from playground_agentic.example_support import generate_text, model_from_env, show
from playground_agentic.scenarios.s10_resumable.graph import (
    JobRequest,
    JobResult,
    JobState,
    WorkItem,
    WorkOutcome,
    build_graph,
)

CONFIG: RunnableConfig = {
    "configurable": {"thread_id": "demo-document-batch"},
    "max_concurrency": 1,
}


class SimulatedWorkerCrash(RuntimeError):
    """An intentional fixture failure, distinct from real provider failures."""


def run_example(
    model: BaseChatModel, directory: Path, *, resume: bool = False
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

    def process(item: WorkItem, attempt: int) -> WorkOutcome:
        del attempt
        if item.id == "runbook" and not resume:
            raise SimulatedWorkerCrash("Document service disconnected after the first item")
        calls.append(item.id)
        summary = generate_text(
            model,
            "Summarize this internal document in one sentence. "
            "Use only the supplied facts; no external lookup.",
            {"document_id": item.id, "text": item.payload},
        )
        return WorkOutcome(status="completed", output=summary)

    with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        saver.setup()
        graph = build_graph(process, checkpointer=saver)
        snapshot = graph.get_state(CONFIG)
        if resume:
            if not snapshot.next:
                raise ValueError("No pending job. Run start first with this directory.")
            state = graph.invoke(None, config=CONFIG, durability="sync")
            result = JobResult.model_validate(
                {"status": state["status"], "records": state["records"]}
            )
            return {"result": result.model_dump(), "model_calls_this_process": calls}
        if snapshot.values:
            raise ValueError("This job already exists. Resume it or choose a new --directory.")
        request = JobRequest(
            items=[
                WorkItem(
                    id="policy",
                    payload="Audit logs are retained for 365 days; legal holds "
                    "suspend deletion until compliance releases the hold.",
                ),
                WorkItem(
                    id="runbook",
                    payload="When checkout p95 latency exceeds 2 seconds for "
                    "5 minutes, the on-call checks provider errors and queue saturation.",
                ),
            ]
        )
        try:
            graph.invoke(cast(JobState, request.model_dump()), config=CONFIG, durability="sync")
        except SimulatedWorkerCrash:
            return {
                "status": "simulated_crash",
                "model_calls_this_process": calls,
                "next_step": "Run this module with resume and the same --directory.",
            }
        raise RuntimeError("The start phase should have encountered the simulated crash")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["start", "resume"])
    parser.add_argument("--directory", type=Path, default=Path(".demo-state/batch"))
    args = parser.parse_args()
    show(run_example(model_from_env(), args.directory, resume=args.phase == "resume"))
