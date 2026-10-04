"""Observe a document batch: mixed outcomes, then a separate-process crash/resume.

Run with --help. The fixture model includes explicitly synthetic token usage; --live
uses the configured endpoint. Only this optional entry point initializes the OTel SDK.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import sleep
from typing import Literal
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from opentelemetry import trace
from opentelemetry.trace import SpanContext
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from playground_agentic.example_support import model_from_env, show
from playground_agentic.testing import ScriptedChatModel

from .graph import JobRequest, JobResult, WorkItem, WorkOutcome, build_graph
from .otel_sdk import configured_telemetry, langchain_callback
from .telemetry import JobTelemetry, current_trace_id

Phase = Literal["mixed", "start", "resume"]
THREAD = "observability-document-batch"


class SimulatedWorkerCrash(RuntimeError):
    """An intentional interrupted batch, not a provider failure."""


def fixture_model() -> ScriptedChatModel:
    def respond(messages, _tools) -> AIMessage:
        text = messages[-1].text
        if "audit" in text:
            summary = "Retain audit records for 90 days and review access monthly."
        elif "latency" in text:
            summary = "Check the latency dashboard before rolling back a degraded release."
        else:
            summary = "Review the supplier's integration guide before importing its catalog."
        return AIMessage(
            content=summary,
            # Fixed demonstration counts, not a tokenizer or a cost estimate.
            usage_metadata={"input_tokens": 40, "output_tokens": 16, "total_tokens": 56},
        )

    return ScriptedChatModel(responder=respond, model_name="fixture-summary")


def previous_context(path: Path) -> SpanContext | None:
    """Only W3C traceparent is persisted. No baggage, prompts, or SDK objects."""
    try:
        carrier = {"traceparent": path.read_text(encoding="ascii").strip()}
    except (OSError, UnicodeError):
        return None
    context = TraceContextTextMapPropagator().extract(carrier)
    span = trace.get_current_span(context).get_span_context()
    return span if span.is_valid else None


def save_context(path: Path, telemetry: JobTelemetry) -> None:
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    try:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(carrier["traceparent"], encoding="ascii")
        temporary.replace(path)
    except OSError:
        # Losing correlation must not prevent a checkpointed job from running.
        telemetry.logger.warning("job.trace_link_not_saved")


def run_example(
    telemetry: JobTelemetry,
    *,
    phase: Phase = "mixed",
    directory: Path = Path(".demo-state/otel-job"),
    model: BaseChatModel | None = None,
    provider: str = "fixture",
    stream: bool = False,
) -> dict[str, object]:
    selected_model = model if model is not None else fixture_model()
    # Model metadata takes precedence over LangChain's adapter-derived provider
    # ("litellm"). Per-invocation metadata does not. Preserve the caller's model.
    selected_model = selected_model.model_copy(
        update={"metadata": {**(selected_model.metadata or {}), "ls_provider": provider}}
    )
    executed: list[str] = []

    def process(item: WorkItem, attempt: int) -> WorkOutcome:
        executed.append(item.id)
        if item.id == "supplier" and attempt == 1:
            return WorkOutcome(status="retryable", error="simulated document gateway timeout")
        if item.id == "scan":
            return WorkOutcome(status="failed", error="scan has no readable text")
        if item.id == "runbook" and phase == "start":
            raise SimulatedWorkerCrash("simulated worker shutdown before model call")
        messages = [
            SystemMessage(content="Summarize this document in one actionable sentence."),
            HumanMessage(content=item.payload),
        ]
        # The inherited official callback observes invocation parameters and usage.
        if stream:
            response = None
            for chunk in selected_model.stream(messages):
                if not isinstance(chunk, AIMessageChunk):
                    raise ValueError(
                        "The model fell back to a non-streaming response. Configure a streaming "
                        "adapter (for LiteLLM, streaming=True) or omit --stream."
                    )
                response = chunk if response is None else response + chunk
            if response is None:
                raise ValueError("The model returned no chunks")
        else:
            response = selected_model.invoke(messages)
        if not response.text.strip():
            raise ValueError("The model returned no text")
        return WorkOutcome(status="completed", output=response.text)

    items = [WorkItem(id="policy", payload="Keep audit records 90 days. Review access monthly.")]
    if phase == "mixed":
        items.extend(
            [
                WorkItem(
                    id="supplier", payload="Review the supplier integration guide before import."
                ),
                WorkItem(id="scan", payload="[unreadable scanned document]"),
            ]
        )
    else:
        items.append(
            WorkItem(
                id="runbook", payload="For elevated latency, check the dashboard before rollback."
            )
        )
    request = JobRequest(items=items)
    process_observed = telemetry.processor(process)
    trace_id = ""
    job_id = str(uuid4()) if phase == "mixed" else THREAD

    def invoke(graph, config: RunnableConfig, link_file: Path | None = None) -> JobResult:
        nonlocal trace_id
        previous = previous_context(link_file) if phase == "resume" and link_file else None
        with telemetry.job(phase, job_id, previous) as observation:
            trace_id = current_trace_id()
            if link_file:
                save_context(link_file, telemetry)
            state = graph.invoke(
                None if phase == "resume" else request.model_dump(),
                config={
                    **config,
                    "run_name": "document_batch",
                    "callbacks": [langchain_callback(telemetry)],
                },
                durability="sync" if link_file else None,
            )
            result = JobResult.model_validate(state)
            observation.outcome = result.status
            return result

    try:
        if phase == "mixed":
            result = invoke(build_graph(process_observed), {"max_concurrency": 3})
        else:
            directory.mkdir(parents=True, exist_ok=True)
            with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
                graph = build_graph(process_observed, checkpointer=saver)
                config: RunnableConfig = {
                    "configurable": {"thread_id": THREAD},
                    "max_concurrency": 1,
                }
                snapshot = graph.get_state(config)
                if phase == "start" and snapshot.values:
                    raise ValueError("This directory already has a job; resume or choose a new one")
                if phase == "resume" and not snapshot.next:
                    raise ValueError("No pending job; run start first in the same directory")
                result = invoke(graph, config, directory / "previous.traceparent")
    except SimulatedWorkerCrash:
        return {
            "status": "interrupted",
            "trace_id": trace_id,
            "executed_items": executed,
            "next_step": "Run resume with the same directory and model mode",
        }
    return {**result.model_dump(), "trace_id": trace_id, "executed_items": executed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("mixed", "start", "resume"), nargs="?", default="mixed")
    parser.add_argument(
        "--export",
        choices=("console", "otlp"),
        default=os.getenv("PLAYGROUND_OTEL_EXPORTER", "console"),
    )
    parser.add_argument(
        "--live", action="store_true", help="Use LLM_* settings instead of fixtures"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Consume model streams and observe chunk timing"
    )
    parser.add_argument("--directory", type=Path, default=Path(".demo-state/otel-job"))
    parser.add_argument("--repeat", type=int, default=1, help="Repeat mixed batches in one process")
    parser.add_argument("--interval", type=float, default=2, help="Seconds between mixed batches")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 1000 or not 0 <= args.interval <= 60:
        parser.error("repeat must be 1..1000 and interval must be 0..60 seconds")
    if args.phase != "mixed" and args.repeat != 1:
        parser.error("--repeat is only supported for mixed batches")
    if args.export not in {"console", "otlp"}:
        parser.error("PLAYGROUND_OTEL_EXPORTER must be console or otlp")
    model = model_from_env(streaming=args.stream) if args.live else None
    provider = os.getenv("OTEL_GENAI_PROVIDER", "other") if args.live else "fixture"
    with configured_telemetry(args.export) as telemetry:
        for iteration in range(args.repeat):
            show(
                run_example(
                    telemetry,
                    phase=args.phase,
                    directory=args.directory,
                    model=model,
                    provider=provider,
                    stream=args.stream,
                )
            )
            if iteration + 1 < args.repeat:
                sleep(args.interval)


if __name__ == "__main__":
    main()
