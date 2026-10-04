"""Optional application-owned OTel setup and boundaries for this scenario only.

Business instrumentation only. Framework/GenAI instrumentation and SDK construction
belong to otel_sdk.py and its declarative profiles. The graph depends on neither.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Link, Span, SpanContext, StatusCode

from .graph import Processor, WorkItem, WorkOutcome

SCOPE = "playground_agentic.s10_resumable"
WORKFLOW = "document_batch"
DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120)


def mark_error(span: Span, error_type: str) -> None:
    """Do not export exception messages/stack traces: providers may embed input or keys."""
    span.set_attribute("error.type", error_type)
    span.set_status(StatusCode.ERROR)


@dataclass
class JobObservation:
    span: Span
    outcome: str = "error"


class JobTelemetry:
    """Injectable providers make tests deterministic and keep SDK ownership explicit."""

    def __init__(
        self, traces: TracerProvider, metrics: MeterProvider, logs: LoggerProvider
    ) -> None:
        self.traces, self.metrics, self.logs = traces, metrics, logs
        self.tracer = traces.get_tracer(SCOPE, "0.1.0")
        meter = metrics.get_meter(SCOPE, "0.1.0")
        self.jobs = meter.create_counter("playground.job.invocations", unit="{invocation}")
        self.job_duration = meter.create_histogram(
            "playground.job.duration",
            unit="s",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )
        self.attempts = meter.create_counter("playground.worker.attempts", unit="{attempt}")
        self.worker_duration = meter.create_histogram(
            "playground.worker.duration",
            unit="s",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )
        self.retries = meter.create_counter("playground.worker.retries", unit="{attempt}")
        self.active = meter.create_up_down_counter("playground.worker.active", unit="{worker}")
        # A private logger avoids intercepting third-party messages or changing root handlers.
        self.logger = logging.Logger(SCOPE, level=logging.INFO)
        self.logger.propagate = False
        self.handler = LoggingHandler(logger_provider=logs)
        self.logger.addHandler(self.handler)

    @contextmanager
    def job(
        self, phase: str, job_id: str, previous: SpanContext | None = None
    ) -> Iterator[JobObservation]:
        attrs = {"workflow.name": WORKFLOW, "workflow.phase": phase}
        # Each CLI invocation is a new trace. Durable predecessors are links, not parents.
        with self.tracer.start_as_current_span(
            "job.invoke",
            context=Context(),
            links=[Link(previous)] if previous and previous.is_valid else [],
            attributes={**attrs, "workflow.job.id": job_id},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            observation = JobObservation(span)
            started = perf_counter()
            self.logger.info("job.started", extra={**attrs, "workflow.job.id": job_id})
            try:
                yield observation
            except Exception:
                observation.outcome = "error"
                mark_error(span, "workflow_error")
                raise
            finally:
                dimensions = {**attrs, "workflow.outcome": observation.outcome}
                span.set_attribute("workflow.outcome", observation.outcome)
                self.jobs.add(1, dimensions)
                self.job_duration.record(perf_counter() - started, dimensions)
                level = {
                    "completed": logging.INFO,
                    "partial": logging.WARNING,
                    "failed": logging.WARNING,
                    "error": logging.ERROR,
                }.get(observation.outcome, logging.ERROR)
                self.logger.log(level, "job.finished", extra=dimensions)

    def processor(self, process: Processor) -> Processor:
        def observed(item: WorkItem, attempt: int) -> WorkOutcome:
            attrs = {"workflow.name": WORKFLOW}
            with self.tracer.start_as_current_span(
                "worker.process",
                attributes={**attrs, "work.item.id": item.id, "work.attempt": attempt},
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                started = perf_counter()
                status = "error"
                self.active.add(1, attrs)
                if attempt > 1:
                    self.retries.add(1, attrs)
                try:
                    result = process(item, attempt)
                    status = result.status
                    if status != "completed":
                        mark_error(
                            span,
                            "transient_dependency"
                            if status == "retryable"
                            else "unprocessable_item",
                        )
                    return result
                except Exception:
                    mark_error(span, "worker_error")
                    raise
                finally:
                    dimensions = {**attrs, "work.outcome": status}
                    span.set_attribute("work.outcome", status)
                    self.attempts.add(1, dimensions)
                    self.worker_duration.record(perf_counter() - started, dimensions)
                    self.active.add(-1, attrs)
                    level = logging.INFO if status == "completed" else logging.WARNING
                    if status == "error":
                        level = logging.ERROR
                    self.logger.log(
                        level,
                        "worker.finished",
                        extra={
                            **dimensions,
                            "work.item.id": item.id,
                            "work.attempt": attempt,
                        },
                    )

        return observed

    def close(self) -> None:
        # Metrics flush first while trace/log exporters are still alive. All own providers.
        try:
            self.metrics.force_flush(timeout_millis=5000)
            self.traces.force_flush(timeout_millis=5000)
            self.logs.force_flush(timeout_millis=5000)
        finally:
            self.logger.removeHandler(self.handler)
            self.handler.close()
            self.metrics.shutdown(timeout_millis=5000)
            self.traces.shutdown()
            self.logs.shutdown()


def current_trace_id() -> str:
    return f"{trace.get_current_span().get_span_context().trace_id:032x}"
