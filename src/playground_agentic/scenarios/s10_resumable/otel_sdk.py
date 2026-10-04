"""One application bootstrap; maintained framework instrumentation; explicit data policy.

The two packaged YAML profiles construct the SDK through the public declarative API.
The exporter below only enforces this example's content policy; it delegates transport,
batching, retries and encoding to OpenTelemetry. See the guide for the beta API limits.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from opentelemetry import metrics, trace
from opentelemetry._logs import get_logger_provider
from opentelemetry.configuration import configure_sdk, load_config_file
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.instrumentation.genai.langchain.version import __version__
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SpanExporter, SpanExportResult
from opentelemetry.trace import Status
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.re import parse_env_headers

from .telemetry import JobTelemetry

# An allowlist is deliberate: NO_CONTENT alone does not remove exception descriptions.
SPAN_ATTRIBUTES = {
    "workflow.name",
    "workflow.phase",
    "workflow.outcome",
    "workflow.job.id",
    "work.item.id",
    "work.attempt",
    "work.outcome",
    "error.type",
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.workflow.name",
    "gen_ai.agent.name",
    "gen_ai.tool.name",
    "gen_ai.tool.type",
    "gen_ai.conversation.id",
    "gen_ai.request.model",
    "gen_ai.request.stream",
    "gen_ai.response.model",
    "gen_ai.response.finish_reasons",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_write.input_tokens",
    "gen_ai.usage.reasoning.output_tokens",
}


class SafeSpanExporter(SpanExporter):
    """Public exporter extension registered in pyproject.toml for declarative profiles."""

    def __init__(
        self,
        protocol: str = "console",
        endpoint: str | None = None,
        headers: str | None = None,
        *,
        delegate: SpanExporter | None = None,
    ) -> None:
        if delegate is not None:
            self.delegate = delegate
        elif protocol == "console":
            self.delegate = ConsoleSpanExporter()
        elif protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            self.delegate = OTLPSpanExporter(
                endpoint=endpoint, headers=parse_env_headers(headers or "", liberal=True), timeout=5
            )
        else:
            raise ValueError("Unsupported trace exporter protocol")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        safe_spans = [
            ReadableSpan(
                name=span.name,
                context=span.context,
                parent=span.parent,
                resource=span.resource,
                attributes={
                    k: v for k, v in (span.attributes or {}).items() if k in SPAN_ATTRIBUTES
                },
                events=(),
                links=span.links,
                kind=span.kind,
                status=Status(span.status.status_code),
                start_time=span.start_time,
                end_time=span.end_time,
                instrumentation_scope=span.instrumentation_scope,
            )
            for span in spans
        ]
        return self.delegate.export(safe_spans)

    def shutdown(self) -> None:
        self.delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.delegate.force_flush(timeout_millis)


def langchain_callback(telemetry: JobTelemetry) -> OpenTelemetryLangChainCallbackHandler:
    """The official callback, scoped to one graph invocation and inherited by its models.

    Using config['callbacks'] avoids the 1.2b0 global instrumentor injecting itself into
    LangGraph's separate graph-lifecycle manager (where it has no on_resume method).
    No third-party class is patched and no callback implementation is copied here.
    """
    if (
        os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT")
        != "NO_CONTENT"
    ):
        raise ValueError(
            "This demo requires OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT"
        )
    if os.getenv("OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK"):
        raise ValueError("This demo does not enable content-upload completion hooks")
    if os.getenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", "false").lower() != "false":
        raise ValueError("This demo keeps GenAI content events disabled")
    if LangChainInstrumentor().is_instrumented_by_opentelemetry:
        raise RuntimeError(
            "LangChain is already instrumented; reuse the owning application's setup"
        )
    handler = TelemetryHandler(
        tracer_provider=telemetry.traces,
        meter_provider=telemetry.metrics,
        logger_provider=telemetry.logs,
        instrumentation_scope_name="opentelemetry.instrumentation.genai.langchain",
        instrumentation_scope_version=__version__,
    )
    return OpenTelemetryLangChainCallbackHandler(telemetry_handler=handler)


@contextmanager
def configured_telemetry(export: Literal["console", "otlp"] = "console") -> Iterator[JobTelemetry]:
    """Standalone process entry point. Configure once; reusable library code never calls this."""
    if (
        isinstance(trace.get_tracer_provider(), TracerProvider)
        or isinstance(metrics.get_meter_provider(), MeterProvider)
        or isinstance(get_logger_provider(), LoggerProvider)
    ):
        raise RuntimeError("An SDK is already configured; reuse its providers instead")
    path = Path(os.getenv("OTEL_CONFIG_FILE") or Path(__file__).with_name(f"otel-{export}.yaml"))
    config = load_config_file(path)
    if config.disabled or not all(
        (config.tracer_provider, config.meter_provider, config.logger_provider)
    ):
        raise ValueError("The lab requires all three SDK providers in its declarative profile")
    configure_sdk(config)
    traces, meters, logs = (
        trace.get_tracer_provider(),
        metrics.get_meter_provider(),
        get_logger_provider(),
    )
    if (
        not isinstance(traces, TracerProvider)
        or not isinstance(meters, MeterProvider)
        or not isinstance(logs, LoggerProvider)
    ):
        raise ValueError("The lab requires all three SDK providers in its declarative profile")
    telemetry = JobTelemetry(traces, meters, logs)
    try:
        yield telemetry
    finally:
        telemetry.close()
