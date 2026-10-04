"""Check emitted signals and wire format, not the implementation's call sequence."""

import json
import logging
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from opentelemetry import trace
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider, TraceBasedExemplarFilter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import StatusCode

from playground_agentic.example_support import model_from_env
from playground_agentic.scenarios.s10_resumable.graph import JobRequest, WorkItem, build_graph, run
from playground_agentic.scenarios.s10_resumable.otel_example import run_example
from playground_agentic.scenarios.s10_resumable.otel_sdk import SafeSpanExporter, langchain_callback
from playground_agentic.scenarios.s10_resumable.telemetry import JobTelemetry
from playground_agentic.testing import ScriptedChatModel


@contextmanager
def capture(ratio=1):
    resource = Resource.create({"service.name": "observability-test"})
    spans, records, reader = (
        InMemorySpanExporter(),
        InMemoryLogRecordExporter(),
        InMemoryMetricReader(),
    )
    traces = TracerProvider(resource=resource, sampler=ParentBased(TraceIdRatioBased(ratio)))
    traces.add_span_processor(SimpleSpanProcessor(SafeSpanExporter(delegate=spans)))
    logs = LoggerProvider(resource=resource)
    logs.add_log_record_processor(SimpleLogRecordProcessor(records))
    metrics = MeterProvider(
        resource=resource,
        metric_readers=[reader],
        exemplar_filter=TraceBasedExemplarFilter(),
    )
    telemetry = JobTelemetry(traces, metrics, logs)
    try:
        yield telemetry, spans, records, reader
    finally:
        telemetry.close()


def metrics_by_name(reader):
    return {
        metric.name: metric
        for resource in reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }


def test_parallel_batch_correlates_signals_and_counts_actual_work():
    with capture() as (telemetry, exporter, logs, reader):
        result = run_example(telemetry)
        assert result["status"] == "partial"
        spans = exporter.get_finished_spans()
        roots = [span for span in spans if span.parent is None]
        assert len(roots) == 1
        root = roots[0]
        assert root.name == "job.invoke"
        assert root.status.status_code is StatusCode.UNSET  # Valid business outcome.
        assert root.attributes["workflow.outcome"] == "partial"
        workers = [span for span in spans if span.name == "worker.process"]
        assert len(workers) == 4
        workflows = [
            s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_workflow"
        ]
        assert len(workflows) == 1
        workflow = workflows[0]
        assert workflow.parent.span_id == root.context.span_id
        assert all(span.parent.span_id == workflow.context.span_id for span in workers)
        models = [span for span in spans if span.name == "chat fixture-summary"]
        assert len(models) == 2
        # Official callback instrumentation follows LangChain run ancestry. The plain
        # business workers and model invocations are siblings under the workflow.
        assert all(span.parent.span_id == workflow.context.span_id for span in models)
        assert all(
            s.instrumentation_scope.name == "opentelemetry.instrumentation.genai.langchain"
            for s in [workflow, *models]
        )
        assert all(span.context.trace_id == root.context.trace_id for span in spans)
        assert sum(s.status.status_code is StatusCode.ERROR for s in workers) == 2
        records = [item.log_record for item in logs.get_finished_logs()]
        assert len(records) == 6  # Start/end and four attempts; model telemetry is automatic.
        assert all(record.trace_id == root.context.trace_id for record in records)
        assert all(record.span_id in {s.context.span_id for s in spans} for record in records)
        metrics = metrics_by_name(reader)
        attempts = metrics["playground.worker.attempts"].data.data_points
        assert {p.attributes["work.outcome"]: p.value for p in attempts} == {
            "completed": 2,
            "retryable": 1,
            "failed": 1,
        }
        assert metrics["playground.worker.retries"].data.data_points[0].value == 1
        assert metrics["playground.worker.active"].data.data_points[0].value == 0
        assert metrics["playground.job.invocations"].data.data_points[0].value == 1
        token_metric = metrics["gen_ai.client.token.usage"]
        assert token_metric.unit == "{token}"
        assert {
            p.attributes["gen_ai.token.type"]: (p.sum, p.count)
            for p in token_metric.data.data_points
        } == {"input": (80, 2), "output": (32, 2)}
        duration = metrics["playground.worker.duration"]
        assert duration.unit == "s"
        assert sum(p.count for p in duration.data.data_points) == 4
        exemplars = [e for p in duration.data.data_points for e in p.exemplars]
        assert exemplars and all(e.trace_id == root.context.trace_id for e in exemplars)
        for metric in metrics.values():
            for point in metric.data.data_points:
                assert not any(key.endswith(".id") for key in point.attributes)


def test_private_data_does_not_reach_any_signal_and_errors_remain_visible():
    private_data = "private-key-and-document-content-123"

    def fail(_item, _attempt):
        raise RuntimeError(private_data)

    with capture() as (telemetry, exporter, logs, reader):
        with pytest.raises(RuntimeError, match=private_data), telemetry.job("mixed", "opaque-job"):
            run(
                JobRequest(items=[WorkItem(id="opaque-item", payload=private_data)]),
                compiled_graph=build_graph(telemetry.processor(fail)),
                config={"callbacks": [langchain_callback(telemetry)]},
            )
        spans = exporter.get_finished_spans()
        assert len(spans) == 3  # Business invocation + automatic workflow + worker.
        assert all(span.status.status_code is StatusCode.ERROR for span in spans)
        assert all(not span.events for span in spans)  # No raw exception message/stack.
        rendered = " ".join(span.to_json() for span in spans)
        rendered += " ".join(r.to_json() for r in logs.get_finished_logs())
        rendered += reader.get_metrics_data().to_json()
        assert private_data not in rendered
        assert "worker_error" in rendered and "workflow_error" in rendered


def test_unsampled_trace_still_has_metrics_and_correlated_logs():
    with capture(ratio=0) as (telemetry, exporter, logs, reader):
        result = run_example(telemetry)
        assert not exporter.get_finished_spans()
        assert all(
            r.log_record.trace_id == int(result["trace_id"], 16) for r in logs.get_finished_logs()
        )
        assert metrics_by_name(reader)["playground.job.invocations"].data.data_points[0].value == 1


def test_missing_model_usage_is_not_reported_as_zero_and_model_errors_are_sanitized():
    def fail(_messages, _tools):
        raise ValueError("sensitive provider response")

    with capture() as (telemetry, exporter, logs, reader):
        with telemetry.job("mixed", "test") as job:
            config = {"callbacks": [langchain_callback(telemetry)]}
            ScriptedChatModel(responder=lambda *_: AIMessage(content="private output")).invoke(
                "private input", config=config
            )
            with pytest.raises(ValueError, match="sensitive"):
                ScriptedChatModel(responder=fail).invoke("private input", config=config)
            job.outcome = "partial"
        metrics = metrics_by_name(reader)
        assert not any(".usage." in name for name in metrics)
        duration = metrics["gen_ai.client.operation.duration"].data.data_points
        assert sum(p.count for p in duration) == 2
        assert any("ValueError" in p.attributes.get("error.type", "") for p in duration)
        rendered = " ".join(s.to_json() for s in exporter.get_finished_spans())
        rendered += " ".join(r.to_json() for r in logs.get_finished_logs())
        assert "private output" not in rendered and "sensitive provider response" not in rendered


def test_instrumentation_reuses_providers_without_changing_global_handlers():
    original_provider = trace.get_tracer_provider()
    original_handlers = list(logging.getLogger().handlers)
    with capture() as (telemetry, _, _, _):
        assert trace.get_tracer_provider() is original_provider
        assert logging.getLogger().handlers == original_handlers
    assert not telemetry.logger.handlers
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry


def test_streaming_emits_chunk_timing_without_duplicate_inference_or_usage():
    with capture() as (telemetry, exporter, _logs, reader):
        assert run_example(telemetry, stream=True)["status"] == "partial"
        spans = exporter.get_finished_spans()
        models = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat"]
        assert len(models) == 2
        assert all(s.attributes.get("gen_ai.request.stream") for s in models)
        metrics = metrics_by_name(reader)
        first_chunk = metrics["gen_ai.client.operation.time_to_first_chunk"]
        assert sum(p.count for p in first_chunk.data.data_points) == 2
        assert "gen_ai.client.operation.time_per_output_chunk" in metrics
        usage = metrics["gen_ai.client.token.usage"]
        assert {p.attributes["gen_ai.token.type"]: p.sum for p in usage.data.data_points} == {
            "input": 80,
            "output": 32,
        }


def test_official_callback_observes_tools_without_capturing_arguments_or_results():
    @tool
    def lookup_document(document_id: str) -> str:
        """Look up a private document."""
        return "private-document-content-321"

    with capture() as (telemetry, exporter, logs, reader):
        with telemetry.job("mixed", "test") as job:
            lookup_document.invoke(
                {"document_id": "private-document-id-321"},
                config={"callbacks": [langchain_callback(telemetry)]},
            )
            job.outcome = "completed"
        spans = exporter.get_finished_spans()
        tools = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "execute_tool"]
        assert len(tools) == 1
        assert "gen_ai.execute_tool.duration" in metrics_by_name(reader)
        rendered = " ".join(s.to_json() for s in spans)
        rendered += " ".join(r.to_json() for r in logs.get_finished_logs())
        assert "private-document-content-321" not in rendered
        assert "private-document-id-321" not in rendered


def test_reference_rejects_content_capture_before_calling_a_model(monkeypatch):
    with capture() as (telemetry, exporter, _logs, _reader):
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT")
        with pytest.raises(ValueError, match="NO_CONTENT"):
            langchain_callback(telemetry)
        assert not exporter.get_finished_spans()


@pytest.mark.parametrize("stream", [False, True])
def test_real_litellm_adapter_reports_provider_model_and_usage(monkeypatch, stream):
    import tiktoken

    # Prevent LiteLLM's import-time tokenizer/cost downloads. No token estimation
    # runs in this test; the mocked provider response supplies actual usage fields.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: Mock())
    from langchain_litellm import ChatLiteLLM

    # Keep the actual LangChain adapter/callback path, replacing only provider I/O.
    def completion(_self, **kwargs):
        assert kwargs["stream"] is stream
        usage = {"prompt_tokens": 27, "completion_tokens": 9, "total_tokens": 36}
        if stream:
            assert kwargs["stream_options"] == {"include_usage": True}
            return iter(
                [
                    {"choices": [{"delta": {"role": "assistant", "content": "private "}}]},
                    {"choices": [{"delta": {"content": "summary"}}]},
                    {"choices": [], "usage": usage},
                ]
            )
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "private summary"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    monkeypatch.setattr(ChatLiteLLM, "completion_with_retry", completion)
    monkeypatch.setenv("LLM_MODEL", "openai/adapter-test")
    monkeypatch.setenv("LLM_API_KEY", "test-only")
    monkeypatch.setenv("LLM_API_BASE", "")
    model = model_from_env(streaming=stream)
    with capture() as (telemetry, exporter, _logs, reader):
        assert (
            run_example(telemetry, model=model, provider="openai", stream=stream)["status"]
            == "partial"
        )
        models = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes.get("gen_ai.operation.name") == "chat"
        ]
        assert len(models) == 2
        assert all(s.attributes["gen_ai.provider.name"] == "openai" for s in models)
        assert all(s.attributes["gen_ai.request.model"] == "openai/adapter-test" for s in models)
        usage = metrics_by_name(reader)["gen_ai.client.token.usage"]
        assert {p.attributes["gen_ai.token.type"]: p.sum for p in usage.data.data_points} == {
            "input": 54,
            "output": 18,
        }
        assert "private summary" not in " ".join(s.to_json() for s in models)
        if stream:
            assert all(s.attributes.get("gen_ai.request.stream") for s in models)
            timing = metrics_by_name(reader)["gen_ai.client.operation.time_to_first_chunk"]
            assert sum(p.count for p in timing.data.data_points) == 2


def test_streaming_example_rejects_silent_adapter_fallback():
    model = ScriptedChatModel(
        responder=lambda *_: AIMessage(content="A complete non-streaming answer"),
        disable_streaming=True,
    )
    with (
        capture() as (telemetry, _spans, _logs, _reader),
        pytest.raises(ValueError, match="non-streaming response"),
    ):
        run_example(telemetry, model=model, stream=True)


@pytest.mark.integration
def test_declarative_profile_controls_sampling_resource_and_metric_dimensions():
    # SDK globals are set only once: exercise the public bootstrap in a fresh process.
    from playground_agentic.scenarios.s10_resumable import otel_sdk

    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env.update(
        {
            "OTEL_CONFIG_FILE": str(Path(otel_sdk.__file__).with_name("otel-console.yaml")),
            "OTEL_SERVICE_NAME": "yaml-profile-test",
            "OTEL_TRACES_SAMPLER_ARG": "0.0",
            # This implicit env setting must not override the authoritative YAML sampler.
            "OTEL_TRACES_SAMPLER": "always_on",
        }
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from playground_agentic.scenarios.s10_resumable.otel_sdk import configured_telemetry
from playground_agentic.scenarios.s10_resumable.otel_example import run_example
with configured_telemetry('otlp') as telemetry:
    run_example(telemetry)
    telemetry.metrics.get_meter('test').create_counter('playground.job.test').add(
        1, {'workflow.name': 'document_batch', 'user.id': 'secret-identity'})
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert not process.stderr
    assert "secret-identity" not in process.stdout
    decoder = json.JSONDecoder()
    remaining, exported = process.stdout.strip(), []
    while remaining:
        value, end = decoder.raw_decode(remaining)
        exported.append(value)
        remaining = remaining[end:].strip()
    assert not any("context" in value for value in exported)  # No sampled spans.
    metric_resources = [r for value in exported for r in value.get("resource_metrics", [])]
    assert metric_resources
    log_records = [value for value in exported if "body" in value]
    assert log_records and all(
        record["trace_id"] != "0x00000000000000000000000000000000" for record in log_records
    )
    resources = [r["resource"]["attributes"] for r in metric_resources]
    resources.extend(r["resource"]["attributes"] for r in log_records)
    assert all(r["service.name"] == "yaml-profile-test" for r in resources)
    assert len({r["service.instance.id"] for r in resources}) == 1
    test_points = [
        point
        for resource in metric_resources
        for scope in resource["scope_metrics"]
        for metric in scope["metrics"]
        if metric["name"] == "playground.job.test"
        for point in metric["data"]["data_points"]
    ]
    assert test_points and all(
        p["attributes"] == {"workflow.name": "document_batch"} for p in test_points
    )


@pytest.mark.integration
def test_fresh_process_resume_exports_all_three_otlp_signals_and_links_traces(tmp_path):
    received = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Isolate this wire test from the developer's configured OTLP endpoints/headers.
    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"http://127.0.0.1:{server.server_port}"
    outputs = []
    try:
        for phase in ("start", "resume"):
            process = subprocess.run(  # noqa: S603 - fixed interpreter/module, local fixture path
                [
                    sys.executable,
                    "-m",
                    "playground_agentic.scenarios.s10_resumable.otel_example",
                    phase,
                    "--export",
                    "otlp",
                    "--directory",
                    str(tmp_path / "job"),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            assert not process.stderr
            outputs.append(json.loads(process.stdout))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert outputs[0]["status"] == "interrupted"
    assert outputs[1]["status"] == "completed"
    assert outputs[1]["executed_items"] == ["runbook"]
    assert {path for path, _ in received} == {"/v1/traces", "/v1/logs", "/v1/metrics"}
    spans, records, metrics = [], [], []
    for path, body in received:
        if path == "/v1/traces":
            batch = ExportTraceServiceRequest.FromString(body)
            spans.extend(
                s for r in batch.resource_spans for scope in r.scope_spans for s in scope.spans
            )
        elif path == "/v1/logs":
            batch = ExportLogsServiceRequest.FromString(body)
            records.extend(
                r
                for resource in batch.resource_logs
                for scope in resource.scope_logs
                for r in scope.log_records
            )
        else:
            batch = ExportMetricsServiceRequest.FromString(body)
            metrics.extend(
                m
                for r in batch.resource_metrics
                for scope in r.scope_metrics
                for m in scope.metrics
            )
    roots = {s.trace_id.hex(): s for s in spans if not s.parent_span_id}
    start, resume = (roots[o["trace_id"]] for o in outputs)
    assert start.trace_id != resume.trace_id
    assert resume.links[0].trace_id == start.trace_id
    assert resume.links[0].span_id == start.span_id
    assert Counter(s.name for s in spans)["worker.process"] == 3
    assert all(r.trace_id.hex() in roots for r in records)
    assert any(m.name == "playground.job.invocations" for m in metrics)
    assert any(m.name == "gen_ai.client.token.usage" for m in metrics)
    assert any(s.name == "invoke_workflow document_batch" for s in spans)
    assert all(not s.status.message for s in spans)
