# Observability and security

The graphs are ready to be observed because their nodes, routes, state updates, subgraphs, and
terminal statuses are explicit. The destination application owns exporters and data policy.

For a runnable reference, use the [scenario 10 OpenTelemetry lab](../observability/README.md).
It covers all three signals, exporter lifecycle, sampling, privacy, bounded metric dimensions,
trace exemplars, cross-process span links, and a local Collector/Grafana dashboard.

## Minimal OpenTelemetry boundary

Add a workflow span where an async application invokes the graph (scenario 06 uses async retrieval):

```python
from opentelemetry import trace

tracer = trace.get_tracer("my_product.workflows")

with tracer.start_as_current_span(
    "workflow.s06_grounded_knowledge", record_exception=False, set_status_on_exception=False
) as span:
    span.set_attribute("workflow.name", "s06_grounded_knowledge")
    span.set_attribute("workflow.thread_id", thread_id)
    try:
        result = await graph.ainvoke(inputs, config=config)
    except Exception:
        span.set_status(trace.StatusCode.ERROR)
        span.set_attribute("error.type", "workflow_error")
        raise
    span.set_attribute("workflow.status", result["status"])
```

Use maintained LangChain/LangGraph instrumentation or callbacks when node/model/tool spans are
needed. Do not add a playground-specific telemetry decorator to every node.

Useful span attributes:

- workflow and node names;
- route and terminal status;
- attempt, iteration, model-call, and tool-call counts;
- duration and token usage;
- checkpoint/run identifiers that contain no user content.

Use only low-cardinality labels such as workflow, node, route and status on metrics. Thread, run,
checkpoint and user IDs have high cardinality; keep them out of metric labels. Configure an SDK
and exporter to collect spans; creating a span with the API alone does not install an exporter.

Keep prompts, messages, retrieved text, tool arguments/results, memory values, and Skill contents
out of traces by default.

## Security review by boundary

| Boundary | Required controls |
|---|---|
| Model | input/output validation, budgets, provider data policy |
| Tool/executor | allowlist, authorization, schema validation, idempotency |
| Checkpointer/Store | tenant namespace, encryption, retention, deletion |
| MCP | server/tool allowlist, authentication, egress, timeout, output validation |
| Skill | trusted source, path confinement, review scripts/resources |
| Deep Agent | backend isolation, permissions, tool limits, artifact validation |
| Human approval | authenticated actor, durable audit, typed resume value |

Security controls belong at the smallest boundary that can enforce them. A parent graph should not
trust a bounded agent merely because the agent has a restrictive prompt.
