# Retries, caching, and streaming

These are execution policies, not new scenarios.

## Node retry

Apply LangGraph `RetryPolicy` to an idempotent node that can fail transiently. Do not retry schema
errors, policy denials, or permanent domain failures.

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "retrieve",
    retrieve,
    retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
)
```

Scenario 10 additionally shows graph-visible domain retry: a worker returns `retryable`, the graph
increments the item attempt, and exhausted items become explicit partial failures.

## Node caching

Use `CachePolicy` for deterministic, expensive reads—not user-specific or mutating work.

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

builder.add_node("retrieve", retrieve, cache_policy=CachePolicy(ttl=120))
graph = builder.compile(cache=InMemoryCache())
```

The destination project must define cache keys, tenant isolation, invalidation, and sensitive-data
policy.

## Streaming

Streaming does not change topology. Select the smallest mode the caller needs:

- `messages` for model tokens;
- `updates` for node state deltas;
- `values` for complete state snapshots;
- `custom` for application progress emitted by nodes;
- subgraph namespaces when the caller must distinguish nested execution.

Keep UI rendering and transport-specific event schemas outside scenario graphs.

## Budgets and failure classes

`recursion_limit` is a final execution guard, not a business completion rule. Graphs should reach
explicit partial or budget-exhausted states first. Cap model iterations, tool calls per run,
dynamic worker count and concurrency separately. Provider retries multiply the number of actual
requests; cap total tokens/cost and wall time at the application boundary.

| Failure | Response |
|---|---|
| Transient network/rate-limit error | Bounded retry with backoff, respecting provider guidance |
| Invalid model output | Fail visibly or use a separately bounded repair path |
| Missing user data or approval | Interrupt with a typed payload |
| Permanent per-item domain error | Explicit failed item and partial job result |
| Programming error | Propagate for diagnosis; do not fabricate success |
| Unknown external effect outcome | Reconcile using durable operation identity |

Set `max_concurrency` in invocation config for `Send` workloads. Checkpointing does not enforce a
provider's rate limit. Set timeouts on tools, retrieval and model clients. Synchronous clients need
care under async serving: cancellation cannot necessarily stop an already running remote effect.
