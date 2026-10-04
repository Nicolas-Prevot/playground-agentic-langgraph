# Cross-cutting recipes

These are optional integration seams, not a framework layer.

| Need | Recipe | Start with |
|---|---|---|
| Provider-neutral model injection | [Models](models.md) | Every model-backed scenario |
| Checkpoints, thread memory, Store, or subgraph scope | [Persistence](persistence.md) | 03, 04, 08, 10 |
| Tools or data exposed through MCP | [MCP](mcp.md) | 02 or 06 |
| Progressive domain instructions | [Skills](skills.md) | 07 or 09 |
| Controls inside a bounded agent loop | [Middleware](middleware.md) | 03, 04, 09 |
| Transient failures, expensive nodes, or UI streaming | [Resilience](resilience.md) | 02, 05, 06, 10 |
| Outcome and trajectory quality | [Evaluation](evaluation.md) | Every scenario |
| Tracing and trust boundaries | [Observability and security](observability-security.md) | Every scenario |

Copy a recipe into the application layer only after the destination project has identified the
need. Do not create a global “feature-enabled graph” that combines them by default.
