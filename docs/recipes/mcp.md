# MCP as an adapter

MCP changes how a graph reaches a capability; it should not define the graph shape.

Scenario 06 therefore depends on a small `KnowledgeClient` protocol. The in-process and MCP
implementations satisfy the same contract:

```python
client = McpKnowledgeClient()
graph = build_graph(client)
```

Use the same approach elsewhere:

- Scenario 02: load selected MCP tools and pass them to the explicit tool loop.
- Scenario 05: give one worker a narrow MCP-backed dependency.
- Scenario 09: expose only explicitly approved MCP tools to the bounded Deep Agent.

Keep MCP client construction, transport lifecycle, authentication, and response parsing in an
adapter module. Validate the adapter output with Pydantic before it enters graph state.

## Security boundary

- Allowlist server identities and tool names independently.
- Treat tool descriptions and results as untrusted input.
- Enforce egress, authentication, timeouts, payload-size limits, and schema validation.
- Do not assume Deep Agent filesystem permissions constrain MCP or custom tools.
- Keep mutating MCP tools behind the same authorization and approval boundary as ordinary tools.

