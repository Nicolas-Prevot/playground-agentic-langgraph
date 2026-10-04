# Middleware inside bounded agents

Middleware controls a `create_agent` or Deep Agent loop. It is not a replacement for explicit
StateGraph routing around that loop.

Use maintained middleware where it owns the concern:

- model- and tool-call limits for loop budgets;
- model/tool retry for transient provider or tool failures;
- model fallback for provider degradation;
- summarization or context editing for long conversations;
- PII handling and guardrails around model/tool calls;
- HITL middleware for approval of particular agent-selected tools.

```python
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[ModelCallLimitMiddleware(run_limit=4, exit_behavior="error")],
)

workflow = StateGraph(State).add_node("bounded_agent", agent)
```

Use ordinary graph nodes and conditional edges when a decision changes application topology. Use
middleware only when the control belongs inside the bounded agent lifecycle. Test middleware
configuration independently from parent-graph routes.

