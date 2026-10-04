# Model injection

## Goal

Keep provider selection outside graph topology. A model-backed builder accepts a LangChain
`BaseChatModel`; deterministic functions or scripted models remain valid in tests.

For runnable model-backed demonstrations of every scenario, use the
[example run guide](../running-examples.md). The configuration helper in `example_support.py`
belongs only to demos; graphs continue to accept injected dependencies.

The examples load local settings with `uv run --env-file .env`. Copy the repository's
[.env.example](../../.env.example) to `.env` and configure the endpoint there.

```python
model = make_model_from_application_config()
graph = build_graph(model)
```

For LiteLLM, keep the adapter in the destination application's configuration layer:

```python
import os

from langchain_litellm import ChatLiteLLM

model = ChatLiteLLM(
    model=os.environ["LLM_MODEL"],
    api_base=os.environ.get("LLM_API_BASE", "").strip() or None,
    api_key=os.environ.get("LLM_API_KEY", "").strip() or None,
    request_timeout=60,
    max_retries=1,
    max_tokens=2048,
)
```

Choose a provider-supported model identifier. Tool calling and schema output must be supported
by the actual endpoint. Use `with_structured_output(Schema, method="function_calling")` for the
demo adapters; validate output and allow capability errors to surface. Lower temperature does
not guarantee reproducibility and some models do not accept that parameter. A schema enforces
shape, not truth. See [official model documentation](https://docs.langchain.com/oss/python/langchain/models).
