"""Deterministic chat model used only by examples and tests.

Scenario graph builders accept normal LangChain models.  This helper keeps examples
repeatable without introducing a provider factory or a playground runtime.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field


@dataclass
class CallCounter:
    value: int = 0


Responder = Callable[[list[BaseMessage], Sequence[BaseTool | dict[str, Any]]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """A real ``BaseChatModel`` with deterministic scripted responses."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Responder = Field(exclude=True)
    model_name: str = "scripted"
    bound_tools: list[BaseTool | dict[str, Any]] = Field(default_factory=list, exclude=True)
    counter: CallCounter = Field(default_factory=CallCounter, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ScriptedChatModel:
        del tool_choice, kwargs
        clone = copy(self)
        clone.bound_tools = list(tools)  # type: ignore[assignment]
        clone.counter = self.counter
        return clone

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.counter.value += 1
        return ChatResult(
            generations=[ChatGeneration(message=self.responder(messages, self.bound_tools))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        self.counter.value += 1
        message = self.responder(messages, self.bound_tools)
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=message.content,
                    tool_calls=message.tool_calls,
                    usage_metadata=message.usage_metadata,
                    chunk_position="last",
                )
            )
            return
        parts = re.findall(r"\S+\s*", str(message.content)) or [""]
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=part,
                    usage_metadata=message.usage_metadata if last else None,
                    chunk_position="last" if last else None,
                )
            )
