"""LLM providers.

Frank talks to models through a small, SDK-agnostic contract so the agent loop
never depends on one particular client. Every provider speaks the same language:
it takes a model id + messages + tool schemas and returns a ``ChatResponse``.

All providers are built on the ``openai`` library because it is thin, standard,
and every backend we support speaks the OpenAI-compatible ``/chat/completions``
endpoint:

  * OpenRouter  -> https://openrouter.ai/api/v1
  * llama.cpp   -> http://localhost:8080/v1

Adding a new backend means adding one more provider class; the agent loop never
changes.
"""

import json
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

# The messages and tools are shared plain dicts/lists (OpenAI format), produced
# by tools.py. Keeping them untyped here avoids coupling this module to tools.
MESSAGES = List[dict]
TOOLS = List[dict]

# Fallback model for OpenRouter when none is configured or passed on the CLI.
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict  # already parsed into a plain dict


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    content: str
    tool_calls: List[ToolCall]
    usage: Usage


class Provider:
    """Interface the agent loop depends on."""

    name: str = "provider"

    def chat(self, model: str, messages: MESSAGES, tools: TOOLS) -> ChatResponse:
        raise NotImplementedError  # pragma: no cover


class _OpenAICompatibleProvider(Provider):
    """Base class backed by the ``openai`` library.

    Sends the request and normalises the OpenAI-style completion into a
    ``ChatResponse``.
    """

    def __init__(self, base_url: str, api_key: str, model: Optional[str] = None):
        # ``api_key="sk-no-key"`` keeps the client happy for local servers that
        # don't authenticate.
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = model

    def chat(self, model: str, messages: MESSAGES, tools: TOOLS) -> ChatResponse:
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = response.usage or Usage()
        return ChatResponse(
            content=message.content or "",
            tool_calls=[
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_loads(tc.function.arguments),
                )
                for tc in (message.tool_calls or [])
            ],
            usage=Usage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            ),
        )


class OpenRouterProvider(_OpenAICompatibleProvider):
    name = "openrouter"


class LlamaCppProvider(_OpenAICompatibleProvider):
    """A local llama.cpp server (``llama-server``) speaking OpenAI compat."""

    name = "llama"


def to_assistant_message(response: ChatResponse) -> dict:
    """Convert a ChatResponse back into an OpenAI-style assistant message dict,
    so it can be appended to the conversation and sent back to the model."""
    message: dict = {"role": "assistant", "content": response.content or None}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": _dumps(tc.arguments)},
            }
            for tc in response.tool_calls
        ]
    return message


def _loads(raw: Optional[str]) -> dict:
    """Parse the model's function-arguments JSON string (tolerating None)."""
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def _dumps(arguments: dict) -> str:
    """Serialize tool-call arguments back to the JSON string the API expects."""
    return json.dumps(arguments)


def make_provider(
    provider: str,
    openrouter_api_key: Optional[str],
    openrouter_base_url: str,
    llama_base_url: str,
    llama_model: Optional[str],
    openrouter_model: str = "",
) -> Provider:
    """Build a provider from config values read from the environment."""
    provider = (provider or "openrouter").strip().lower()

    if provider == "openrouter":
        if not openrouter_api_key:
            raise SystemExit(
                "FRANK_PROVIDER=openrouter but OPENROUTER_API_KEY is not set. "
                "Add it to .env, or set FRANK_PROVIDER=llama for a local model."
            )
        return OpenRouterProvider(
            base_url=openrouter_base_url,
            api_key=openrouter_api_key,
            model=openrouter_model or DEFAULT_MODEL,
        )

    if provider == "llama":
        return LlamaCppProvider(
            base_url=llama_base_url, api_key="sk-no-key", model=llama_model
        )

    raise SystemExit(f"unknown FRANK_PROVIDER: {provider!r}")
