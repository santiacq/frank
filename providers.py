"""LLM providers.

Frank talks to models through a small, SDK-agnostic contract so the agent loop
never depends on one particular client. Every provider speaks the same language:
it takes a model id + messages + tool schemas and returns a ``ChatResponse``.

All providers are built on the ``openai`` library because it is thin, standard,
and every backend we support speaks the OpenAI-compatible ``/chat/completions``
endpoint:

  * OpenRouter  -> https://openrouter.ai/api/v1
  * llama.cpp   -> http://localhost:8080/v1

Adding a new backend means adding one more provider instance; the agent loop
never changes.
"""

import json
from dataclasses import dataclass

from openai import OpenAI

# The messages and tools are shared plain dicts/lists (OpenAI format), produced
# by tools.py. Keeping them untyped here avoids coupling this module to tools.
MESSAGES = list[dict]
TOOLS = list[dict]

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
    tool_calls: list[ToolCall]
    usage: Usage | None


class Provider:
    """One OpenAI-compatible endpoint the agent loop can talk to.

    Thin wrapper around the ``openai`` client that normalises completions into
    ``ChatResponse``.
    """

    def __init__(self, base_url: str, api_key: str, model: str | None = None):
        # ``api_key="sk-no-key"`` keeps the client happy for local servers that
        # don't authenticate.
        self.name = base_url  # used in /stats; a URL tells you which backend
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = model

    def chat(self, model: str, messages: MESSAGES, tools: TOOLS) -> ChatResponse:
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
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
            usage=_usage(response.usage),
        )


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


def _usage(usage) -> Usage | None:
    """Normalise an openai CompletionUsage into our own Usage (or None)."""
    if usage is None:
        return None
    return Usage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )


def _loads(raw: str | None) -> dict:
    """Parse the model's function-arguments JSON string (tolerating None)."""
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def _dumps(arguments: dict) -> str:
    """Serialize tool-call arguments back to the JSON string the API expects."""
    return json.dumps(arguments)


def make_provider(
    name: str,
    base_url: str,
    api_key: str | None,
    model: str | None = None,
) -> Provider:
    """Build a provider for one OpenAI-compatible endpoint.

    ``name`` is only checked to route error messages for the two backends we
    know about -- the object returned is the same either way.
    """
    if not api_key and name != "llama":
        raise SystemExit(
            f"FRANK_PROVIDER={name} but no API key was found. "
            "Add it to .env, or set FRANK_PROVIDER=llama for a local model."
        )
    default_model = model or (DEFAULT_MODEL if name == "openrouter" else None)
    return Provider(base_url=base_url, api_key=api_key or "sk-no-key", model=default_model)
