"""The agent loop.

One turn works like this: send the whole conversation to the model; if it
replies with tool calls, confirm + execute them, append the results to the
conversation, and go around again. When the model replies with plain text and
no tool calls, the turn is over and control returns to the REPL.
"""

import json
import os

import tools
from providers import to_assistant_message

SYSTEM_PROMPT = f"""\
You are Frank, a coding agent running in a terminal.
Current working directory: {os.getcwd()}

Use the available tools to inspect and modify code. Prefer edit_file for small
changes and write_file for new files. Keep answers short; the user is in a
terminal. The user may deny a tool call -- if so, adapt or ask what to do.
"""

DIM = "\033[2m"
RESET = "\033[0m"


def _dump(obj):
    """Make pydantic models JSON-serializable for debug output."""
    return obj.model_dump(exclude_none=True) if hasattr(obj, "model_dump") else obj


def debug_dump(label: str, data) -> None:
    body = json.dumps(_dump(data), indent=2, default=_dump)
    print(f"{DIM}[debug] {label}\n{body}{RESET}")


class Agent:
    def __init__(self, provider, model, confirm, debug=False):
        self.provider = provider  # an OpenRouter/llama provider exposing .chat()
        self.model = model
        self.confirm = confirm  # callback: (tool_name, args) -> bool
        self.debug = debug
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tokens = 0          # total tokens this session (exact ints)
        self.calls = 0           # number of API calls this session
        if self.debug:
            # The tools payload is identical on every request, so show it once.
            debug_dump("tools sent with every request", tools.SCHEMAS)

    def clear(self) -> None:
        """Reset the session: conversation back to the system prompt, stats to zero."""
        self.messages = [self.messages[0]]
        self.tokens = 0
        self.calls = 0

    def run(self, user_input: str) -> None:
        """Run one full turn: from user input until the model stops calling tools."""
        self.messages.append({"role": "user", "content": user_input})

        while True:
            if self.debug:
                debug_dump(f"request to {self.model}", self.messages)

            result = self.provider.chat(
                model=self.model,
                messages=self.messages,
                tools=tools.SCHEMAS,
            )

            usage = result.usage
            if usage is not None:
                self.tokens += usage.total_tokens
            self.calls += 1

            if self.debug:
                debug_dump("response", result)

            message = to_assistant_message(result)  # ChatResponse -> assistant message dict
            # It is a plain dict now, so it can go straight back into the history
            # (needed so the model sees its own tool_calls next iteration).
            self.messages.append(message)

            if result.content:
                print(f"\n{result.content}\n")

            if not result.tool_calls:
                return

            for call in result.tool_calls:
                name = call.name
                args = call.arguments

                if self.confirm(name, args):
                    output = tools.run(name, args)
                else:
                    output = "User denied this tool call."

                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )
