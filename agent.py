"""The agent loop.

One turn works like this: send the whole conversation to the model; if it
replies with tool calls, confirm + execute them, append the results to the
conversation, and go around again. When the model replies with plain text and
no tool calls, the turn is over and control returns to the REPL.
"""

import json
import os

import tools

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
    def __init__(self, client, model, confirm, debug=False):
        self.client = client
        self.model = model
        self.confirm = confirm  # callback: (tool_name, args) -> bool
        self.debug = debug
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tokens = 0          # total tokens this session (exact ints)
        self.cost_umicros = 0    # running cost in integer microdollars (1 USD = 1_000_000)
        self.cost_unknown = False  # True if any call didn't report a cost
        self.calls = []          # (total_tokens, cost_usd|None, model) per API call
        if self.debug:
            # The tools payload is identical on every request, so show it once.
            debug_dump("tools sent with every request", tools.SCHEMAS)

    def run(self, user_input: str) -> None:
        """Run one full turn: from user input until the model stops calling tools."""
        self.messages.append({"role": "user", "content": user_input})

        while True:
            if self.debug:
                debug_dump(f"request to {self.model}", self.messages)

            result = self.client.chat.send(
                model=self.model,
                messages=self.messages,
                tools=tools.SCHEMAS,
            )

            usage = result.usage
            if usage is not None:
                self.tokens += usage.total_tokens
                cost = usage.cost
                self.calls.append((usage.total_tokens, cost, self.model))
                if cost is None:
                    self.cost_unknown = True
                else:
                    # Accumulate in integer microdollars so tiny floats
                    # can't accumulate rounding drift.
                    self.cost_umicros += round(cost * 1_000_000)

            if self.debug:
                debug_dump("response", result)

            message = result.choices[0].message
            # The response message object is a valid input message, so it can go
            # straight back into the history (needed so the model sees its own
            # tool_calls next iteration).
            self.messages.append(message)

            if message.content:
                print(f"\n{message.content}\n")

            if not message.tool_calls:
                return

            for call in message.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                if self.confirm(name, args):
                    output = tools.run(name, args)
                else:
                    output = "User denied this tool call."

                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )
