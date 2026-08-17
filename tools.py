"""Frank's tools.

Each tool is two things:
  1. a JSON schema describing it to the model (sent with every API request)
  2. a plain Python function that actually runs it locally

The model never executes anything itself -- it only replies with "call tool X
with these arguments", and the agent loop dispatches to the functions below.
"""

import subprocess
from pathlib import Path

# Cap tool output so one huge file or noisy command can't flood the context.
MAX_OUTPUT_CHARS = 10_000


def bash(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=120
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        output += f"\n(exit code {result.returncode})"
    return output.strip() or "(no output)"


def read_file(path: str) -> str:
    return Path(path).read_text()


def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} characters to {path}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    p = Path(path)
    content = p.read_text()
    count = content.count(old_string)
    if count == 0:
        return "Error: old_string was not found in the file."
    if count > 1:
        return (
            f"Error: old_string appears {count} times; it must be unique. "
            "Include more surrounding context."
        )
    p.write_text(content.replace(old_string, new_string))
    return f"Edited {path}"


# name -> {function to run, schema info to advertise}
TOOLS = {
    "bash": {
        "fn": bash,
        "description": "Run a shell command and return its combined stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
            },
            "required": ["command"],
        },
    },
    "read_file": {
        "fn": read_file,
        "description": "Read a text file and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "fn": write_file,
        "description": (
            "Write content to a file, creating it (and parent directories) if "
            "needed, overwriting it if it exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "content": {"type": "string", "description": "Full new content of the file."},
            },
            "required": ["path", "content"],
        },
    },
    "edit_file": {
        "fn": edit_file,
        "description": "Replace one unique occurrence of old_string with new_string in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace; must appear exactly once in the file.",
                },
                "new_string": {"type": "string", "description": "Text to replace it with."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}

# The list sent to the API on every request (OpenAI-style function-tool format).
SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }
    for name, tool in TOOLS.items()
]


def run(name: str, args: dict) -> str:
    """Execute a tool. Errors are returned as text so the model can react to
    them instead of crashing the harness."""
    try:
        output = TOOLS[name]["fn"](**args)
    except Exception as e:
        output = f"Error: {type(e).__name__}: {e}"
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} chars total)"
    return output
