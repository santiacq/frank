# Frank

A minimal coding agent harness that runs in your terminal, built on
[OpenRouter](https://openrouter.ai). It inspects and edits code, runs shell
commands, and keeps a running tally of tokens and cost — all through a simple
REPL with a deny-by-default confirmation prompt before every tool call.

## Install

```sh
uv sync
```

## Setup

Copy the example env file and add your key:

```sh
cp .env.example .env
```

Then set `OPENROUTER_API_KEY` in `.env`.

## Run

```sh
uv run python main.py
```

### REPL commands

| Command | Description |
| --- | --- |
| `/model <slug>` | Switch models (default: `deepseek/deepseek-v4-flash-0731`) |
| `/stats` | Show session tokens, cost, and API request count |
| `/debug` | Toggle debug output |
| `/quit` | Exit |

Type anything else to send it to the model. Every tool call asks for
confirmation before running; deny to skip.

## How it works

The agent sends the whole conversation to the model. If it replies with tool
calls, they're confirmed and executed, the results are appended, and the loop
repeats. When the model replies with plain text, the turn ends and control
returns to the REPL.
