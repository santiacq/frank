"""Frank: a minimal coding agent harness. Entry point, REPL, and confirmation UI."""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openrouter import OpenRouter

from agent import Agent

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"

BOLD = "\033[1m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def confirm(name: str, args: dict) -> bool:
    """Shown before every tool call. Default is deny."""
    pretty = json.dumps(args, indent=2)
    if len(pretty) > 1000:
        pretty = pretty[:1000] + "\n... (truncated)"
    print(f"\n{YELLOW}{BOLD}tool: {name}{RESET}\n{pretty}")
    answer = input(f"{YELLOW}allow? [y/N]{RESET} ")
    return answer.strip().lower() in ("y", "yes")


def main():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")

    parser = argparse.ArgumentParser(description="Frank, a minimal coding agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Any OpenRouter model slug.")
    parser.add_argument("--debug", action="store_true", help="Print raw model requests/responses.")
    cli_args = parser.parse_args()

    client = OpenRouter(api_key=api_key)
    agent = Agent(client, cli_args.model, confirm, debug=cli_args.debug)

    print(f"Frank -- model: {agent.model} -- /model <slug> to switch, /debug to toggle, /quit to exit")

    while True:
        try:
            user_input = input(f"{BOLD}frank @ {agent.model} >{RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/debug":
            agent.debug = not agent.debug
            print(f"debug {'on' if agent.debug else 'off'}")
            continue
        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                agent.model = parts[1]
            print(f"model: {agent.model}")
            continue

        try:
            agent.run(user_input)
        except Exception as e:
            # API errors (bad model slug, network, etc.) shouldn't kill the REPL.
            print(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
