"""Frank: a minimal coding agent harness. Entry point, REPL, and confirmation UI."""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from agent import Agent
from providers import make_provider

DEFAULT_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLAMA_BASE_URL = "http://localhost:8080/v1"

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


def build_provider(override: str | None):
    """Resolve config from the environment and build the provider."""
    name = override or os.environ.get("FRANK_PROVIDER", DEFAULT_PROVIDER)
    if name == "llama":
        return make_provider(
            name=name,
            base_url=os.environ.get("LLAMA_CPP_BASE_URL", DEFAULT_LLAMA_BASE_URL),
            api_key="sk-no-key",  # local servers don't authenticate
            model=os.environ.get("LLAMA_CPP_MODEL"),
        )
    return make_provider(
        name=name,
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model=os.environ.get("OPENROUTER_MODEL"),
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Frank, a minimal coding agent.")
    parser.add_argument("--model", help="Override the provider's model.")
    parser.add_argument("--provider", help="Override FRANK_PROVIDER (openrouter | llama).")
    parser.add_argument("--debug", action="store_true", help="Print raw model requests/responses.")
    cli_args = parser.parse_args()

    provider = build_provider(cli_args.provider)
    model = cli_args.model or provider.default_model
    if not model:
        sys.exit("no model set: pass --model or set the provider's model in .env")

    agent = Agent(provider, model, confirm, debug=cli_args.debug)

    print(
        f"Frank -- model: {agent.model} -- {provider.name} -- /help to see commands"
    )

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
        if user_input == "/help":
            print(
                "/model <name> -- switch models\n"
                "/stats -- session tokens and request count\n"
                "/debug -- toggle raw request/response output\n"
                "/clear -- reset the session (keeps the model)\n"
                "/quit -- exit\n"
                "anything else -- send to the model\n"
            )
            continue
        if user_input == "/clear":
            agent.clear()
            print("conversation cleared")
            continue
        if user_input == "/debug":
            agent.debug = not agent.debug
            print(f"debug {'on' if agent.debug else 'off'}")
            continue
        if user_input == "/stats":
            print(
                f"\n{YELLOW}{BOLD}session stats{RESET}\n"
                f"  provider:     {provider.name}\n"
                f"  tokens:       {agent.tokens}\n"
                f"  api requests: {agent.calls}"
            )
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
