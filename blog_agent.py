"""Blog writing agent.

Given a topic, the agent plans a blog post, optionally researches the web,
writes every section in parallel and saves the finished Markdown file.

This file holds all the agent logic and the command line interface. It is
organised in clearly marked sections, in this order:

    1. Config        2. Quota guard    3. Schemas    4. Prompts
    5. Nodes         6. Graph          7. CLI
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

# =============================================================================
# 1. CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# One constant for the model, overridable with the GEMINI_MODEL env var.
DEFAULT_MODEL = "gemini-4-hydrogen"
MODEL_NAME = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

# Settings per node. Gemini 4 models use fixed sampling (they ignore
# `temperature`), so the thinking level is the knob that matters for them: low
# for the cheap routing step, medium for planning, low for writing prose.
# Models that do honour `temperature` (older ones, via GEMINI_MODEL) get these
# values too.
THINKING_LEVEL = {"router": "low", "planner": "medium", "writer": "low"}
TEMPERATURE = {"router": 0.0, "planner": 0.2, "writer": 0.7}
REQUEST_TIMEOUT_SECONDS = 120

# Environment variables that hold secrets. Their values are never printed.
SECRET_ENV_VARS = ("GEMINI_API_KEY", "TAVILY_API_KEY")


class AgentError(Exception):
    """An error with a message that is safe and useful to show to the user."""


class MissingKeyError(AgentError):
    """A required API key is not set."""


def redact(text: str) -> str:
    """Replace any secret value found in `text` with ***."""
    for name in SECRET_ENV_VARS:
        value = os.getenv(name)
        if value:
            text = text.replace(value, "***")
    return text


def gemini_key() -> str:
    """Return the Gemini API key, or raise a readable error if it is missing."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise MissingKeyError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def build_llm(node: str) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model for one node ('router', 'planner' or 'writer')."""
    settings = {
        "model": MODEL_NAME,
        # The library looks for GOOGLE_API_KEY by default, so pass ours explicitly.
        "google_api_key": gemini_key(),
        "thinking_level": THINKING_LEVEL[node],
        # 1 means a single attempt. The library default retries silently,
        # which would burn the tiny daily quota behind our back.
        "max_retries": 1,
        "timeout": REQUEST_TIMEOUT_SECONDS,
    }
    llm = ChatGoogleGenerativeAI(**settings)
    # Only send `temperature` to models that honour it.
    if (llm.profile or {}).get("temperature", True):
        llm = ChatGoogleGenerativeAI(**settings, temperature=TEMPERATURE[node])
    return llm


class SmokePing(BaseModel):
    """Tiny schema used by the smoke test to check structured output."""

    ok: bool
    message: str


def smoke_test() -> None:
    """Make ONE real call to check that the key, model name and structured output work."""
    llm = build_llm("router")
    print(f"Model: {MODEL_NAME} (thinking level: {llm.thinking_level})")
    reply = llm.with_structured_output(SmokePing).invoke(
        "Set ok to true and write a greeting of at most five words in message."
    )
    print(f"Reply: ok={reply.ok}, message={reply.message!r}")
    print("Smoke test passed.")


# =============================================================================
# 7. CLI
# =============================================================================


def main() -> int:
    """Command line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Blog writing agent")
    parser.add_argument("--smoke", action="store_true", help="make one real call to verify setup")
    args = parser.parse_args()
    try:
        if args.smoke:
            smoke_test()
            return 0
        parser.print_help()
        return 0
    except AgentError as err:
        print(f"Error: {redact(str(err))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
