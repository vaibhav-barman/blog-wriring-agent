"""Blog writing agent.

Given a topic, the agent plans a blog post, optionally researches the web,
writes every section in parallel and saves the finished Markdown file.

This file holds all the agent logic and the command line interface. It is
organised in clearly marked sections, in this order:

    1. Config        2. Quota guard    3. Schemas    4. Prompts
    5. Nodes         6. Graph          7. CLI
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

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

# Free-tier limits for this model: 5 requests per minute and 20 per day.
DAILY_LIMIT = 20  # requests per day the free tier allows
DAILY_STOP_AT = 18  # we stop here and keep 2 requests as a safety margin
REQUESTS_PER_MINUTE = 4  # safely below the 5 per minute limit
MAX_BACKOFF_RETRIES = 3  # retries after a 429 or 503 error
BACKOFF_BASE_SECONDS = 15  # waits 15s, 30s, 60s (doubling each time)
MAX_WAIT_SECONDS = 90  # never wait longer than this for one retry

# Local runtime state (both are gitignored).
CACHE_DIR = ROOT / ".cache"
USAGE_FILE = ROOT / ".usage.json"

# Environment variables that hold secrets. Their values are never printed.
SECRET_ENV_VARS = ("GEMINI_API_KEY", "TAVILY_API_KEY")


class AgentError(Exception):
    """An error with a message that is safe and useful to show to the user."""


class MissingKeyError(AgentError):
    """A required API key is not set."""


class DailyQuotaError(AgentError):
    """The daily request budget is used up."""


class LLMCallError(AgentError):
    """A model call failed for a reason we cannot fix by retrying."""


class StructuredOutputError(AgentError):
    """The model returned invalid structured output, even after one retry."""


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


# =============================================================================
# 2. QUOTA GUARD  (rate limiter, daily counter, disk cache, 429 backoff)
# =============================================================================
# The free tier allows 5 requests per minute and only 20 per day, so every model
# call goes through `call_llm` below. In order, it:
#   1. returns a cached answer if the same call was made before (free),
#   2. reserves one request from the persistent daily counter (stops at 18),
#   3. waits for the rate limiter attached to the model (4 per minute),
#   4. retries with exponential backoff on 429 / 503 errors,
#   5. retries once if the model returns invalid structured output.

# One limiter shared by every model instance. A bucket of 1 means no bursts:
# calls are spaced 15 seconds apart, which keeps us under 5 per minute.
RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=REQUESTS_PER_MINUTE / 60,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def build_llm(node: str) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model for one node ('router', 'planner' or 'writer')."""
    settings = {
        "model": MODEL_NAME,
        # The library looks for GOOGLE_API_KEY by default, so pass ours explicitly.
        "google_api_key": gemini_key(),
        "thinking_level": THINKING_LEVEL[node],
        "rate_limiter": RATE_LIMITER,
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


@lru_cache(maxsize=None)
def get_llm(node: str) -> ChatGoogleGenerativeAI:
    """Return the model for a node, building it only once."""
    return build_llm(node)


# ---- persistent daily counter ------------------------------------------------

_usage_lock = threading.Lock()  # section workers run in parallel threads


def _read_usage() -> dict[str, int]:
    """Read the {date: request count} mapping from disk (empty if missing or broken)."""
    try:
        data = json.loads(USAGE_FILE.read_text())
        return {str(day): int(count) for day, count in data.items()}
    except (OSError, ValueError, AttributeError):
        return {}


def requests_used_today() -> int:
    """How many requests we have sent today."""
    return _read_usage().get(date.today().isoformat(), 0)


def usage_summary() -> str:
    """One line for the user, e.g. '7 of 20 requests used today'."""
    return f"{requests_used_today()} of {DAILY_LIMIT} requests used today"


def reserve_request() -> int:
    """Count one request against today's budget and return the new total.

    The check and the increment happen under one lock, so parallel workers can
    never push us past DAILY_STOP_AT. Failed attempts are counted too, which
    errs on the safe side.
    """
    with _usage_lock:
        usage = _read_usage()
        today = date.today().isoformat()
        used = usage.get(today, 0)
        if used >= DAILY_STOP_AT:
            raise DailyQuotaError(
                f"Daily request limit reached: {used} of {DAILY_LIMIT} requests used today "
                f"(stopping at {DAILY_STOP_AT} to keep a safety margin). Try again tomorrow. "
                "Cached results are still free, so re-running a finished topic costs nothing."
            )
        usage[today] = used + 1
        recent = dict(sorted(usage.items())[-7:])  # keep the last 7 days only
        tmp_file = USAGE_FILE.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(recent, indent=2))
        tmp_file.replace(USAGE_FILE)
        return recent[today]


# ---- disk cache --------------------------------------------------------------


def cache_key(node: str, messages: list[BaseMessage], schema: type[BaseModel] | None) -> str:
    """Hash of everything that can change the answer: model, settings, schema, prompt."""
    payload = {
        "model": MODEL_NAME,
        "thinking": THINKING_LEVEL[node],
        "temperature": TEMPERATURE[node],
        "schema": schema.model_json_schema() if schema else None,
        "messages": [[message.type, message.content] for message in messages],
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def cache_get(key: str) -> object | None:
    """Return the cached value (a JSON-compatible object) or None."""
    try:
        return json.loads((CACHE_DIR / f"{key}.json").read_text())["value"]
    except (OSError, ValueError, KeyError):
        return None


def cache_put(key: str, value: object) -> None:
    """Store a JSON-compatible value under `key`."""
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps({"model": MODEL_NAME, "value": value}))


# ---- errors and backoff ------------------------------------------------------


def error_text(err: BaseException) -> str:
    """The message of an exception plus the messages of everything that caused it."""
    parts = []
    while err is not None and len(parts) < 5:
        parts.append(str(err))
        err = err.__cause__
    return " | ".join(parts)


def classify_error(err: BaseException) -> str:
    """Sort an API error into 'daily', 'rate_limit', 'unavailable' or 'other'."""
    text = error_text(err)
    lower = text.lower()
    if "429" in text or "resource_exhausted" in lower:
        # The quota id in a 429 body names the window, e.g. ...PerDay...
        if "perday" in lower or "per day" in lower:
            return "daily"
        return "rate_limit"
    transient_status = re.search(r"\b(500|502|503|504)\b", text)
    if transient_status or "unavailable" in lower or "timed out" in lower or "timeout" in lower:
        return "unavailable"
    return "other"


def backoff_seconds(err: BaseException, retry: int) -> float:
    """Exponential backoff, or the delay the API asked for if that is longer."""
    wait = BACKOFF_BASE_SECONDS * 2**retry
    hint = re.search(r"retry in ([\d.]+)s", error_text(err), re.IGNORECASE)
    if hint:
        wait = max(wait, float(hint.group(1)) + 1)
    return min(wait, MAX_WAIT_SECONDS)


def invoke_with_backoff(runnable, messages: list[BaseMessage]):
    """Invoke a model (or a model + parser chain) once, retrying on 429 / 503 errors."""
    for retry in range(MAX_BACKOFF_RETRIES + 1):
        reserve_request()  # raises DailyQuotaError at the stop line
        try:
            return runnable.invoke(messages)
        except (OutputParserException, ValidationError):
            raise  # invalid output is handled by the caller, not retried here
        except Exception as err:
            kind = classify_error(err)
            if kind == "daily":
                raise DailyQuotaError(
                    "Google says the daily quota for this model is used up. Try again tomorrow."
                ) from err
            if kind == "other":
                raise LLMCallError(f"The model call failed: {redact(str(err))[:300]}") from err
            if retry == MAX_BACKOFF_RETRIES:
                what = "rate limit" if kind == "rate_limit" else "temporary outage"
                raise LLMCallError(
                    f"The model is still hitting a {what} after {MAX_BACKOFF_RETRIES} retries. "
                    f"Wait a minute and run the same command again (finished calls are cached). "
                    f"Last error: {redact(str(err))[:200]}"
                ) from err
            wait = backoff_seconds(err, retry)
            print(
                f"  [{kind}] retry {retry + 1}/{MAX_BACKOFF_RETRIES} in {wait:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait)


# ---- the one entry point for model calls -------------------------------------


def call_llm(
    node: str,
    messages: list[BaseMessage],
    schema: type[BaseModel] | None = None,
    *,
    use_cache: bool = True,
):
    """Run one guarded model call and return a `schema` instance (or a string if no schema).

    Invalid output (a parse error, a schema violation or empty text) is retried
    once, then raises StructuredOutputError.
    """
    key = cache_key(node, messages, schema)
    if use_cache:
        cached = cache_get(key)
        if cached is not None:
            return schema.model_validate(cached) if schema else cached

    llm = get_llm(node)
    runnable = llm.with_structured_output(schema) if schema else llm
    problem = "no output"
    for _attempt in range(2):
        try:
            response = invoke_with_backoff(runnable, messages)
            if schema is None:
                result = str(response.text).strip()
                if not result:
                    problem = "empty text"
                    continue
            elif isinstance(response, schema):
                result = response
            else:
                result = schema.model_validate(response)  # a dict, or None -> error
        except (OutputParserException, ValidationError) as err:
            problem = redact(" ".join(str(err).split()))[:150]  # one line
            continue
        if use_cache:
            cache_put(key, result.model_dump(mode="json") if schema else result)
        return result
    what = schema.__name__ if schema else "text"
    raise StructuredOutputError(f"The model returned invalid {what} output twice ({problem}).")


# =============================================================================
# 7. CLI
# =============================================================================


class SmokePing(BaseModel):
    """Tiny schema used by the smoke test to check structured output."""

    ok: bool
    message: str


def smoke_test() -> None:
    """Make ONE real call to check that the key, model name and structured output work."""
    print(f"Model: {MODEL_NAME} (thinking level: {THINKING_LEVEL['router']})")
    reply = call_llm(
        "router",
        [HumanMessage("Set ok to true and write a greeting of at most five words in message.")],
        SmokePing,
        use_cache=False,  # the point is to hit the real model
    )
    print(f"Reply: ok={reply.ok}, message={reply.message!r}")
    print(f"Smoke test passed. {usage_summary()}.")


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
