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
import operator
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field, ValidationError

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

# Where finished posts are saved (gitignored except for .gitkeep).
BLOGS_DIR = ROOT / "blogs"

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
# 2. QUOTA GUARD  (rate limiter, daily counter, disk cache, 429 backoff, fake LLM)
# =============================================================================
# The free tier allows 5 requests per minute and only 20 per day, so every model
# call goes through `call_llm` below. In order, it:
#   1. returns a cached answer if the same call was made before (free),
#   2. reserves one request from the persistent daily counter (stops at 18),
#   3. waits for the rate limiter attached to the model (4 per minute),
#   4. retries with exponential backoff on 429 / 503 errors,
#   5. retries once if the model returns invalid structured output.
# With FAKE_LLM=1 the model is replaced by a deterministic stub, and steps 1-4
# are skipped: no network, no quota, no cache. Use it to develop and test.

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
def real_llm(node: str) -> ChatGoogleGenerativeAI:
    """Build the real model for a node only once."""
    return build_llm(node)


def get_llm(node: str):
    """Return the fake model when FAKE_LLM=1, otherwise the real Gemini model."""
    return FakeLLM(node) if is_fake() else real_llm(node)


# ---- fake LLM ----------------------------------------------------------------

# Structured outputs of the fake model, keyed by schema class name. Each builder
# takes the prompt text and returns a valid instance. Later sections register
# their own builders next to the schemas they belong to.
FAKE_STRUCTURED: dict[str, Callable[[str], BaseModel]] = {}


def fake_output(schema_name: str):
    """Decorator that registers a fake structured-output builder for a schema."""

    def register(builder: Callable[[str], BaseModel]) -> Callable[[str], BaseModel]:
        FAKE_STRUCTURED[schema_name] = builder
        return builder

    return register


def is_fake() -> bool:
    """True when FAKE_LLM=1. Read on every call so tests can switch it on."""
    return os.getenv("FAKE_LLM") == "1"


def prompt_text(messages: list[BaseMessage]) -> str:
    """Join all message contents into one string."""
    return "\n".join(str(message.content) for message in messages)


def labelled(prompt: str, label: str, default: str = "") -> str:
    """Value of a 'LABEL: value' line in a prompt (this is how the fake LLM reads prompts)."""
    match = re.search(rf"^{label}: (.+)$", prompt, re.MULTILINE)
    return match.group(1).strip() if match else default


def fake_text(prompt: str) -> str:
    """Plain-text answer of the fake model: a short section written from the prompt labels."""
    title = labelled(prompt, "SECTION TITLE", "Fake section")
    topic = labelled(prompt, "TOPIC", "the topic")
    brief = labelled(prompt, "SECTION BRIEF")
    return (
        f"## {title}\n\n"
        f"This section of the post about {topic} is placeholder text from the fake LLM. {brief}\n\n"
        "A second short paragraph keeps the section looking like real prose."
    )


class FakeLLM:
    """Deterministic stand-in for the Gemini model (no network, no quota)."""

    def __init__(self, node: str) -> None:
        self.node = node

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Answer with plain text, like a chat model."""
        return AIMessage(content=fake_text(prompt_text(messages)))

    def with_structured_output(self, schema: type[BaseModel]) -> RunnableLambda:
        """Return a runnable that builds a valid `schema` instance."""
        builder = FAKE_STRUCTURED.get(schema.__name__)
        if builder is None:
            raise AgentError(f"The fake LLM has no builder for {schema.__name__}.")
        return RunnableLambda(lambda messages: builder(prompt_text(messages)))


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
    line = f"{requests_used_today()} of {DAILY_LIMIT} requests used today"
    return f"{line} (fake LLM: nothing was counted)" if is_fake() else line


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
    if is_fake():
        return runnable.invoke(messages)  # no quota, no limiter, nothing to retry
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
    use_cache = use_cache and not is_fake()  # fake answers must never pollute the cache
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
# 3. SCHEMAS AND GRAPH STATE
# =============================================================================

# How much the post depends on fresh facts from the web.
Mode = Literal["closed_book", "hybrid", "open_book"]


class Task(BaseModel):
    """One section of the blog post."""

    id: int = Field(description="1-based position of the section in the post")
    title: str = Field(description="Specific, descriptive section title")
    brief: str = Field(
        description="2-3 sentences: what this section covers and what it leaves to other sections"
    )
    target_words: int = Field(ge=50, le=800, description="Target length of the section in words")
    needs_evidence: bool = Field(
        description="True only if this section relies on the supplied research evidence"
    )
    image_prompt: str | None = Field(
        description="Prompt for one illustration for this section, or null for no image"
    )


class Plan(BaseModel):
    """The plan for the whole blog post."""

    blog_title: str = Field(description="Title of the post")
    audience: str = Field(description="Who the post is written for")
    tone: str = Field(description="Voice and tone to use in every section")
    tasks: list[Task] = Field(min_length=4, max_length=6, description="The sections, in order")


class RouterDecision(BaseModel):
    """The router's decision about web research."""

    mode: Mode = Field(description="closed_book, hybrid or open_book")
    queries: list[str] = Field(
        max_length=4, description="2-4 short web search queries, or an empty list for closed_book"
    )
    reason: str = Field(description="One sentence explaining the decision")


class EvidenceItem(BaseModel):
    """One piece of research evidence (a chunk of a web page)."""

    title: str
    url: str
    snippet: str
    published: str | None = None


class WorkerInput(TypedDict):
    """What each parallel worker receives (sent with `Send`, not part of the shared state)."""

    topic: str
    plan: Plan
    task: Task
    use_cache: bool


class State(TypedDict, total=False):
    """The graph state. Nodes return partial updates that LangGraph merges in."""

    topic: str
    mode: Mode
    queries: list[str]
    raw_results: list[dict]  # raw web search results
    chunks: list[dict]  # text chunks cut from those results
    evidence: list[EvidenceItem]  # the chunks we kept
    plan: Plan
    # Parallel workers each append one (task_id, markdown) pair; operator.add merges them.
    sections: Annotated[list[tuple[int, str]], operator.add]
    final_markdown: str
    file_path: str
    # Options for this run.
    research_enabled: bool
    images_enabled: bool
    use_cache: bool
    plan_only: bool  # stop after planning (saves requests)


# =============================================================================
# 4. PROMPTS
# =============================================================================
# NOTE: the fake LLM reads the labelled lines ("TOPIC:", "SECTION TITLE:", ...)
# in the human messages, so keep those labels when you edit the wording.

# Shared by the planner and the writers so the whole post sounds like one author.
STYLE_GUIDE = """Style guide for the whole post:
- Clear structure: short paragraphs. Use bullet points or numbered steps only where they make the content easier to scan.
- Concrete examples: prefer a specific example, number, short code snippet or scenario over an abstract statement.
- No filler and no marketing language. Never write phrases like "in today's fast-paced world", "game-changer", "unlock the power", "delve into", "revolutionize" or "it is important to note".
- Write for the audience and in the tone set by the plan. Plain Markdown only, no HTML.
- Make a point once, then move on. No padding."""

ROUTER_SYSTEM = """You decide whether a blog post needs fresh facts from the web before it is planned.

Choose exactly one mode:
- closed_book: the topic is timeless (concepts, how-to guides, fundamentals). Your own knowledge is enough.
- hybrid: the topic is mostly timeless, but a few facts may be recent or worth checking (tools, versions, benchmarks, adoption numbers).
- open_book: the topic is time-sensitive and relies on current facts (news, releases, prices, rankings, "latest", recent events).

For hybrid and open_book, return 2 to 4 short web search queries (3 to 8 words each) that would find the facts the post needs.
For closed_book, return an empty list of queries.
Add a one-sentence reason."""


def router_messages(topic: str) -> list[BaseMessage]:
    """Prompt for the router."""
    human = f"TOPIC: {topic}\nCURRENT YEAR: {date.today().year}\n\nChoose the mode and the queries."
    return [SystemMessage(ROUTER_SYSTEM), HumanMessage(human)]


PLANNER_SYSTEM = f"""You are an expert blog editor. You turn a topic into a plan for a blog post.
Several writers will each write one section without seeing each other's work, so the plan must keep them apart.

{STYLE_GUIDE}

Planning rules:
- Plan 4 or 5 sections (never more than 6): the introduction first, the conclusion last, and 2 or 3 body sections between them.
- Word targets: introduction 100-180, each body section 250-400, conclusion 100-180. The total must land between 1000 and 1800 words.
- Section titles are specific and describe the content. Do not use a bare "Introduction" or "Conclusion".
- No overlapping sections. Each brief states in 2 or 3 sentences what the section covers and names one thing it leaves to another section.
- Order the sections so that each one builds on the previous one.
- Number the sections 1, 2, 3, ... in reading order.
- Pick a specific audience and a tone (for example "practical and friendly") that fit the topic.
- image_prompt is null for every section."""


def planner_messages(topic: str) -> list[BaseMessage]:
    """Prompt for the planner."""
    return [SystemMessage(PLANNER_SYSTEM), HumanMessage(f"TOPIC: {topic}\n\nPlan the blog post.")]


WRITER_SYSTEM = f"""You are a skilled blog writer. You write exactly ONE section of a blog post, in Markdown.

{STYLE_GUIDE}

Rules for your section:
- Write only this section. Start with the heading "## <section title>" and use "###" for any sub-headings.
- Stay inside the scope of your section brief. Other writers cover the other sections of the outline: do not repeat their content, do not introduce or summarise the whole post (unless your section is the introduction or the conclusion) and do not add a post title.
- Aim for the target word count, within about 15%.
- Output only the Markdown of the section: no preamble, no notes and no code fence around the answer."""


def worker_messages(topic: str, plan: Plan, task: Task) -> list[BaseMessage]:
    """Prompt for the worker that writes one section."""
    outline = "\n".join(f"{t.id}. {t.title}: {t.brief}" for t in plan.tasks)
    human = f"""TOPIC: {topic}
BLOG TITLE: {plan.blog_title}
AUDIENCE: {plan.audience}
TONE: {plan.tone}

FULL OUTLINE (every section, with its scope):
{outline}

YOU ARE WRITING SECTION {task.id} OF {len(plan.tasks)}.
SECTION TITLE: {task.title}
SECTION BRIEF: {task.brief}
TARGET WORDS: {task.target_words}

Write this section now. Start with the heading "## {task.title}"."""
    return [SystemMessage(WRITER_SYSTEM), HumanMessage(human)]


# =============================================================================
# 5. NODES
# =============================================================================


# Words the fake router looks for in the topic (the real router uses the model).
FAKE_OPEN_BOOK_WORDS = ("latest", "news", "this year", "release", "price", "2025", "2026")
FAKE_HYBRID_WORDS = ("best", "tools", "compare", " vs ", "trends", "state of")


@fake_output("RouterDecision")
def fake_router_decision(prompt: str) -> RouterDecision:
    """Fake router output, chosen by keywords in the topic."""
    topic = labelled(prompt, "TOPIC", "the topic")
    lowered = f" {topic.lower()} "
    if any(word in lowered for word in FAKE_OPEN_BOOK_WORDS):
        mode: Mode = "open_book"
    elif any(word in lowered for word in FAKE_HYBRID_WORDS):
        mode = "hybrid"
    else:
        mode = "closed_book"
    queries = [] if mode == "closed_book" else [topic, f"{topic} overview"]
    return RouterDecision(mode=mode, queries=queries, reason=f"Fake router chose {mode}.")


def router(state: State) -> dict:
    """Router node: decide how much the post depends on fresh web facts.

    With research disabled (no --research flag, or no Tavily key) this forces
    closed_book without calling the model, which saves one request.
    """
    if not state["research_enabled"]:
        return {"mode": "closed_book", "queries": []}
    decision = call_llm(
        "router", router_messages(state["topic"]), RouterDecision, use_cache=state["use_cache"]
    )
    if decision.mode == "closed_book":
        return {"mode": "closed_book", "queries": []}
    queries = [query.strip() for query in decision.queries if query.strip()][:4]
    return {"mode": decision.mode, "queries": queries or [state["topic"]]}


def route_after_router(state: State) -> str:
    """Conditional edge: go and research, or straight to planning."""
    return "orchestrator" if state["mode"] == "closed_book" else "research"


def research(state: State) -> dict:
    """Research node (no LLM call). The next commits fill this in: search, chunk, filter."""
    return {}


@fake_output("Plan")
def fake_plan(prompt: str) -> Plan:
    """Fake planner output: an introduction, two body sections and a conclusion."""
    topic = labelled(prompt, "TOPIC", "the topic")
    outline = [
        (f"Why {topic} matters", 150),
        (f"The core ideas behind {topic}", 300),
        (f"A practical example of {topic}", 300),
        ("Key takeaways", 150),
    ]
    tasks = [
        Task(
            id=number,
            title=title,
            brief=f"Section {number} of the post about {topic}.",
            target_words=words,
            needs_evidence=False,
            image_prompt=None,
        )
        for number, (title, words) in enumerate(outline, start=1)
    ]
    return Plan(
        blog_title=f"{topic}: A Practical Guide",
        audience="curious developers",
        tone="clear and friendly",
        tasks=tasks,
    )


def clean_plan(plan: Plan) -> Plan:
    """Safety net: put tasks in id order and renumber them 1, 2, 3, ... without gaps."""
    ordered = sorted(plan.tasks, key=lambda task: task.id)
    tasks = [task.model_copy(update={"id": number}) for number, task in enumerate(ordered, start=1)]
    return plan.model_copy(update={"tasks": tasks})


def orchestrator(state: State) -> dict:
    """Planner node: one LLM call that turns the topic into a Plan."""
    plan = call_llm(
        "planner", planner_messages(state["topic"]), Plan, use_cache=state["use_cache"]
    )
    return {"plan": clean_plan(plan)}


def fan_out(state: State) -> list[Send]:
    """Conditional edge: send one worker per task in the plan. They run in parallel."""
    if state.get("plan_only"):
        return []  # nothing to write: the run ends after the plan
    return [
        Send(
            "worker",
            {
                "topic": state["topic"],
                "plan": state["plan"],
                "task": task,
                "use_cache": state["use_cache"],
            },
        )
        for task in state["plan"].tasks
    ]


def normalize_section(title: str, text: str) -> str:
    """Make sure a section is plain Markdown that starts with '## <title>'."""
    text = text.strip()
    fenced = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", text, re.DOTALL)
    if fenced:  # the model wrapped the whole answer in a code fence
        text = fenced.group(1).strip()
    lines = text.splitlines()
    if lines and re.match(r"^#{1,6}\s", lines[0]):
        lines = lines[1:]  # drop whatever heading the model chose; the plan's title wins
    body = "\n".join(lines).strip()
    return f"## {title}\n\n{body}"


def worker(payload: WorkerInput) -> dict:
    """Worker node: one LLM call that writes one section. Returns [(task_id, markdown)]."""
    task = payload["task"]
    text = call_llm(
        "writer",
        worker_messages(payload["topic"], payload["plan"], task),
        use_cache=payload["use_cache"],
    )
    return {"sections": [(task.id, normalize_section(task.title, text))]}


def slugify(text: str, max_length: int = 60) -> str:
    """Turn a title into a safe file name, e.g. 'Hello, World!' -> 'hello-world'."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].strip("-") or "blog"


def reducer(state: State) -> dict:
    """Reducer node (no LLM call): order the sections, add the title and save the post."""
    plan = state["plan"]
    written = dict(state["sections"])
    missing = [task.title for task in plan.tasks if task.id not in written]
    if missing:
        raise AgentError(f"Sections are missing from the post: {', '.join(missing)}")
    ordered = [written[task_id] for task_id in sorted(written)]
    markdown = "\n\n".join([f"# {plan.blog_title}", *ordered]).strip() + "\n"
    BLOGS_DIR.mkdir(exist_ok=True)
    path = BLOGS_DIR / f"{slugify(plan.blog_title)}.md"
    path.write_text(markdown, encoding="utf-8")
    return {"final_markdown": markdown, "file_path": str(path)}


# =============================================================================
# 6. GRAPH
# =============================================================================


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(State)
    graph.add_node("router", router)
    graph.add_node("research", research)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("worker", worker)
    graph.add_node("reducer", reducer)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router", route_after_router, {"research": "research", "orchestrator": "orchestrator"}
    )
    graph.add_edge("research", "orchestrator")
    graph.add_conditional_edges("orchestrator", fan_out, ["worker"])
    graph.add_edge("worker", "reducer")  # runs once, after every worker has finished
    graph.add_edge("reducer", END)
    return graph.compile()


def research_available() -> bool:
    """Web research needs a Tavily key."""
    return bool(os.getenv("TAVILY_API_KEY"))


def initial_state(
    topic: str,
    *,
    research: bool = False,
    images: bool = False,
    use_cache: bool = True,
    plan_only: bool = False,
) -> State:
    """The state a run starts from. Research stays off unless a Tavily key exists."""
    return {
        "topic": topic,
        "mode": "closed_book",
        "queries": [],
        "raw_results": [],
        "chunks": [],
        "evidence": [],
        "sections": [],
        "final_markdown": "",
        "file_path": "",
        "research_enabled": research and research_available(),
        "images_enabled": images,
        "use_cache": use_cache,
        "plan_only": plan_only,
    }


def stream_blog(graph, state: State):
    """Run the graph and yield (node_name, update) each time a node finishes."""
    for chunk in graph.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            yield node, update or {}


def apply_update(state: State, update: dict) -> None:
    """Merge a node's update into `state`, the way LangGraph does (sections are appended)."""
    for key, value in update.items():
        state[key] = state.get(key, []) + value if key == "sections" else value


def run_blog(graph, state: State, on_progress: Callable[[str, dict], None] | None = None) -> State:
    """Run the graph to the end and return the final state."""
    final = dict(state)
    for node, update in stream_blog(graph, state):
        apply_update(final, update)
        if on_progress:
            on_progress(node, update)
    return final


# =============================================================================
# 7. CLI
# =============================================================================


class SmokePing(BaseModel):
    """Tiny schema used by the smoke test to check structured output."""

    ok: bool
    message: str


@fake_output("SmokePing")
def fake_smoke_ping(prompt: str) -> SmokePing:
    return SmokePing(ok=True, message="Hello from the fake LLM")


def smoke_test() -> None:
    """Make ONE real call to check that the key, model name and structured output work."""
    mode = "FAKE LLM, no real call" if is_fake() else f"thinking level: {THINKING_LEVEL['router']}"
    print(f"Model: {MODEL_NAME} ({mode})")
    reply = call_llm(
        "router",
        [HumanMessage("Set ok to true and write a greeting of at most five words in message.")],
        SmokePing,
        use_cache=False,  # the point is to hit the real model
    )
    print(f"Reply: ok={reply.ok}, message={reply.message!r}")
    print("Smoke test passed.")


def print_plan(plan: Plan) -> None:
    """Print the plan in a readable form."""
    total = sum(task.target_words for task in plan.tasks)
    print(f"\n{plan.blog_title}")
    print(f"Audience: {plan.audience} | Tone: {plan.tone} | ~{total} words\n")
    for task in plan.tasks:
        print(f"  {task.id}. {task.title} (~{task.target_words} words)")
        print(f"     {task.brief}")


def print_progress(node: str, update: dict) -> None:
    """Print one line each time a graph node finishes."""
    if node == "worker":
        task_id, markdown = update["sections"][0]
        print(f"  [worker] section {task_id} written ({len(markdown.split())} words)")
    elif node == "router":
        print(f"  [router] mode: {update['mode']}")
    else:
        print(f"  [{node}] done")


def generate(args: argparse.Namespace) -> None:
    """Run the agent for one topic and print the result."""
    if args.research and not research_available():
        print(
            "Warning: --research ignored because TAVILY_API_KEY is not set (add it to .env).",
            file=sys.stderr,
        )
    if args.images:
        print("Note: --images is accepted but not active yet.")
    state = initial_state(
        args.topic,
        research=args.research,
        images=args.images,
        use_cache=not args.no_cache,
        plan_only=args.plan_only,
    )
    final = run_blog(build_graph(), state, print_progress)
    print_plan(final["plan"])
    if args.plan_only:
        return
    words = len(final["final_markdown"].split())
    print(f"\n{len(final['sections'])} sections, {words} words.")
    print(f"Saved: {os.path.relpath(final['file_path'])}")


def main() -> int:
    """Command line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Blog writing agent: plans, researches (optional) and writes a blog post."
    )
    parser.add_argument("topic", nargs="?", help="what the blog post is about")
    parser.add_argument("--research", action="store_true", help="use web research (needs TAVILY_API_KEY)")
    parser.add_argument("--images", action="store_true", help="generate images for the post")
    parser.add_argument("--no-cache", action="store_true", help="ignore the disk cache and call the model")
    parser.add_argument("--plan-only", action="store_true", help="only plan the post and print the plan")
    parser.add_argument("--smoke", action="store_true", help="make one real call to verify the setup")
    args = parser.parse_args()
    if not args.smoke and not args.topic:
        parser.print_help()
        return 0
    try:
        smoke_test() if args.smoke else generate(args)
        code = 0
    except AgentError as err:
        print(f"\nError: {redact(str(err))}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        code = 130
    print(usage_summary())
    return code


if __name__ == "__main__":
    sys.exit(main())
