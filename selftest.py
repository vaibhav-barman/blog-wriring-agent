"""Self-test for the blog writing agent. Run it with:  uv run selftest.py

1. The quota guard is checked against a scripted stand-in for Gemini: cache,
   backoff, the daily stop, parallel workers and the outage breaker.
2. Three topics go through the whole graph using the fake LLM and a stubbed web
   search, and the saved Markdown files are checked.

There is no network access and no quota use: your real blogs/ and images/ folders,
your cache and your usage counter are left alone (everything goes to a temporary
folder).
"""

import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

# Switch the fake LLM on before the agent runs. Research only turns on when a
# Tavily key exists, so set a placeholder; the search itself is stubbed below.
os.environ["FAKE_LLM"] = "1"
os.environ["TAVILY_API_KEY"] = "selftest-placeholder"

import blog_agent as agent  # noqa: E402  (must come after the environment is set)

# Each canned page talks about a different product, so the agent's duplicate
# filter does not merge them into one.
PAGES = [
    "The best vector databases differ in how they index data. Pinecone is fully managed and "
    "hides index tuning, which suits small teams that want to ship quickly without any operations work. ",
    "Choosing among the best vector databases also means weighing licences. Qdrant is written in "
    "Rust, offers payload filtering next to similarity search and can run on a single laptop. ",
    "Another of the best vector databases is Milvus, built for very large collections. It splits "
    "storage and compute, so a cluster can grow to billions of embeddings across many machines. ",
]


def fake_search_results(queries: list[str], pages: int) -> list[dict]:
    """Canned search results: `pages` pages from different websites, for the first query."""
    return [
        {
            "query": queries[0],
            "title": f"Source page {number}",
            "url": f"https://site{number}.example.org/article",
            "content": "",
            "raw_content": PAGES[number - 1] * 3,
            "published": "2026-10-15",
            "score": 0.9,
        }
        for number in range(1, pages + 1)
    ]


class Case:
    """One topic to run through the graph, and what we expect from it."""

    def __init__(self, name: str, topic: str, mode: str, **options) -> None:
        self.name, self.topic, self.mode = name, topic, mode
        self.pages = options.pop("pages", 0)  # how many sources the stubbed search returns
        self.images = options.pop("images", False)
        self.expect_note = options.pop("expect_note", False)


CASES = [
    # Timeless topic: the router picks closed_book, so there is no research at all.
    Case("closed_book", "How do hash maps work", "closed_book", images=True),
    # Mostly timeless, with fresh facts: research finds enough sources (3), so no warning note.
    Case("hybrid", "best vector databases", "hybrid", pages=3),
    # Time-sensitive: research finds a single source, so the post carries a thin-evidence note.
    Case("open_book", "latest vector database release news", "open_book", pages=1, expect_note=True),
]


def check(condition: bool, message: str) -> None:
    """Fail the case with a readable message."""
    if not condition:
        raise AssertionError(message)


def run_case(case: Case, workdir: Path) -> None:
    """Run one topic through the graph and check the result."""
    searches: list[str] = []  # the modes search_web was called with

    def stub_search(queries: list[str], mode: str = "hybrid") -> list[dict]:
        searches.append(mode)
        return fake_search_results(queries, case.pages)

    agent.search_web = stub_search
    agent.BLOGS_DIR = workdir / "blogs"
    agent.IMAGES_DIR = workdir / "images"

    state = agent.initial_state(case.topic, research=True, images=case.images)
    final = agent.run_blog(agent.build_graph(), state)

    # The router chose the expected mode, and research only ran when it should.
    check(final["mode"] == case.mode, f"mode is {final['mode']}, expected {case.mode}")
    if case.mode == "closed_book":
        check(not searches, "closed_book must not search the web")
    else:
        check(searches == [case.mode], f"search called with {searches}, expected [{case.mode!r}]")
        check(len(final["evidence"]) >= 1, "research produced no evidence")

    # The file exists and holds exactly the assembled post.
    path = Path(final["file_path"])
    check(path.is_file(), f"output file {path} does not exist")
    markdown = path.read_text(encoding="utf-8")
    check(markdown == final["final_markdown"], "saved file differs from the final markdown")

    # Title first, then every planned section in plan order, each with real content.
    plan = final["plan"]
    check(markdown.startswith(f"# {plan.blog_title}\n"), "the post does not start with its title")
    headings = re.findall(r"^## (.+)$", markdown, re.MULTILINE)
    expected = [task.title for task in plan.tasks]
    check(headings[: len(expected)] == expected, f"headings {headings} do not match the plan {expected}")
    for number, section in sorted(final["sections"]):
        check(len(section.split()) >= 20, f"section {number} is nearly empty")

    # Sources, citations and the thin-evidence note.
    urls_in_post = set(re.findall(r"\]\((https?://[^)\s]+)\)", markdown))
    allowed = {item.url for item in final["evidence"]}
    check(urls_in_post <= allowed, f"the post links to URLs that are not evidence: {urls_in_post - allowed}")
    if case.mode == "closed_book":
        check("## Sources" not in markdown and not urls_in_post, "closed_book post must have no links")
    else:
        check(headings[-1] == "Sources", "the post has no Sources section")
        check(urls_in_post, "no evidence was cited")
    check(("> **Note:**" in markdown) == case.expect_note, "thin-evidence note shown or hidden wrongly")

    # Images: two files, linked relatively from the post.
    if case.images:
        files = sorted((workdir / "images").glob("*.png"))
        check(len(files) == agent.MAX_IMAGES, f"expected {agent.MAX_IMAGES} images, found {len(files)}")
        check(markdown.count("![") == len(files) and "](../images/" in markdown, "image links are missing")
    else:
        check("![" not in markdown and "image placeholder" not in markdown, "unexpected image in post")

    words = len(markdown.split())
    print(f"  PASS  {case.name:12} mode={final['mode']:11} {len(plan.tasks)} sections, {words} words, {len(allowed)} evidence")


class ScriptedModel:
    """Stands in for Gemini in the guard checks. `behave(n)` returns what call number n
    answers with: a value, or an exception to raise."""

    def __init__(self, behave) -> None:
        self.behave, self.calls, self.lock = behave, 0, threading.Lock()

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        with self.lock:
            self.calls += 1
            number = self.calls
        result = self.behave(number)
        if isinstance(result, BaseException):
            raise result
        return result


def check_quota_guard() -> None:
    """Check the quota guard against a scripted model (6 checks)."""
    ok = agent.SmokePing(ok=True, message="hi")
    rate_limited = RuntimeError("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerMinute. Please retry in 3s")
    overloaded = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
    today = agent.date.today().isoformat()
    sleeps: list[float] = []
    saved = (agent.USAGE_FILE, agent.CACHE_DIR, agent.get_llm, agent.OUTAGE_COOLDOWN_SECONDS, time.sleep)
    saved_fake = os.environ["FAKE_LLM"]

    def call(prompt: str):
        return agent.call_llm("router", [HumanMessage(prompt)], agent.SmokePing)

    def use(model: ScriptedModel, name: str, already_used: int = 0) -> ScriptedModel:
        """Install a scripted model with a fresh cache folder, counter and breaker."""
        agent.get_llm = lambda node: model
        agent.CACHE_DIR = Path(tmp) / name
        agent.USAGE_FILE.write_text(f'{{"{today}": {already_used}}}')
        agent.note_outage(failed=False)
        sleeps.clear()
        return model

    def run_threads(count: int) -> list:
        """Call the guard from `count` threads at once and collect result or exception."""
        outcomes: list = []

        def work(number: int) -> None:
            try:
                outcomes.append(call(f"parallel {number}"))
            except agent.AgentError as err:
                outcomes.append(err)

        threads = [threading.Thread(target=work, args=(number,)) for number in range(count)]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        return outcomes

    os.environ["FAKE_LLM"] = "0"  # exercise the real code path, with a scripted model
    time.sleep = lambda seconds: sleeps.append(seconds)  # backoff must not really wait
    try:
        with tempfile.TemporaryDirectory() as tmp:
            agent.USAGE_FILE = Path(tmp) / "usage.json"

            model = use(ScriptedModel(lambda n: ok), "cache")
            check(call("same") == call("same") and model.calls == 1, "cache: a repeated call must be free")
            check(agent.requests_used_today() == 1, "cache: only one request should be counted")

            model = use(ScriptedModel(lambda n: rate_limited if n == 1 else ok), "backoff")
            check(call("x") == ok and model.calls == 2 and sleeps == [15], f"backoff: calls={model.calls} sleeps={sleeps}")

            model = use(ScriptedModel(lambda n: ok), "stop", already_used=17)
            call("first")
            try:
                call("second")
                check(False, "stop: the 19th request must be blocked")
            except agent.DailyQuotaError:
                check(model.calls == 1 and agent.requests_used_today() == 18, "stop: must stop at 18 of 20")

            model = use(ScriptedModel(lambda n: ok), "parallel", already_used=12)
            passed = [outcome for outcome in run_threads(10) if outcome == ok]
            check(len(passed) == 6 and agent.requests_used_today() == 18, f"parallel: {len(passed)} calls got through from 12 used, expected 6")

            model = use(ScriptedModel(lambda n: overloaded), "outage")
            outcomes = run_threads(5)
            check(all(isinstance(o, agent.LLMCallError) and "overloaded" in str(o) for o in outcomes), "outage: every worker must fail with the readable message")
            worst_case = 5 * (agent.MAX_BACKOFF_RETRIES + 1)
            check(model.calls <= agent.OUTAGE_STOP_AFTER + 5 and model.calls < worst_case, f"outage: {model.calls} requests sent (without a breaker: {worst_case})")
            agent.OUTAGE_COOLDOWN_SECONDS = 0  # the cooldown has passed and the model has recovered
            model.behave = lambda n: ok
            check(call("after the outage") == ok, "outage: calls must work again after the cooldown")

            model = use(ScriptedModel(lambda n: None), "invalid")
            try:
                call("bad output")
                check(False, "invalid: bad output must raise")
            except agent.StructuredOutputError:
                check(model.calls == 2, "invalid: must retry exactly once")
    finally:
        agent.USAGE_FILE, agent.CACHE_DIR, agent.get_llm, agent.OUTAGE_COOLDOWN_SECONDS, time.sleep = saved
        os.environ["FAKE_LLM"] = saved_fake
        agent.note_outage(failed=False)
    print("  PASS  quota guard  cache, backoff, 18-of-20 stop, parallel workers, outage breaker, invalid output")


def main() -> int:
    """Run all checks. Returns 0 if they all pass."""
    usage_before = agent.USAGE_FILE.read_bytes() if agent.USAGE_FILE.exists() else None
    failures = 0
    print("Running the blog agent self-test:")
    try:
        check_quota_guard()
    except Exception as err:
        failures += 1
        print(f"  FAIL  quota guard  {type(err).__name__}: {err}")
    for case in CASES:
        with tempfile.TemporaryDirectory() as workdir:
            try:
                run_case(case, Path(workdir))
            except Exception as err:  # report every failing case, not just the first
                failures += 1
                print(f"  FAIL  {case.name:12} {type(err).__name__}: {err}")

    usage_after = agent.USAGE_FILE.read_bytes() if agent.USAGE_FILE.exists() else None
    if usage_before != usage_after:
        failures += 1
        print("  FAIL  the self-test changed the real usage counter")

    print("\nAll checks passed." if failures == 0 else f"\n{failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
