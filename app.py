"""Streamlit UI for the blog writing agent.

Run it with:  uv run streamlit run app.py

All the agent logic lives in blog_agent.py. This file is only the interface:
a topic form, live progress, the plan, the finished post and a list of old posts.
"""

import re
from pathlib import Path

import streamlit as st

import blog_agent as agent

st.set_page_config(page_title="Blog Writing Agent", page_icon="✍️", layout="wide")

NO_SELECTION = "(none)"
# A Markdown image on its own line: ![alt](path)
IMAGE_LINE = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)\)\s*$", re.MULTILINE)


@st.cache_resource
def get_graph():
    """Compile the LangGraph workflow once and reuse it on every rerun."""
    return agent.build_graph()


def show_usage(placeholder) -> None:
    """Fill a placeholder with today's request counter (the 'N of 20' line)."""
    used = agent.requests_used_today()
    with placeholder.container():
        st.progress(min(used / agent.DAILY_LIMIT, 1.0))
        st.caption(agent.usage_summary())


def render_markdown(markdown: str) -> None:
    """Render a post. Local images are shown with st.image, because Streamlit's
    Markdown cannot resolve the relative paths (../images/...) saved in the file."""
    position = 0
    for match in IMAGE_LINE.finditer(markdown):
        st.markdown(markdown[position : match.start()])
        path = (agent.BLOGS_DIR / match.group(2)).resolve()
        if path.is_file() and path.is_relative_to(agent.ROOT):
            st.image(str(path), caption=match.group(1))
        position = match.end()
    st.markdown(markdown[position:])


def show_plan(plan: agent.Plan) -> None:
    """The plan, inside an expander."""
    with st.expander("Plan", expanded=False):
        st.markdown(f"**{plan.blog_title}**  \nAudience: {plan.audience} · Tone: {plan.tone}")
        for task in plan.tasks:
            evidence = " · uses research" if task.needs_evidence else ""
            st.markdown(f"{task.id}. **{task.title}** (~{task.target_words} words{evidence})  \n{task.brief}")


def show_post(markdown: str, file_name: str, plan: agent.Plan | None = None) -> None:
    """The plan (if known), a download button and the post itself."""
    if plan:
        show_plan(plan)
    st.download_button("Download Markdown", markdown, file_name=file_name, mime="text/markdown")
    st.divider()
    render_markdown(markdown)


def describe(node: str, update: dict) -> str | None:
    """One line of progress for a finished node (None for nodes that say nothing)."""
    if node == "router":
        queries = update["queries"]
        searching = f", searching for: {', '.join(queries)}" if queries else ""
        return f"Router chose **{update['mode']}**{searching}"
    if node == "research":
        return f"Research kept {len(update['evidence'])} evidence chunks from {len(update['raw_results'])} results"
    if node == "orchestrator":
        return f"Planner made a plan with {len(update['plan'].tasks)} sections"
    if node == "images" and update:
        return f"Images ready: {len(update['images'])}"
    if node == "reducer":
        return f"Post assembled and saved to `{Path(update['file_path']).name}`"
    return None


def run_agent(topic: str, research: bool, images: bool) -> dict | None:
    """Run the graph with live progress. Returns the result, or None if it failed."""
    graph = get_graph()
    state = agent.initial_state(topic, research=research, images=images)
    final = dict(state)
    sections_done = 0
    with st.status("Writing your blog...", expanded=True) as status:
        try:
            for chunk in graph.stream(state, stream_mode="updates"):  # one dict per finished node
                for node, update in chunk.items():
                    update = update or {}
                    agent.apply_update(final, update)
                    if node == "worker":
                        sections_done += 1
                        total = len(final["plan"].tasks)
                        st.write(f"Section {sections_done} of {total} written")
                    elif line := describe(node, update):
                        st.write(line)
        except agent.AgentError as err:
            status.update(label="Stopped", state="error")
            st.error(agent.redact(str(err)))
            return None
        except Exception as err:  # keep the app alive on anything unexpected
            status.update(label="Stopped", state="error")
            st.error(f"Unexpected error: {agent.redact(str(err))[:300]}")
            return None
        status.update(label="Done", state="complete", expanded=False)
    return {
        "markdown": final["final_markdown"],
        "file_name": Path(final["file_path"]).name,
        "plan": final["plan"],
    }


# ---- sidebar ---------------------------------------------------------------------

# A finished run asks for the old-post picker to reset, but a widget's value can
# only be changed before the widget is created, so the reset is applied here.
if st.session_state.pop("reset_previous", False):
    st.session_state["previous"] = NO_SELECTION

st.sidebar.header("Blog Writing Agent")
if agent.is_fake():
    st.sidebar.warning("FAKE_LLM is on: no real model calls are made.")
st.sidebar.caption(f"Model: {agent.MODEL_NAME}")
usage_box = st.sidebar.empty()
show_usage(usage_box)

old_posts = sorted(agent.BLOGS_DIR.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
previous = st.sidebar.selectbox(
    "Previous posts", [NO_SELECTION, *(path.name for path in old_posts)], key="previous"
)

# ---- main page -------------------------------------------------------------------

st.title("Blog Writing Agent")
st.write("Give it a topic. It plans the post, writes every section in parallel and saves the Markdown.")

with st.form("topic_form"):
    topic = st.text_input("Topic", placeholder="e.g. How vector databases index data")
    research = st.toggle(
        "Use web research",
        disabled=not agent.research_available(),
        help="Searches the web and cites its sources. Needs TAVILY_API_KEY in .env.",
    )
    images = st.toggle("Generate images", help="Uses extra requests from the daily quota.")
    submitted = st.form_submit_button("Write the blog", type="primary")
if not agent.research_available():
    st.caption("Web research is off because TAVILY_API_KEY is not set in .env.")
if not agent.is_fake():
    st.caption("Real runs take a few minutes: the free tier allows only 4 requests per minute.")

if submitted:
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        result = run_agent(topic.strip(), research, images)
        show_usage(usage_box)  # refresh the sidebar counter after the run
        if result:
            st.session_state["result"] = result
            st.session_state["reset_previous"] = True
            show_post(result["markdown"], result["file_name"], result["plan"])
elif previous != NO_SELECTION:
    path = agent.BLOGS_DIR / previous
    if path.is_file():
        show_post(path.read_text(encoding="utf-8"), path.name)
elif "result" in st.session_state:
    saved = st.session_state["result"]
    show_post(saved["markdown"], saved["file_name"], saved["plan"])
