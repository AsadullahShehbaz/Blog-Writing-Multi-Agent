from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import pandas as pd
import streamlit as st

# ── Import your compiled LangGraph app ──────────────────────────────────────
from graph import app   # adjust if your entry-point is different

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main title */
h1 { font-family: 'DM Serif Display', serif !important; letter-spacing: -0.5px; }

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: #0f0f14;
    border-right: 1px solid #2a2a35;
}
section[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
section[data-testid="stSidebar"] .stTextArea textarea {
    background: #1a1a24 !important;
    border: 1px solid #3a3a4a !important;
    color: #e8e8f0 !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%;
    padding: 0.6rem 1rem !important;
    font-size: 1rem !important;
    transition: opacity 0.2s;
}
section[data-testid="stSidebar"] .stButton button:hover { opacity: 0.88; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 2px solid #e8e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    background: transparent !important;
    color: #888 !important;
}
.stTabs [aria-selected="true"] {
    background: #6c63ff !important;
    color: white !important;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.metric-card {
    background: linear-gradient(135deg, #f8f7ff, #f0eeff);
    border: 1px solid #d8d0ff;
    border-radius: 12px;
    padding: 14px 20px;
    flex: 1;
    min-width: 120px;
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #3d35b5;
    font-family: 'DM Serif Display', serif;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    background: #ede9ff;
    color: #5b4fcf;
    margin: 2px;
}
.badge.green { background: #e6f9f0; color: #1a7f52; }
.badge.orange { background: #fff4e6; color: #b35c00; }
.badge.red { background: #fff0f0; color: #b32020; }

/* Section preview card */
.section-card {
    border: 1px solid #e4e4f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    background: #fafafe;
    transition: box-shadow 0.2s;
}
.section-card:hover { box-shadow: 0 4px 16px rgba(108,99,255,0.1); }
.section-card .sec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #1a1a2e;
    margin-bottom: 4px;
}
.section-card .sec-meta {
    font-size: 0.78rem;
    color: #888;
}

/* Past blog radio buttons */
div[data-testid="stRadio"] label {
    font-size: 0.84rem !important;
    padding: 4px 0 !important;
}

/* Status/info boxes */
.info-box {
    background: #f0f0ff;
    border-left: 4px solid #6c63ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #3d3580;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def list_past_blogs() -> List[Path]:
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict, payload: Any) -> Dict:
    if isinstance(payload, dict):
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            current_state.update(next(iter(payload.values())))
        else:
            current_state.update(payload)
    return current_state


def mode_badge(mode: str) -> str:
    colors = {"closed_book": "green", "hybrid": "orange", "open_book": "red"}
    labels = {"closed_book": "📚 Closed Book", "hybrid": "🔀 Hybrid", "open_book": "🌐 Open Book"}
    cls = colors.get(mode, "")
    lbl = labels.get(mode, mode)
    return f'<span class="badge {cls}">{lbl}</span>'


# ── Session state init ────────────────────────────────────────────────────────
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []

logs: List[str] = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ Blog Agent")
    st.markdown("---")

    st.markdown("**Topic**")
    topic = st.text_area(
        label="topic_input",
        label_visibility="collapsed",
        placeholder="e.g. Self-Attention in Transformers",
        height=130,
    )

    as_of = st.date_input("📅 As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.markdown("---")
    st.markdown("**📂 Past Blogs**")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs yet.")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:40]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title[:38]}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "select_blog",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load Blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "final": md_text,
                }
                st.rerun()

# ── Main title ────────────────────────────────────────────────────────────────
st.markdown("# Blog Writing Agent")
st.markdown("*Powered by LangGraph · GPT + Tavily*")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs([
    "🧩 Plan", "🔎 Evidence", "📝 Preview", "🧾 Logs"
])

# ── Run on button click ───────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("⚠️ Please enter a topic before generating.")
        st.stop()

    st.session_state["logs"] = []

    inputs: Dict[str, Any] = {
        "topic":          topic.strip(),
        "mode":           "",
        "needs_research": False,
        "queries":        [],
        "evidence":       [],
        "plan":           None,
        "as_of":          as_of.isoformat(),
        "recency_days":   7,
        "sections":       [],
        "final":          "",
    }

    status  = st.status("⚙️ Running graph…", expanded=True)
    prog    = st.empty()
    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1:
                v = next(iter(payload.values()))
                if isinstance(v, dict):
                    node_name = next(iter(payload.keys()))

            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode":           current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries":        (current_state.get("queries") or [])[:4],
                "evidence_count": len(current_state.get("evidence") or []),
                "tasks":          len((current_state.get("plan") or {}).get("tasks", []))
                                  if isinstance(current_state.get("plan"), dict) else None,
                "sections_done":  len(current_state.get("sections") or []),
            }
            prog.json(summary)
            logs.append(f"[{kind}] {json.dumps(payload, default=str)[:800]}")

        elif kind == "final":
            st.session_state["last_out"] = payload
            st.session_state["logs"].extend(logs)
            status.update(label="✅ Done!", state="complete", expanded=False)
            logs.append("[final] complete")

# ── Render result ─────────────────────────────────────────────────────────────
out = st.session_state.get("last_out")

if not out:
    for tab in [tab_plan, tab_evidence, tab_preview, tab_logs]:
        with tab:
            st.markdown('<div class="info-box">Enter a topic in the sidebar and click <strong>Generate Blog</strong> to get started.</div>', unsafe_allow_html=True)

else:
    # ── PLAN TAB ──────────────────────────────────────────────────────────────
    with tab_plan:
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan data (loaded from file).")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan = plan_obj
            else:
                plan = json.loads(json.dumps(plan_obj, default=str))

            # Header
            st.markdown(f"### {plan.get('blog_title', 'Untitled')}")

            mode_str  = out.get("mode", plan.get("mode", ""))
            kind_str  = plan.get("blog_kind", "")
            audience  = plan.get("audience", "")
            tone      = plan.get("tone", "")
            tasks     = plan.get("tasks", [])
            evidence  = out.get("evidence") or []
            n_queries = len(out.get("queries") or [])

            # Metric cards
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="label">Sections</div>
                    <div class="value">{len(tasks)}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Evidence</div>
                    <div class="value">{len(evidence)}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Queries</div>
                    <div class="value">{n_queries}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Total Words</div>
                    <div class="value">{sum(t.get('target_words',0) for t in tasks):,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mode + kind badges
            st.markdown(
                mode_badge(mode_str) +
                f'<span class="badge">{kind_str}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Audience:** {audience}  \n**Tone:** {tone}")
            st.markdown("---")

            # Section cards
            st.markdown("#### 📋 Sections")
            for t in sorted(tasks, key=lambda x: x.get("id", 0)):
                flags = []
                if t.get("requires_code"):       flags.append('<span class="badge">💻 Code</span>')
                if t.get("requires_citations"):  flags.append('<span class="badge orange">📎 Citations</span>')
                if t.get("requires_research"):   flags.append('<span class="badge orange">🔍 Research</span>')
                tags_html = "".join(f'<span class="badge green">{tg}</span>' for tg in (t.get("tags") or []))
                bullets_html = "".join(f"<li>{b}</li>" for b in t.get("bullets", []))
                st.markdown(f"""
                <div class="section-card">
                    <div class="sec-title">{t.get('id','')}.  {t.get('title','')}</div>
                    <div class="sec-meta">🎯 {t.get('goal','')}  ·  ~{t.get('target_words','')} words</div>
                    <div style="margin-top:6px">{"".join(flags)} {tags_html}</div>
                    <ul style="margin:8px 0 0 16px; font-size:0.85rem; color:#555">{bullets_html}</ul>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🗂 Raw plan JSON"):
                st.json(plan)

    # ── EVIDENCE TAB ──────────────────────────────────────────────────────────
    with tab_evidence:
        evidence = out.get("evidence") or []
        queries  = out.get("queries") or []

        if queries:
            st.markdown("#### 🔍 Search Queries")
            for i, q in enumerate(queries, 1):
                st.markdown(f"`{i}.` {q}")
            st.markdown("---")

        if not evidence:
            st.info("No evidence found — closed_book mode or no Tavily results.")
        else:
            st.markdown(f"#### 📚 {len(evidence)} Evidence Items")
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append({
                    "Title":        e.get("title", ""),
                    "Published":    e.get("published_at", ""),
                    "Source":       e.get("source", ""),
                    "URL":          e.get("url", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "URL": st.column_config.LinkColumn("URL", display_text="🔗 Open"),
                }
            )

    # ── PREVIEW TAB ───────────────────────────────────────────────────────────
    with tab_preview:
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No markdown content found.")
        else:
            # Stats bar
            word_count = len(final_md.split())
            char_count = len(final_md)
            section_count = final_md.count("\n## ")
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="label">Words</div>
                    <div class="value">{word_count:,}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Characters</div>
                    <div class="value">{char_count:,}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Sections</div>
                    <div class="value">{section_count}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])

            plan_obj   = out.get("plan")
            blog_title = (
                plan_obj.blog_title if hasattr(plan_obj, "blog_title")
                else plan_obj.get("blog_title", "blog") if isinstance(plan_obj, dict)
                else extract_title_from_md(final_md, "blog")
            )
            md_filename = f"{safe_slug(blog_title)}.md"

            with col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                    use_container_width=True,
                )

            with col2:
                # Bundle zip (md only, no images)
                buf = BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    z.writestr(md_filename, final_md.encode("utf-8"))
                st.download_button(
                    "📦 Download Bundle (ZIP)",
                    data=buf.getvalue(),
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

            st.markdown("---")

            # Rendered preview
            with st.container():
                st.markdown(final_md)

    # ── LOGS TAB ──────────────────────────────────────────────────────────────
    with tab_logs:
        all_logs = st.session_state.get("logs", [])
        if not all_logs:
            st.info("No logs yet. Run a generation first.")
        else:
            st.markdown(f"**{len(all_logs)} log entries**")
            st.text_area(
                "Event log",
                value="\n\n".join(all_logs[-60:]),
                height=500,
                label_visibility="collapsed",
            )
            if st.button("🗑 Clear logs"):
                st.session_state["logs"] = []
                st.rerun()