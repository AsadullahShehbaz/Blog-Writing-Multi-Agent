<div align="center">

<img src="https://img.shields.io/badge/LangGraph-1.0.3-FF6B35?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-Groq-00B4D8?style=for-the-badge&logo=chainlink&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.47-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Tavily-Search_API-6C63FF?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge"/>

<br/><br/>

# 🤖 AutoBlog — Multi-Agent Blog Writer
### *Autonomous • Parallel • Research-Grounded*

> A production-grade **multi-agent system** built with LangGraph that autonomously researches, plans, and writes full blog posts — with parallel section writing via the **Map-Reduce pattern**.

<br/>

**[Live Demo]([Streamlit Web App](https://autoblogagent.streamlit.app/))** · **[Architecture Deep Dive](#-architecture)** · **[Quick Start](#-quick-start)**

<br/>

</div>

---

## 📌 What Is This?

You type a topic. The system does the rest.

AutoBlog is not a simple "call GPT and get text" wrapper. It is a **five-node stateful agent graph** where each node has a distinct role, agents communicate through a shared typed state, and section-writing happens **fully in parallel** — the same pattern used in production LLM pipelines at scale.

```
Input: "Write a blog on Retrieval-Augmented Generation"
         │
         ▼
     [ ROUTER ]        → Decides: needs web search? what mode?
         │
    ┌────┴────┐
    │         │
[ RESEARCH ]  │        → Tavily web search + LLM deduplication
    │         │
    └────┬────┘
         ▼
  [ ORCHESTRATOR ]     → Plans 5–9 sections with goals & bullets
         │
   ┌─────┼─────┐
   ▼     ▼     ▼
[W1]  [W2]  [W3] ...   → All sections written IN PARALLEL
   └─────┼─────┘
         ▼
    [ REDUCER ]        → Sorts, assembles, saves final .md
         │
         ▼
  final_blog_post.md   ✅
```

**Output:** A structured, well-cited, multi-section Markdown blog post — written in seconds.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🧠 **Intelligent Routing** | Classifies every topic as `closed_book`, `hybrid`, or `open_book` before deciding whether to search |
| 🔍 **Grounded Research** | Tavily web search with LLM-powered deduplication and date-window filtering |
| 📋 **Structured Planning** | Orchestrator generates a full `Plan` with typed `Task` objects using Pydantic `with_structured_output` |
| ⚡ **Parallel Writing** | LangGraph `Send` API fans out all section writers simultaneously — **4–7× faster** than sequential |
| 🔒 **Type-Safe State** | `operator.add` reducer annotation ensures parallel workers never overwrite each other |
| 🆓 **100% Free LLMs** | Powered by **Groq** (llama3, mixtral, gemma) — zero OpenAI cost |
| 🖥️ **Clean UI** | Streamlit frontend with live progress and Markdown preview |
| 💾 **Auto-Save** | Final blog post saved as `.md` file automatically |

---

## 🏗️ Architecture

### The Five Nodes

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH STATE GRAPH                        │
│                                                                     │
│   START → [ ROUTER ] ──────────────────────────→ [ ORCHESTRATOR ]  │
│                  │                                       │          │
│                  └──→ [ RESEARCH ] ──────────────────────┘          │
│                                                          │          │
│                                          fanout() → Send API        │
│                                     ┌────────┬────────┐            │
│                                     ▼        ▼        ▼            │
│                                  [W:0]    [W:1]    [W:N]           │
│                                     └────────┴────────┘            │
│                                              │ operator.add         │
│                                              ▼                      │
│                                        [ REDUCER ] → END            │
└─────────────────────────────────────────────────────────────────────┘
```

### Node Responsibilities

**`router_node`** — The Decision Maker
- Classifies topic into 3 modes: `closed_book` / `hybrid` / `open_book`
- Sets `needs_research`, `queries[]`, and `recency_days`
- Uses conditional edges to route to Research or skip directly to Orchestrator

**`research_node`** — The Web Searcher *(optional)*
- Runs parallel Tavily queries (up to 60 raw results)
- Uses `with_structured_output(EvidencePack)` for LLM deduplication
- Applies hard date filter for `open_book` mode (stale news = hallucination risk)

**`orchestrator_node`** — The Editor-in-Chief
- Produces a validated `Plan` with 5–9 `Task` objects
- Each Task carries: `title`, `goal`, `bullets[]`, `target_words`, `requires_code`, `requires_citations`
- Forces `blog_kind = "news_roundup"` for `open_book` as a programmatic safety net

**`worker_node`** — The Section Writer *(runs N times in parallel)*
- Receives only its own `Task` payload — stateless and isolated
- Applies mode-specific citation rules
- Returns `{"sections": [(task.id, markdown)]}`

**`reducer_node`** — The Assembler
- Sorts sections by `task.id` to restore outline order
- Joins with `\n\n`, prepends `# Blog Title`
- Saves final Markdown to disk

---

## 🧬 State Design

The `State` TypedDict is the **single source of truth** flowing through every node:

```python
class State(TypedDict):
    topic:          str                                          # user input
    as_of:          str                                          # reference date
    mode:           str                                          # router → closed_book | hybrid | open_book
    needs_research: bool                                         # router decision
    queries:        List[str]                                    # search queries
    recency_days:   int                                          # freshness window
    evidence:       List[EvidenceItem]                           # research output
    plan:           Optional[Plan]                               # orchestrator output
    sections:       Annotated[List[tuple[int, str]], operator.add]  # ← parallel-safe!
    final:          str                                          # reducer output
```

> **Why `Annotated[..., operator.add]`?**
> Without it, Worker 3 would silently overwrite Worker 1's section. The annotation tells LangGraph to **concatenate** updates from all parallel nodes instead of last-write-wins.

---

## 🗂️ Project Structure

```
autoblog-multiagent-langgraph/
│
├── schemas/
│   └── __init__.py        # Pydantic models: RouterDecision, EvidenceItem,
│                          #   EvidencePack, Task, Plan, State
│
├── utils/
│   └── __init__.py        # Groq LLM client, Tavily wrapper, date helpers
│
├── nodes/
│   ├── __init__.py        # Exports all 5 node functions
│   ├── router.py          # Topic classification + routing logic
│   ├── research.py        # Tavily search + LLM dedup + date filter
│   ├── orchestrator.py    # Blog outline planning (Plan + Task[])
│   ├── worker.py          # Single-section writer (runs in parallel)
│   └── reducer.py         # Sort + assemble + save final post
│
├── graph/
│   ├── __init__.py        # Exports compiled app
│   ├── builder.py         # StateGraph wiring: nodes + edges + compile
│   └── fanout.py          # Send API: orchestrator → parallel workers
│
├── app.py                 # Streamlit UI
├── main.py                # CLI entry point: run(topic)
├── requirements.txt       # Minimal, Python 3.11 compatible
├── runtime.txt            # python-3.11 (Streamlit Cloud)
└── .env                   # GROQ_API_KEY, TAVILY_API_KEY
```

---

## ⚙️ Design Patterns Used

### 1. Map-Reduce for Parallel Writing
```python
# fanout.py — sends N workers simultaneously
def fanout(state: State) -> List[Send]:
    return [
        Send("worker", {"task": task.model_dump(), "plan": plan.model_dump(), ...})
        for task in state["plan"].tasks
    ]
```
7 sections run in ~6 seconds instead of ~35 seconds sequential.

### 2. Structured Output = Guaranteed JSON
```python
# No more json.loads() failures from markdown-wrapped responses
planner = llm.with_structured_output(Plan)
plan: Plan = planner.invoke([system_msg, human_msg])  # always a valid Plan object
```

### 3. Conditional Routing
```python
g.add_conditional_edges("router", route_next, {
    "research":     "research",
    "orchestrator": "orchestrator",
})
```
Evergreen topics skip research entirely. News topics always search first.

### 4. Programmatic Safety Net
```python
# LLMs are non-deterministic — never trust them for critical control logic
if mode == "open_book":
    plan.blog_kind = "news_roundup"  # force it, regardless of what LLM said
```

### 5. Graceful Degradation
```python
if not raw_results:
    return {"evidence": []}  # graph continues, orchestrator handles empty evidence
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- [Groq API Key](https://console.groq.com) (free)
- [Tavily API Key](https://tavily.com) (free tier available)

### 1. Clone & Install

```bash
git clone https://github.com/AsadullahShehbaz/autoblog-multiagent-langgraph.git
cd autoblog-multiagent-langgraph
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
TAVILY_API_KEY=tvly_your_key_here
```

### 3. Run

**Streamlit UI:**
```bash
streamlit run app.py
```

**CLI:**
```python
from main import run
result = run("Write a blog on Self-Attention Mechanism")
print(result["final"])
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push repo to GitHub
2. Add `runtime.txt` to repo root:
   ```
   python-3.11
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io) → New App
4. Add secrets in **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   TAVILY_API_KEY = "tvly_..."
   ```
5. Deploy ✅

> **Why `runtime.txt`?** Streamlit Cloud defaults to Python 3.14 which lacks compiled wheels for several LangGraph dependencies. Pinning to 3.11 ensures clean installs.

---

## 💡 Topic Examples

| Topic | Mode | Research? | Expected Output |
|---|---|---|---|
| `"What is backpropagation?"` | `closed_book` | ❌ | Conceptual tutorial, no citations |
| `"Compare top vector databases 2025"` | `hybrid` | ✅ | Comparative post with cited tool names |
| `"AI news this week"` | `open_book` | ✅ | News roundup, every claim cited |
| `"Implement a RAG pipeline in Python"` | `closed_book` | ❌ | Code-heavy tutorial |
| `"Latest LLM releases March 2026"` | `open_book` | ✅ | Dated roundup with source links |

---

## 🛠️ Customization Guide

| Goal | File to Edit |
|---|---|
| Change the LLM model (e.g., llama3 → mixtral) | `utils/__init__.py` |
| Add a new field to the blog plan | `schemas/__init__.py` → `Plan` class |
| Change how topics are classified | `nodes/router.py` → `ROUTER_SYSTEM` prompt |
| Add a new graph node | `nodes/your_node.py` + `graph/builder.py` |
| Change how sections are written | `nodes/worker.py` → `WORKER_SYSTEM` prompt |
| Change output format (e.g., HTML instead of Markdown) | `nodes/reducer.py` |

---

## 📦 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Agent Framework** | LangGraph 1.0.3 | Stateful graph execution, Send API for parallel dispatch |
| **LLM Provider** | Groq (free) | Ultra-fast inference, free tier, llama3/mixtral/gemma |
| **LLM Interface** | LangChain-Groq | `with_structured_output()` for validated Pydantic responses |
| **Web Search** | Tavily API | Clean, LLM-optimized search results with metadata |
| **Validation** | Pydantic v2 | Schema enforcement between all agent nodes |
| **UI** | Streamlit | Rapid deployment, no frontend boilerplate |
| **Python** | 3.11 | Stable wheel support for entire ML/AI dependency tree |

---

## 🧠 What I Learned Building This

This project was built during my **AI/ML Engineering learning journey** (11 months in, BSCS 8th semester). Key concepts applied from real-world agentic system design:

- **Why stateful graphs beat plain Python pipelines** — conditional routing, shared memory, parallel execution
- **The operator.add reducer trick** — solving race conditions in concurrent LLM calls without locks
- **Structured outputs are non-negotiable** — `json.loads(llm_response)` breaks in production; Pydantic schemas don't
- **Programmatic safety nets** — LLMs are stochastic; critical control paths need deterministic overrides
- **Minimal payload to workers** — stateless, isolated workers are easier to test and harder to corrupt

---

## 📄 License

MIT License — free to use, modify, and deploy.

---

<div align="center">

**Built with ❤️ by [Asadullah Shehbaz](https://github.com/AsadullahShehbaz)**

*8th Semester BSCS · Kaggle Notebook Master · Dataset Grandmaster*

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/asadullahcreative)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat-square&logo=kaggle)](https://kaggle.com/your-username)

*If this project helped you understand multi-agent systems, consider leaving a ⭐*

</div>
