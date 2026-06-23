# LEARNING.md — How I Built DocuMind From Zero

This is the complete record of how I built **DocuMind**, a production-style agentic multimodal RAG system, from an empty folder to a containerized, Kubernetes-deployed, observed, evaluated, CI/CD-shipped application.

I wrote this so that **a complete beginner can reproduce the whole thing step by step** — every command, every installation, every file, and (importantly) **every error I actually hit and exactly how I fixed them**. I'm on **Windows**, using **Git-bash (MINGW64)** as my terminal and **Docker Desktop**. If you're on Mac/Linux, the commands are almost identical (path separators and `.exe` suffixes differ).

For each phase I list **which source files belong to it**, so you can match this guide to the code in the repo.

> **How to read this:** go top to bottom. Don't skip the "Errors I hit" boxes — they're the parts that actually cost me time, and you'll probably hit the same ones.

---

## Table of contents

- [Big picture](#big-picture)
- [Phase 1 — Environment setup](#phase-1)
- [Phase 2 — RAG core (embeddings + vector DB)](#phase-2)
- [Phase 3 — Multimodal (images via VLM + multimodal embeddings)](#phase-3)
- [Phase 4 — Local SLM serving (Ollama)](#phase-4)
- [Phase 5 — Cloud LLM + provider abstraction](#phase-5)
- [Phase 6 — From RAG to Agent (LangGraph)](#phase-6)
- [Phase 7 — Backend API (FastAPI)](#phase-7)
- [Phase 8 — Frontend (React)](#phase-8)
- [Phase 9 — Containerization (Docker + compose)](#phase-9)
- [Phase 10 — Kubernetes (kind)](#phase-10)
- [Phase 11 — Observability & evaluation (Langfuse)](#phase-11)
- [Phase 12 — CI/CD + security](#phase-12)
- [Final notes & cleanup](#final)

---

<a name="big-picture"></a>
## Big picture

DocuMind is an assistant that reads my documents and answers questions about them, grounded in what's actually in the documents (so it doesn't make things up). I built it in 12 phases, each adding one capability:

1. Environment → 2. RAG core → 3. Multimodal → 4. Local model → 5. Cloud model + abstraction → 6. Agent → 7. API → 8. UI → 9. Containers → 10. Kubernetes → 11. Observability → 12. CI/CD + security.

**The models and tools I chose (mid-2026, current):**
- Embeddings: **Gemini Embedding 2** (`gemini-embedding-2`) — multimodal, so images and text share one vector space.
- Vision model (VLM): **Gemini 3.5 Flash** (`gemini-3.5-flash`).
- Vector database: **Qdrant** (open source, runs in Docker).
- Local small model (SLM): **Ollama** running `llama3.1:8b` (and the tiny `qwen3.5:0.8b`).
- Cloud model (LLM): **OpenAI** `gpt-4o-mini` (cheap), or Gemini.
- Agent framework: **LangGraph**.
- Backend: **FastAPI**. Frontend: **React + Vite**.
- Containers: **Docker + docker-compose**. Orchestration: **Kubernetes via kind**.
- Observability: **Langfuse**. CI/CD: **GitHub Actions + Trivy**.

---

<a name="phase-1"></a>
## Phase 1 — Environment setup

**Goal:** a clean project skeleton and my secrets in place.

**Files created:** project folders, `.env`, `.gitignore`.

### Steps

```bash
mkdir documind && cd documind
git init
mkdir backend frontend infra docs
python -m venv .venv
source .venv/Scripts/activate        # Windows Git-bash; you'll see (.venv) in your prompt
```

I created a `.env` file in the project root for my API keys:

```dotenv
OPENAI_API_KEY=
GEMINI_API_KEY=
```

(I filled the real values in later, when each phase first needed them.)

I created `.gitignore` so secrets and junk never get committed:

```gitignore
# Secrets (the most important lines)
.env
.env.*
*.key
*.pem

# Python
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/

# Node / React
node_modules/
frontend/dist/

# Docker / data
qdrant_storage/
*.db

# OS/editor junk
.vscode/
.idea/
.DS_Store
Thumbs.db
```

I also dropped a few sample documents into `docs/` — a mix of text and images (I had `Crypto.png` and `LLM_Bench.png`), so I'd have multimodal data for Phase 3.

### Key idea I learned
Secrets can leak in **three places**: the git repo, the container image, and the running cluster. `.gitignore` handles the repo (the only one that mattered in Phase 1); the image and cluster get handled later in Phases 9 and 10.

> **Verify:** `git status` should NOT list `.env` or `.venv/`.

---

<a name="phase-2"></a>
## Phase 2 — RAG core (embeddings + vector DB)

**Goal:** turn documents into vectors stored in Qdrant, and retrieve the most relevant chunks for a question.

**Files created:** `backend/requirements.txt`, `backend/config.py`, `backend/embeddings.py`, `backend/ingest.py`, `backend/retrieve.py`.

### Concept
RAG = **chunk** documents → **embed** each chunk into a vector (numbers capturing meaning) → store in a **vector DB** → at query time, embed the question and find the closest chunks. Then those chunks feed an answer.

### Install

`backend/requirements.txt` (started with):
```
google-genai
qdrant-client
pypdf
python-dotenv
```
```bash
pip install -r backend/requirements.txt
```

### Run Qdrant (no account needed!)

**Important:** Docker Desktop must be open first. I confirm it's running with:
```bash
docker ps        # if this prints a table (even empty), Docker is up
```

Qdrant is the **open-source program**, run locally — I do NOT need a Qdrant cloud account.

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```
- `-d` background, `--name` so I can stop/start it, `-p` exposes ports, `-v` persists data.
- Dashboard: http://localhost:6333/dashboard

### The config file — `backend/config.py`
```python
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]   # documind/ root
load_dotenv(ROOT / ".env")

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documind"
EMBED_DIM = 3072    # Gemini Embedding 2 default size
```
The `parents[1]` trick means scripts find `.env` no matter which folder I run them from.

### Embeddings — `backend/embeddings.py`

> **The big decision in Phase 2:** I used **Gemini Embedding 2**, the newest model. It behaves DIFFERENTLY from older embedding models in two ways, and ignoring these silently produces garbage:
>
> 1. **No `task_type` parameter.** Instead you format the text with a prefix: documents get `title: none | text: ...`, queries get `task: search result | query: ...`.
> 2. **Multiple inputs get AGGREGATED into one vector.** To get one vector per chunk, you must wrap each chunk in its own `types.Content` object.

```python
import os, time
from google import genai
from google.genai import types
from google.genai import errors

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client

EMBED_MODEL = "gemini-embedding-2"

def prepare_document(text, title="none"):
    return f"title: {title} | text: {text}"

def prepare_query(query):
    return f"task: search result | query: {query}"

def embed_texts(texts, batch_size=20, pause_seconds=15):
    if isinstance(texts, str):
        texts = [texts]
    client = _get_client()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
        while True:                       # retry on rate limit
            try:
                result = client.models.embed_content(model=EMBED_MODEL, contents=contents)
                vectors.extend(e.values for e in result.embeddings)
                break
            except errors.ClientError as e:
                if getattr(e, "code", None) == 429:
                    print("  Rate limited — waiting 30s..."); time.sleep(30)
                else:
                    raise
        done = min(i+batch_size, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks...")
        if done < len(texts): time.sleep(pause_seconds)
    return vectors
```

> **ERROR I HIT #1 — rate limit 429.** My first run failed with `429 RESOURCE_EXHAUSTED ... limit: 100 ... gemini-embedding-2`. The **free tier allows ~100 embedding requests per minute**, and each `Content` object counts as one request. My document made **132 chunks**, so it blew past 100. **Fix:** the `batch_size`, `pause_seconds`, and the 429-retry loop above — it slows down and waits instead of crashing. (The version of `embed_texts` shown already includes the fix.)

### Ingestion — `backend/ingest.py`
This reads `docs/`, chunks text, embeds it, and stores vectors in Qdrant. I made it **incremental** (see Phase 3 note — I added that improvement). It also skips image files in Phase 2 (handled in Phase 3).

Core pieces: `chunk_text()` (200-word chunks, 30-word overlap so meaning isn't lost at boundaries), `read_pdf()`, and the Qdrant `upsert`. I embed the **prefixed** text but store the **clean** text in the payload so retrieval shows readable results. The distance metric is **cosine** (standard for text embeddings).

### Retrieval — `backend/retrieve.py`
```python
query_vector = embed_texts(prepare_query(query))[0]
result = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=k)
```
`query_points` is Qdrant's current search method (old `.search()` is deprecated). The query uses `prepare_query` (not `prepare_document`) — the matching half.

### Run + verify
```bash
python backend/embeddings.py            # self-test → "Got an embedding with 3072 dimensions."
python backend/ingest.py                # embeds and stores
python backend/retrieve.py what is the refund policy
```
I read the retrieved chunks and checked they were actually relevant. They were.

> **Common errors:** `KeyError: 'GEMINI_API_KEY'` → key not loaded (check `.env`, run from project root). Connection refused on 6333 → Qdrant not running (`docker start qdrant`). `ModuleNotFoundError` → venv not active / deps not installed.

---

<a name="phase-3"></a>
## Phase 3 — Multimodal (images via VLM + multimodal embeddings)

**Goal:** make image/scanned pages searchable (my `Crypto.png`, `LLM_Bench.png` were being skipped).

**Files created:** `backend/vlm.py`. **Files changed:** `backend/embeddings.py` (added `embed_image`), `backend/ingest.py` (handle images), `backend/retrieve.py` (show result type).

### Concept — two complementary techniques
Because Gemini Embedding 2 is **multimodal**, I do BOTH per image:
1. **Embed the image directly** (`embed_image`) → it lands in the SAME vector space as text → a text query can retrieve the image with no transcription. (Great gist matching.)
2. **VLM description** (`vlm.py` using `gemini-3.5-flash`) → the model "reads" the image, transcribing text and describing charts/tables with numbers → I embed that description as a text chunk. (Captures fine details the raw image embedding misses, and is the explicit "VLM" component.)

### The VLM module — `backend/vlm.py`
```python
VLM_MODEL = "gemini-3.5-flash"
# describe_image(path): sends the image bytes + a "transcribe everything" prompt
# to generate_content and returns the text description (with 429-retry).
```

> **ERROR I HIT #2 — VLM couldn't see my API key.** Running `python backend/vlm.py` directly failed on auth, because `vlm.py` doesn't import `config.py`, so `.env` was never loaded. **Fix:** add these two lines at the top of `vlm.py`:
> ```python
> from dotenv import load_dotenv
> load_dotenv()
> ```
> **General rule I learned:** any file I run *directly* needs its own `load_dotenv()`. Files that `import config` get it for free (because `config.py` calls `load_dotenv` at import time).

### Image embedding — added to `backend/embeddings.py`
```python
def embed_image(path):
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        image_bytes = f.read()
    # embed a SINGLE image Part -> one image vector (3072 dims), with 429-retry
    result = _get_client().models.embed_content(
        model=EMBED_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime)],
    )
    return result.embeddings[0].values
```

### Ingestion change — `backend/ingest.py`
For each image, I create two points: an `image_caption` point (the VLM description, embedded as text) and an `image` point (the raw image embedding). I made ingestion **incremental** here using a content hash:
- It keeps `backend/ingested.json` mapping filename → SHA-256 fingerprint.
- New file → embed it. Changed file (different hash) → delete its old vectors and re-embed. Unchanged → skip (zero API calls, zero cost).

This is the answer to "if I add one new file, don't re-embed everything." I added `backend/ingested.json` to `.gitignore`.

### Run + verify
```bash
python backend/vlm.py docs/Crypto.png         # see the VLM transcription
python backend/ingest.py                        # now embeds the 2 images too
python backend/retrieve.py crypto price trend   # an IMAGE should surface
```
Results are tagged `(IMAGE)` / `(IMAGE_CAPTION)` / `(TEXT)`. Seeing an image returned for a *text* query = cross-modal retrieval working.

---

<a name="phase-4"></a>
## Phase 4 — Local SLM serving (Ollama)

**Goal:** run a small model fully on my machine (the on-prem/edge path), and produce my first complete RAG answer.

**Files created:** `backend/llm_local.py` (later superseded by `llm.py`), `backend/rag.py`.

### Concept
An **SLM** (small model, ~0.5–8B params) is private, cheap, offline, and fast for simple tasks — exactly what edge/on-prem deployments need. **Ollama** runs it locally and exposes an **OpenAI-compatible API**, so the same client code works for local and cloud.

### Install + pull a model
1. Installed Ollama for Windows from https://ollama.com (runs as a tray service on `localhost:11434`).
2. Pulled a model. I used **`llama3.1:8b`** (8B — better answers than a 3B), and also had **`qwen3.5:0.8b`** (tiny/fast).
```bash
ollama pull llama3.1:8b
ollama run llama3.1:8b "say hello"     # quick test; /bye to exit
```

### Install the OpenAI SDK (used to talk to Ollama)
Added `openai` to `backend/requirements.txt`, then `pip install openai`.

### First full RAG answer — `backend/rag.py`
It retrieves chunks (Phases 2–3), builds a grounded prompt ("answer ONLY from the context, say you don't know otherwise, cite sources"), and asks the local model. That grounding prompt is my first defense against hallucination.

```bash
python backend/rag.py what is the refund policy   # full answer + sources used
```

### Tradeoffs I noticed (interview material)
Local 8B = free + private + decent for simple Qs, but slower on long prompts and weaker on hard reasoning. I felt this directly when I added the cloud model in Phase 5.

---

<a name="phase-5"></a>
## Phase 5 — Cloud LLM + provider abstraction

**Goal:** add a cloud frontier model behind ONE interface, so switching providers is just a string.

**Files created:** `backend/llm.py`. **Superseded:** `backend/llm_local.py`. **Changed:** `backend/rag.py` (use `chat()` + provider arg).

### Concept
Never hard-wire business logic to one vendor. I built `chat(messages, provider=...)` with a registry of providers. The trick that makes it elegant: **Ollama, OpenAI, and Gemini all speak the OpenAI chat format**, so one client shape handles all three — only `base_url`, key, and model differ.

### The abstraction — `backend/llm.py`
```python
PROVIDERS = {
  "local":      {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "llama3.1:8b"},
  "local-tiny": {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "qwen3.5:0.8b"},
  "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                 "api_key_env": "GEMINI_API_KEY", "model": "gemini-3.5-flash"},
  "openai":     {"base_url": "https://api.openai.com/v1",
                 "api_key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"},
}
def chat(messages, provider="local", model=None, temperature=None): ...
```

> **Note on model choice:** the current OpenAI flagship is `gpt-5.5`, but I set **`gpt-4o-mini`** because it's much cheaper and totally fine for this project. Swap it anytime.

I also added a tiny `choose_provider()` heuristic (simple questions → local; "why/compare/analyze" or long questions → cloud) to demonstrate **model routing**.

### Run + compare
```bash
python backend/llm.py                                # local + gemini greeting
python backend/rag.py local  what is the refund policy
python backend/rag.py gemini what is the refund policy
python backend/rag.py smart  why might these benchmarks differ   # auto-routes
```

### Ecosystem mapping (the JD listed these)
- **Llama / Qwen** = my local open-weight models.
- **Hugging Face** = where open weights are distributed (Ollama pulls from that ecosystem).
- **Bedrock** = AWS managed gateway fronting many models — would be just another provider entry.
- **OpenAI / Anthropic** = frontier API providers.

---

<a name="phase-6"></a>
## Phase 6 — From RAG to Agent (LangGraph)

**Goal:** turn the fixed pipeline into an **agent** that DECIDES when to use tools.

**Files created:** `backend/agent.py`.

### Concept
A RAG pipeline is a straight line. An **agent** is a loop: the model reasons → optionally calls a **tool** → observes the result → decides to loop again or answer. This is the **ReAct** pattern. **LangGraph** models it as a graph with **State** (the messages), **Nodes** (an `assistant` node = the brain, a `tools` node), and a **conditional edge** that routes to tools or to the end. The leap from RAG: I give the agent a `search_documents` tool, and *the agent decides when to search*.

### Install (LangGraph 1.x)
```bash
pip install langgraph langchain-openai
# optional brains:
pip install langchain-google-genai langchain-ollama
```

### The agent — `backend/agent.py`
- **Tools** (`@tool`): `search_documents(query)` (wraps `retrieve()`), `calculator(expression)` (safe `ast`-based arithmetic, no `eval`).
- **Brain**: `get_model(provider)` → defaults to `ChatOpenAI(model="gpt-4o-mini")`; can use Gemini or local.
- **Graph** (current v1 API):
```python
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)        # calls model.bind_tools(TOOLS)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)   # -> "tools" or END
builder.add_edge("tools", "assistant")          # THE LOOP
agent = builder.compile(checkpointer=MemorySaver())           # memory across turns
```

### Run + watch it think
```bash
python backend/agent.py
# You: what is the refund policy   -> calls search_documents, then answers
# You: what is 1847 * 56           -> calls calculator (not search!)
# You: who won the world cup 1998  -> answers directly, no tool
# You: summarize that in one line  -> remembers context (checkpointer)
```
`stream_mode="values"` + `pretty_print()` shows the loop: Human → AI(tool call) → Tool result → AI(final).

> I also added `print(agent.get_graph().draw_ascii())` to visualize the loop. That needs `pip install grandalf`.

> **Note:** the local brain (`llama3.1:8b`) can call tools but is less reliable at it than cloud models — a good observation that tool-calling quality scales with model size.

---

<a name="phase-7"></a>
## Phase 7 — Backend API (FastAPI)

**Goal:** expose the agent as an HTTP service with streaming, auth, upload, and a health check.

**Files created:** `backend/main.py`. **Changed:** `.env` (added `APP_API_KEY`), `backend/requirements.txt`.

### Install
```
fastapi
uvicorn[standard]
python-multipart
```
```bash
pip install "fastapi" "uvicorn[standard]" python-multipart
```
Added to `.env`:
```dotenv
APP_API_KEY=dev-secret-key
```

### The API — `backend/main.py`
- `GET /health` → `{"status":"ok"}`, **public** (Kubernetes will poll it in Phase 10 — that's why it exists).
- `POST /chat` → streams the agent's answer (`StreamingResponse`); only the final assistant text is streamed (tool steps filtered out via `langgraph_node == "assistant"`).
- `POST /ingest` → save uploaded file to `docs/`, run incremental embed.
- `POST /reindex` → re-scan `docs/`.
- **Auth**: `Depends(require_api_key)` checks the `X-API-Key` header on the sensitive routes; `/health` stays open.
- **CORS** allows `localhost:5173` (Vite) and `localhost:3000` (compose frontend).
- The agent is built **once** at startup and reused.

### Run + test
```bash
cd backend
uvicorn main:app --reload --port 8000      # leave running

# second terminal:
curl http://localhost:8000/health
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" -d '{"question":"what is the refund policy"}'
```
Interactive docs at http://localhost:8000/docs (great for trying `/ingest` with a file picker).

> **ERROR I HIT #3 — `curl: (26) Failed to open/read local data`.** My `-F "file=@docs/Pejman.txt"` failed because I ran curl from a folder where `docs/` wasn't directly beneath me (working-directory issue), and the project path has a space (`E:\My Courses\...`). **Fix:** run curl from the project root and use an explicit relative path: `-F "file=@./docs/Pejman.txt"`. (Or just use the `/docs` web UI file picker, which avoids path quoting entirely.)

---

<a name="phase-8"></a>
## Phase 8 — Frontend (React)

**Goal:** a chat UI with live-streaming answers and file upload.

**Files created:** `frontend/` (Vite app), with `frontend/src/App.jsx` and `frontend/src/App.css` replaced; `frontend/src/index.css` emptied.

### Scaffold
```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
```

### What I built — `frontend/src/App.jsx`
- A chat box that POSTs to `/chat` and reads the response stream with `resp.body.getReader()`, updating the last message as tokens arrive (so the answer types itself out).
- A "+ Add document" button that POSTs to `/ingest` with `FormData`.
- A per-session `thread_id` so the agent remembers the conversation.
- The API key is in the client for the demo (`X-API-Key: dev-secret-key`).

### Run
```bash
# backend running (Phase 7), then:
cd frontend
npm run dev          # http://localhost:5173
```
Ask a question → watch it stream, with cited source filenames. Upload a file → ask about it.

> **Security note I documented** (and why I did NOT build a login system): putting the API key in the browser is fine for a dev demo but insecure for production, because anyone can read it in the network tab. The production fix is a **backend-for-frontend**: the browser authenticates the *user* (session/JWT), and a server proxy holds the real key. I deliberately scoped this out — building auth would be a separate app and would eat time better spent on Phases 9–12. I recorded this as a known limitation in `docs/SECURITY.md`. **Lesson: a documented deliberate gap is judgment; an unnoticed gap is a hole.**

---

<a name="phase-9"></a>
## Phase 9 — Containerization (Docker + compose)

**Goal:** package backend, frontend, and Qdrant so the whole stack boots with one command.

**Files created:** `.dockerignore`, `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`. **Changed:** `backend/config.py` and `backend/llm.py` (env-var configurable URLs).

### Code tweaks for container networking
Inside compose, `localhost` means "this container," so the backend must reach Qdrant by **service name**. I made URLs env-configurable:
- `backend/config.py`: `QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")`
- `backend/llm.py`: `OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")` (used by the `local` providers).

### `.dockerignore` (keeps secrets/junk out of images)
Excludes `.venv`, `node_modules`, `__pycache__`, **`.env`**, `.git`, `dist`, `qdrant_storage*`, `backend/ingested.json`.

### `backend/Dockerfile` (non-root, cached deps)
```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # deps before code = cached
COPY backend/ .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser                                          # non-root = security
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `frontend/Dockerfile` (multi-stage: build with Node, serve with nginx)
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html      # only static files ship
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### `docker-compose.yml`
Three services: `qdrant` (named volume `qdrant_data`), `backend` (`env_file: .env` injects keys at runtime, `QDRANT_URL=http://qdrant:6333`, `./docs` mounted), `frontend` (port 3000→80). `host.docker.internal` lets the backend reach host Ollama.

### Run
```bash
docker stop qdrant            # free port 6333 from the Phase-2 standalone container
# also Ctrl+C any host uvicorn / vite
docker compose up --build
# second terminal:
curl -X POST http://localhost:8000/reindex -H "X-API-Key: dev-secret-key"
# open http://localhost:3000
```

### Errors I hit in this phase (several!)

> **ERROR #4 — empty/crossed Dockerfiles.** First build said `transferring dockerfile: 2B` (backend Dockerfile was basically empty) and the *frontend* tried to pull `python:3.12-slim`. The two files' contents had gotten crossed/blank. **Fix:** rewrite both cleanly with a here-doc so there's no copy-paste ambiguity:
> ```bash
> cat > backend/Dockerfile << 'EOF'
> ...backend content...
> EOF
> cat > frontend/Dockerfile << 'EOF'
> ...frontend content...
> EOF
> ```
> Verify with `head -1 backend/Dockerfile` (should be `FROM python...`) and `head -1 frontend/Dockerfile` (should be `FROM node...`).

> **ERROR #5 — `angchain-ollama`.** Build failed: `Could not find a version that satisfies the requirement angchain-ollama`. Typo in `requirements.txt` — missing the leading `l`. **Fix:** make it `langchain-ollama`.

> **ERROR #6 — `OPENAI_API_KEY` missing in the container.** Backend crashed building the agent because the key wasn't present. The crashed container had started before `.env` was wired. Also the "frontend" logs were showing backend errors — because of the earlier crossed images, the frontend image was actually running backend code. **Fix:** confirm `.env` has the key (`grep -c OPENAI_API_KEY .env` → `1`, values unquoted), then rebuild clean: `docker compose build --no-cache && docker compose up`.

> **ERROR #7 — `qdrant_storage;C` folder.** On Git-bash, `${PWD}` in the `-v` mount got mangled into a weird folder named `qdrant_storage;C`. **Fix (what I did):** I switched persistence to compose's **named volume** (`qdrant_data`) anyway, and just deleted the stray folder: `rm -rf 'qdrant_storage;C'`. Confirmed data was fine via `curl http://localhost:6333/collections/documind` (showed `points_count: 137`).

### Networking mental model (important)
Three "localhost"s: the **browser** uses published ports (`localhost:8000`); the **backend container** reaches Qdrant by service name (`qdrant:6333`); the backend reaches **host Ollama** via `host.docker.internal`. Published ports = host↔container; service names = container↔container.

### Start/stop after first build
```bash
docker compose up            # reuse existing images (no --build)
docker compose up -d         # background
docker compose down          # stop (keeps data volume)
docker compose down -v       # stop + delete vector data
```
Re-run with `--build` only after changing code / Dockerfile / requirements.

---

<a name="phase-10"></a>
## Phase 10 — Kubernetes (kind)

**Goal:** deploy the exact images to a real local cluster and watch it self-heal.

**Files created:** `infra/k8s/qdrant.yaml`, `infra/k8s/backend.yaml`, `infra/k8s/frontend.yaml`.

### Concept (the 5 words)
- **Pod** = smallest unit (a running container); disposable.
- **Deployment** = "keep N pods running; replace dead ones" → self-healing.
- **Service** = stable network name in front of pods.
- **Secret** = sensitive values injected into pods (API keys here).
- **Ingress** = routes external HTTP (I used `port-forward` for the demo).

Mental model: **I declare desired state in YAML; Kubernetes makes reality match it.**

### Install (Windows, ran from folder with `./` + `.exe`)
```bash
curl -LO "https://dl.k8s.io/release/v1.30.0/bin/windows/amd64/kubectl.exe"
curl -Lo kind.exe https://kind.sigs.k8s.io/dl/v0.23.0/kind-windows-amd64
./kubectl.exe version --client
./kind.exe version
```

### Cluster + load images + secret
```bash
docker compose down                      # free ports first
./kind.exe create cluster --name documind
./kubectl.exe get nodes                  # one node, Ready

./kind.exe load docker-image documind-backend  --name documind
./kind.exe load docker-image documind-frontend --name documind

# Secret straight from .env (keys never go in YAML, never in shell history):
./kubectl.exe create secret generic documind-secrets --from-env-file=.env
./kubectl.exe describe secret documind-secrets    # shows key names + byte counts
```

> **Tip I learned:** `--from-env-file=.env` is safer than typing `--from-literal` keys on the command line (no secrets in shell history). Caveat: values in `.env` must be **unquoted** (`KEY=value`, not `KEY="value"`), because the file reader keeps quotes literally.

### Manifests — `infra/k8s/`
- `qdrant.yaml`: Deployment + Service named `qdrant`, `emptyDir` storage (ephemeral, demo-only).
- `backend.yaml`: Deployment with `envFrom: secretRef: documind-secrets`, `QDRANT_URL=http://qdrant:6333`, and **liveness/readiness probes hitting `/health`** (that's why I built `/health` in Phase 7). Plus a Service.
- `frontend.yaml`: Deployment + Service. `imagePullPolicy: IfNotPresent` so it uses the locally loaded image.

### Deploy + access
```bash
./kubectl.exe apply -f infra/k8s/
./kubectl.exe get pods -w               # wait for 3x Running 1/1

# each port-forward holds its terminal:
./kubectl.exe port-forward service/backend 8000:8000
./kubectl.exe port-forward service/frontend 3000:80

# embed + test:
curl -X POST http://localhost:8000/reindex -H "X-API-Key: dev-secret-key"
curl http://localhost:8000/health
```

### Self-healing demo (best part)
```bash
./kubectl.exe delete pod -l app=backend
./kubectl.exe get pods -w               # a NEW backend pod appears automatically
```

> **What I observed / ERROR #8 — only "Embedded 1/1 chunks".** In the cluster, `/health` returned ok and a chat worked, but reindex only embedded **1 chunk**, not 137. Reason: the K8s manifests don't mount my `docs/` (unlike compose) and use ephemeral storage — so the pod only had whatever doc was baked into the image. This is fine for proving the deployment works (Deployments, Services, Secrets, probes, self-healing all demonstrated). Optional fix to get all docs in-cluster: add `COPY docs/ /app/docs` to `backend/Dockerfile`, rebuild, `kind load`, `kubectl rollout restart deployment backend`, reindex. **Production answer:** use a persistent volume / object storage instead of baking docs into the image.

### Teardown
```bash
./kubectl.exe delete -f infra/k8s/
./kind.exe delete cluster --name documind
```

---

<a name="phase-11"></a>
## Phase 11 — Observability & evaluation (Langfuse)

**Goal:** see inside every request (tracing) and measure answer quality (evaluation). This is LLMOps.

**Files created:** `backend/evaluate.py`. **Changed:** `backend/main.py` (optional Langfuse callback), `.env` (Langfuse keys), `backend/requirements.txt`.

### Concept
For an LLM app a `200 OK` can still be a wrong/hallucinated answer. So I trace each request (prompt, retrieval, tool calls, tokens, latency, cost) and run **evals** (faithfulness + relevance) with an **LLM-as-a-judge**. **Langfuse** is the tool.

### Setup
I used **Langfuse Cloud free tier** (self-hosting v3 needs Postgres + Clickhouse + Redis + object storage — too heavy alongside k8s; cloud is identical code and self-hosting is the on-prem talking point). Signed up at https://cloud.langfuse.com, created a project, copied keys into `.env`:
```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```
```bash
pip install langfuse
python -c "from dotenv import load_dotenv; load_dotenv(); from langfuse import get_client; print(get_client().auth_check())"   # True
```

### Tracing — `backend/main.py`
Langfuse v3 changed the SDK. The integration is now:
```python
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()        # reads keys from env
# pass into the LangGraph invocation config:
config["callbacks"] = [langfuse_handler]
config["metadata"] = {"langfuse_session_id": req.thread_id}
```
I made it optional (only activates if `LANGFUSE_PUBLIC_KEY` is set), so the app still runs without Langfuse.

### Run + view
```bash
docker compose up -d qdrant
cd backend
uvicorn main:app --reload --port 8000        # logs: "Langfuse tracing enabled."
# send a couple of /chat requests, then open the Langfuse UI → Tracing → Traces
```
Each trace shows the full agent run (LLM call, tool call + its output, tokens, latency, cost). Conversations group by session (`thread_id`).

### Evaluation — `backend/evaluate.py`
A fixed eval set of questions → run each through the agent (unique `thread_id` per item so it doesn't "remember" previous answers) → an LLM judge scores **faithfulness** and **relevance** 1–5 → prints a report with averages.
```bash
cd backend
python evaluate.py
```
The LLMOps loop: change something (prompt, chunk size, model) → re-run evals → compare scores → keep or revert → watch production traces for drift.

> Prompt management (knowing, not building): Langfuse can version prompts so I could roll back a bad prompt without redeploying.

---

<a name="phase-12"></a>
## Phase 12 — CI/CD + security

**Goal:** automate test → build → vulnerability scan on every push, and write the security overview.

**Files created:** `backend/tests/test_basic.py`, `.github/workflows/ci.yml`. **Expanded:** `docs/SECURITY.md`.

### Tests — `backend/tests/test_basic.py`
Pure tests (no external services, since CI has no Qdrant/keys/Ollama): they check `chunk_text` overlap, the `calculator` tool, and the embedding prefix functions. I set a dummy `OPENAI_API_KEY` at the top so importing modules that build a model client doesn't fail at import time.
```bash
pip install pytest
cd backend
python -m pytest tests -v        # 3 passed
```

### Pipeline — `.github/workflows/ci.yml`
Two jobs on push/PR to `main`:
1. **test** — install deps, run `pytest`.
2. **build-and-scan** (`needs: test`, so it only runs if tests pass) — `docker build` both images, then **Trivy** scans each for CRITICAL/HIGH vulnerabilities.

Trivy is set to `exit-code: '0'` (report-only) for the demo so the build stays green and I can read the scan output; in production you set it to `'1'` to **block** releases on fixable vulnerabilities. (My scans came back **0 vulnerabilities**.)

### Push to GitHub (with a secret-safety check FIRST)
```bash
git check-ignore .env        # MUST print ".env" (means it's ignored). If blank, STOP and fix .gitignore.
git status                   # confirm .env is NOT listed
git add -A
git commit -m "DocuMind: end-to-end agentic RAG with CI"
git branch -M main
git remote add origin https://github.com/<me>/<repo>.git
git push -u origin main
```
Then GitHub → **Actions** tab shows the pipeline running test → build → scan.

> If I ever committed `.env` by accident: `git rm --cached .env`, commit, and **rotate the keys** (treat them as compromised).

### GitLab / Jenkins mapping (the JD named these)
Same `test → build → scan` stages, different syntax: a `.gitlab-ci.yml` with `stages:` and `script:` blocks, or a Jenkins `Jenkinsfile` with `pipeline { stages { ... } }`. The concept transfers directly.

### Security overview — `docs/SECURITY.md`
Organized by the three pillars: **encryption** (TLS in transit via Ingress, Secrets encrypted at rest), **access control** (API-key auth, public `/health`, runtime-injected secrets, non-root containers, plus the planned RBAC + backend-for-frontend), and **vulnerability management** (Trivy in CI, slim/multi-stage images, `.dockerignore`). It also lists the deliberate demo-only simplifications.

---

<a name="final"></a>
## Final notes & cleanup

### What this project demonstrates end to end
Agentic multimodal RAG · SLM + cloud LLM behind one swappable interface · React UI with streaming · Docker + docker-compose · real Kubernetes deployment with Secrets and health probes · Langfuse observability + LLM-as-judge evals · GitHub Actions CI/CD with Trivy scanning · a documented security posture.

### Files I committed that I normally wouldn't
`kind.exe` and `kubectl.exe` are in the repo for Windows convenience. In a real project these binaries should be installed locally and added to `.gitignore`, not committed (they're large and platform-specific).

### Recurring lessons (things that bit me, so they won't bite you)
- **`load_dotenv()`** is needed in any file you run *directly*; files that `import config` get it free.
- **Gemini Embedding 2**: no `task_type` (use text prefixes); multiple inputs aggregate (wrap each in a `Content` object).
- **Free-tier rate limits** are real — batch + pause + retry on 429.
- **curl `@file` paths** are relative to your current folder; use `@./docs/file` from the project root.
- **Docker on Git-bash**: `${PWD}` volume mounts can mangle (`;C`); named volumes avoid this.
- **Compose secrets**: inject via `env_file` at runtime; never bake into images (`.dockerignore` excludes `.env`).
- **`docker compose up` vs `--build`**: only rebuild after code/Dockerfile/requirements changes.
- **Kubernetes** `localhost` ≠ host; use Service names; `/health` exists for probes.

### Full cleanup
```bash
docker compose down -v                       # stop stack + delete vector data
./kind.exe delete cluster --name documind    # delete cluster (if running)
docker stop qdrant && docker rm qdrant       # remove old standalone container (if any)
ollama rm llama3.1:8b                         # remove local model (optional)
```

### Cost reality
Cloud calls (Gemini/OpenAI) bill **only when a request runs** — no idle/background cost. Local components (Qdrant, Ollama, kind) are free. Deleting local data never refunds or re-charges anything; rebuilding just re-runs the (cheap) embedding calls.

---

That's the entire journey, from an empty folder to a production-style system. If you followed every step, you now understand not just *what* each piece does but *why* it's there and *how* the parts connect — which is the whole point.
