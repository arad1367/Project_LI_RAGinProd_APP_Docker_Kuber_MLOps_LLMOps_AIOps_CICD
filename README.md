# DocuMind — Production-Style Agentic Multimodal RAG

An end-to-end, industry-style **Generative + Agentic AI** system: an assistant that ingests your documents (including images, tables, and scanned pages), answers questions about them with grounded citations, and runs as a real service — containerized, deployed to Kubernetes, observable, evaluated, and shipped through a CI/CD pipeline with security scanning.

This repository is **both a working application and a complete learning resource**.

> ## 📚 New here? Read **[`LEARNING.md`](./LEARNING.md)** first.
> `LEARNING.md` is the full, beginner-friendly, zero-to-finished walkthrough of how this entire project was built — every step, every command, every installation, every error encountered and how it was fixed, and which source file belongs to which part. **If you want to learn or reproduce this project, follow `LEARNING.md` from top to bottom.** This README is the operational reference (how to run it); `LEARNING.md` is the teaching guide (how and why it was built).

---

## What it does

- **Retrieval-Augmented Generation (RAG)** over your own documents.
- **Multimodal**: text, PDFs, and images embedded into one shared vector space (text queries can retrieve images), plus a vision model that transcribes charts/tables.
- **Agentic**: a LangGraph agent decides *when* to search documents, do math, or answer directly — it is not a fixed pipeline.
- **Hybrid models**: a small local model (on-prem/edge) and a cloud frontier model behind one swappable interface.
- **Grounded answers** with source citations, defending against hallucination.
- **Observability + evaluation** (LLMOps) and a **CI/CD pipeline** with vulnerability scanning.

## Architecture

```
                ┌──────────────┐
   Browser ───▶ │ React (Vite) │  chat UI, streaming, file upload
                └──────┬───────┘
                       │ HTTP (X-API-Key)
                ┌──────▼───────────────────────────┐
                │ FastAPI backend                   │
                │  /chat  /ingest  /reindex /health │
                │                                   │
                │  LangGraph agent (ReAct loop)     │
                │   ├─ tool: search_documents ──────┼──▶ Qdrant (vector DB)
                │   └─ tool: calculator             │
                │                                   │
                │  Models (one interface):          │
                │   ├─ local SLM  (Ollama)          │
                │   └─ cloud LLM  (OpenAI / Gemini) │
                │                                   │
                │  Embeddings: Gemini Embedding 2   │
                │  Vision (VLM): Gemini 3.5 Flash   │
                └──────────────┬────────────────────┘
                               │ traces / evals
                        ┌──────▼───────┐
                        │  Langfuse    │  (LLMOps observability)
                        └──────────────┘

Packaging: Docker → docker-compose → Kubernetes (kind)
CI/CD: GitHub Actions (test → build → Trivy scan)
```

## Tech stack

| Layer | Choice |
|------|--------|
| Embeddings | Gemini Embedding 2 (multimodal) |
| Vision (VLM) | Gemini 3.5 Flash |
| Vector DB | Qdrant |
| Local model (SLM) | Ollama (`llama3.1:8b`, `qwen3.5:0.8b`) |
| Cloud model (LLM) | OpenAI (`gpt-4o-mini`) / Gemini |
| Agent | LangGraph (ReAct) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite, served by nginx |
| Containers | Docker, docker-compose |
| Orchestration | Kubernetes via `kind` |
| Observability | Langfuse |
| CI/CD | GitHub Actions + Trivy |

## Repository layout

```
.
├── backend/
│   ├── config.py          # shared settings, loads .env, QDRANT_URL
│   ├── embeddings.py       # Gemini Embedding 2: text + image embedding
│   ├── vlm.py              # Vision model: describe/transcribe images
│   ├── ingest.py           # chunk → embed → store (incremental)
│   ├── retrieve.py         # query → vector search → top-k chunks
│   ├── llm.py              # provider abstraction (local/gemini/openai) + routing
│   ├── agent.py            # LangGraph agent + tools (search_documents, calculator)
│   ├── main.py             # FastAPI app (/chat /ingest /reindex /health) + Langfuse
│   ├── evaluate.py         # LLM-as-a-judge evaluation over a fixed set
│   ├── requirements.txt
│   ├── Dockerfile          # non-root Python image
│   └── tests/test_basic.py # pure unit tests (run in CI)
├── frontend/
│   ├── src/App.jsx         # chat UI with streaming + upload
│   ├── src/App.css
│   └── Dockerfile          # multi-stage build → nginx static serving
├── infra/k8s/
│   ├── qdrant.yaml         # Deployment + Service
│   ├── backend.yaml        # Deployment + Service + Secret + health probes
│   └── frontend.yaml       # Deployment + Service
├── .github/workflows/ci.yml  # test → build → Trivy scan
├── docs/                   # your source documents + SECURITY.md
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── LEARNING.md             # ← full build-from-scratch guide (read this)
└── README.md
```

---

## Prerequisites

- **Docker Desktop** (running)
- **Python 3.11+**
- **Node.js 20+**
- **Ollama** (for the local model) — https://ollama.com
- API keys: **Gemini** (embeddings + vision) and **OpenAI** (cloud LLM). A **Langfuse** account is optional (observability).

## Configuration — `.env`

Create a `.env` in the project root (it is git-ignored — **never commit it**):

```dotenv
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
APP_API_KEY=dev-secret-key

# Optional (Phase 11 observability)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Running the project

There are two ways to run it. **Option A (Docker Compose)** is the simplest — the whole stack with one command.

### Option A — Docker Compose (recommended)

**Start, in order:**

```bash
# 0. Make sure Docker Desktop is running, and (for the local model) Ollama is running.
ollama pull llama3.1:8b           # one-time, if not already pulled

# 1. Build and start the whole stack (backend + frontend + Qdrant)
docker compose up --build         # first run; later just: docker compose up

# 2. In a second terminal, embed your documents into the fresh vector DB
curl -X POST http://localhost:8000/reindex -H "X-API-Key: dev-secret-key"

# 3. Open the app
#    Frontend:  http://localhost:3000
#    API docs:  http://localhost:8000/docs
#    Qdrant UI: http://localhost:6333/dashboard
```

**Stop, in order:**

```bash
# Stop containers (keeps the Qdrant data volume)
docker compose down

# Stop AND delete the vector data volume (full reset)
docker compose down -v
```

**Day-to-day:** after the first `--build`, just use `docker compose up` (add `-d` to run in the background; `docker compose logs -f` to watch logs). Only re-run with `--build` after changing code, a `Dockerfile`, or `requirements.txt`.

### Option B — Local development (faster iteration)

```bash
# 1. Activate the Python environment and install deps
python -m venv .venv
source .venv/Scripts/activate          # Windows Git-bash
pip install -r backend/requirements.txt

# 2. Start ONLY Qdrant via compose (reuses its data volume)
docker compose up -d qdrant

# 3. Start the backend (from inside backend/)
cd backend
uvicorn main:app --reload --port 8000

# 4. In another terminal, start the frontend
cd frontend
npm install
npm run dev                            # http://localhost:5173

# 5. Embed documents
curl -X POST http://localhost:8000/reindex -H "X-API-Key: dev-secret-key"
```

---

## API endpoints

| Method | Path | Auth | Purpose |
|-------|------|------|---------|
| GET | `/health` | none | Liveness/readiness (used by Kubernetes) |
| POST | `/chat` | `X-API-Key` | Ask a question; streams the answer |
| POST | `/ingest` | `X-API-Key` | Upload a file; saved + embedded |
| POST | `/reindex` | `X-API-Key` | Re-scan `docs/` and embed new/changed files |

Example:

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" -H "X-API-Key: dev-secret-key" \
  -d '{"question": "what is the refund policy"}'
```

---

## Observability (LLMOps)

With the `LANGFUSE_*` keys set in `.env`, every `/chat` request produces a full trace (prompt, retrieval, tool calls, tokens, latency, cost) in your Langfuse project. Run the evaluation suite:

```bash
cd backend
python evaluate.py        # LLM-as-a-judge: faithfulness + relevance scores
```

---

## Kubernetes deployment (local `kind`)

```bash
# Create cluster
kind create cluster --name documind

# Load locally-built images into the cluster
kind load docker-image documind-backend --name documind
kind load docker-image documind-frontend --name documind

# Create the Secret from your .env (keys never go in the YAML)
kubectl create secret generic documind-secrets --from-env-file=.env

# Deploy
kubectl apply -f infra/k8s/
kubectl get pods -w

# Access (each holds its terminal)
kubectl port-forward service/backend 8000:8000
kubectl port-forward service/frontend 3000:80

# Embed, then use as above
curl -X POST http://localhost:8000/reindex -H "X-API-Key: dev-secret-key"
```

Teardown:

```bash
kubectl delete -f infra/k8s/
kind delete cluster --name documind
```

> Note: the demo Kubernetes manifests use ephemeral storage and bake `docs/` into the image. A production deployment would use a persistent volume / object storage and an Ingress with TLS.

---

## CI/CD

`.github/workflows/ci.yml` runs on every push to `main`:

1. **test** — installs deps and runs `pytest` (pure unit tests).
2. **build-and-scan** — builds both Docker images and scans them with **Trivy** for CRITICAL/HIGH vulnerabilities.

The build job only runs if tests pass. Trivy is report-only in the demo (`exit-code: 0`); set it to `1` to block releases on fixable findings.

---

## Security

See [`docs/SECURITY.md`](./docs/SECURITY.md) for the full overview across encryption, access control, and vulnerability management, including deliberate demo-only simplifications and their production fixes.

---

## Cleanup (everything)

```bash
docker compose down -v                       # stop stack + delete vector data
kind delete cluster --name documind          # delete the cluster (if created)
ollama rm llama3.1:8b                         # remove the local model (optional)
```

Cost note: cloud calls (Gemini/OpenAI) are billed per use only when scripts/requests run; there is no idle/background billing. Local components (Qdrant, Ollama, kind) are free.

---

## Notes

- This is a **learning/portfolio project**, structured to resemble a real production system end to end. No proprietary or company-specific content is included.
- `kind.exe` / `kubectl.exe` are committed here only for convenience on Windows; in a real repository these binaries should be installed locally and git-ignored, not committed.
- Full build narrative, rationale, and troubleshooting: **[`LEARNING.md`](./LEARNING.md)**.
