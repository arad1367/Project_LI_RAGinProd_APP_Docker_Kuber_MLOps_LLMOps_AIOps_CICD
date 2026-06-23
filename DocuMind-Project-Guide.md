# DocuMind — Your One-Project Path Through the Whole Job Description

**An Agentic, Multimodal Document-Intelligence Assistant you build from scratch.**

This single project deliberately touches _every_ requirement in the Lenovo posting. You build it once, end to end, and along the way you understand RAG, agents, SLM/LLM/VLM serving, containers, Kubernetes, CI/CD, security, and LLMOps — not as buzzwords, but as parts you wired together yourself.

> **Read this first.** The goal this week is **understanding the concepts and the process**, not production perfection. By the end you'll have a working system and, more importantly, you'll be able to explain _why_ each piece exists and _how_ they connect. That's what wins interviews.

---

## What you're building (the 30-second pitch)

> "DocuMind is an agentic assistant that ingests a company's documents — including ones with images, tables, and scanned pages — and answers questions about them. It runs a small model locally for cheap/private tasks and calls a large cloud model for hard reasoning. It's containerized, deployed on Kubernetes, fully observable, evaluated automatically, and shipped through a CI/CD pipeline with security scanning."

Say that sentence in the interview and you've already covered 80% of the job description. The rest of this guide is you _earning the right_ to say it.

---

## How each phase maps to the job requirements

| Phase | What you build                        | Requirement it covers                                  |
| ----- | ------------------------------------- | ------------------------------------------------------ |
| 1     | Environment + mental model            | Foundations                                            |
| 2     | RAG core (embeddings + vector DB)     | GenAI solutions, modern RAG                            |
| 3     | Multimodal ingestion (VLM)            | VLMs, multi-modal GenAI                                |
| 4     | Local SLM with Ollama                 | SLMs, on-prem/edge                                     |
| 5     | Cloud LLM + provider abstraction      | LLM ecosystems (Anthropic, OpenAI, Bedrock, Llama, HF) |
| 6     | LangGraph agent + tools               | Agentic AI, inference engines                          |
| 7     | FastAPI backend                       | Python, AI architectures, APIs                         |
| 8     | React frontend                        | React, user interfaces                                 |
| 9     | Docker + docker-compose               | Containerization                                       |
| 10    | Kubernetes (local kind cluster)       | Kubernetes, cloud/on-prem deployment                   |
| 11    | Langfuse observability + evals        | MLOps / LLMOps / AIOps                                 |
| 12    | GitHub Actions CI/CD + Trivy security | CI/CD pipelines, security standards                    |
| Bonus | GPU-serving concepts (vLLM, TensorRT) | NVIDIA technologies (talking points)                   |

---

## Realistic time plan (7 days)

You will _not_ finish every phase to production depth in a week. That's fine and expected. Prioritize **understanding** over completeness. Suggested pacing:

- **Day 1:** Phases 1–2 (environment + RAG core). This is the heart — spend real time here.
- **Day 2:** Phases 3–5 (multimodal, local SLM, cloud LLM).
- **Day 3:** Phase 6 (the agent — the "wow" concept for the interview).
- **Day 4:** Phases 7–8 (backend + frontend, enough to demo).
- **Day 5:** Phases 9–10 (Docker + Kubernetes — the gap you mentioned).
- **Day 6:** Phases 11–12 (observability + CI/CD + security — your other gaps).
- **Day 7:** Review, write your "talking points" notes, rehearse the architecture out loud.

If you run short on time, it is **far better** to deeply understand Phases 1–6 and 9–12 conceptually than to have every line of code working. You can demo a half-built system and still ace the interview if you can _explain_ it.

---

## The tech stack (current as of mid-2026, chosen on purpose)

Every choice below is deliberate, and being able to justify a choice is itself an interview skill.

- **Orchestration / agent:** **LangGraph** — the current production standard for stateful, controllable agents (graph of nodes/edges, checkpointing, human-in-the-loop). Interviewers know it.
- **Embeddings:** **Gemini Embedding** (the current MTEB leader and natively multimodal — text + images in one vector space), with **a local open alternative** (e.g. a BGE/Qwen-family model via Ollama or sentence-transformers) so you can speak to the cloud-vs-on-prem tradeoff. _Knowing how to swap the embedding model is more valuable than the specific model._
- **Vector database:** **Qdrant** — open source, runs in one Docker container, trivial to deploy to Kubernetes later.
- **Local SLM (on-prem/edge):** **Ollama** running a small model (e.g. a ~1–8B Llama/Qwen/Phi-class model). Works on Windows, no datacenter GPU required.
- **Cloud LLM:** A hosted frontier model via API (Anthropic Claude and/or OpenAI). You'll wrap it so the provider is swappable.
- **VLM (vision):** A multimodal model for reading images/scanned pages (the cloud LLM's vision mode is the simplest path; note the local-VLM option too).
- **Backend:** **FastAPI** (Python) — async, typed, the de-facto standard for AI services.
- **Frontend:** **React** (Vite) — the JD explicitly asks for React.
- **Containers:** **Docker** + **docker-compose** (you already have Docker Desktop).
- **Kubernetes:** **kind** (Kubernetes-in-Docker) running locally on Docker Desktop — a real cluster, zero cloud cost.
- **Observability / LLMOps:** **Langfuse** (open source, self-hosted via Docker) — tracing, prompt management, evaluation.
- **CI/CD:** **GitHub Actions** (I'll note the GitLab CI equivalent, since the JD mentions GitLab/Jenkins).
- **Security:** **Trivy** (image/vuln scanning), Kubernetes Secrets, API auth.

---

# PHASE 1 — Foundations & Environment

**Goal:** Understand the system you're about to build, and get a clean working environment on Windows.

### Concept first: what _is_ this system, really?

Three mental models to hold:

1. **RAG (Retrieval-Augmented Generation):** LLMs don't know your private documents and they hallucinate. RAG fixes this by: (a) converting documents into vectors and storing them, (b) when a question comes in, finding the most relevant chunks by vector similarity, (c) stuffing those chunks into the prompt so the model answers _from your data_. Retrieve, then generate.

2. **Agent:** A plain RAG pipeline always does the same steps. An **agent** is an LLM that _decides_ what to do — "should I search the docs? call a calculator? ask the user to clarify? answer directly?" — loops until done, and can use **tools**. This is the "Agentic AI" the JD keeps mentioning.

3. **The serving spectrum (cloud → on-prem → edge):** The same app can call a model over the internet (cloud), inside your own datacenter (on-prem), or on a local device (edge). You'll touch all three: cloud API, local Ollama, and the _concept_ of GPU serving (vLLM/TensorRT).

### Steps

1. **Install tooling** (you have some already):
   - Docker Desktop (you have it) — enable the WSL 2 backend in settings; it makes everything smoother on Windows. (check if installed previously --> `wsl -l -v`)
   - Python 3.11+ — `python --version` in bash to confirm.
   - Node.js 20+ — for the React frontend.
   - Git — `git --version`.
   - A code editor (VS Code recommended).
2. **Create the project skeleton** in bash:
   ```bash
   mkdir documind && cd documind
   git init
   mkdir backend frontend infra docs
   python -m venv .venv
   source .venv/Scripts/activate   # Windows/Git-bash venv path
   ```
3. **Get API keys** and store them in a `.env` file (never commit this):
   - An LLM provider key (Anthropic and/or OpenAI).
   - A Gemini API key (for embeddings) — or plan to use a local embedding model.
   - Create `.gitignore` with `.env`, `.venv/`, `node_modules/` on day one. This is a security habit interviewers notice.
4. **Drop 3–5 sample documents** into `docs/` — ideally a mix: a text-heavy PDF, one with a chart/table, and one scanned/image page. This variety is what makes the _multimodal_ phase real.

> **Interview hook:** "I keep secrets out of the image and the repo from the start — `.env` locally, Kubernetes Secrets in the cluster, never in code or `kubectl describe` output." That one sentence covers the "access control / secrets" security requirement. In your container image — later (Phase 9) you'll package the app into a Docker "image." A beginner mistake is to copy the key into the image while building it. Then anyone who pulls that image gets your key. The fix is to inject the key at runtime instead of baking it in. You'll see this in Phase 9.In your running Kubernetes cluster — later (Phase 10) the app runs as "pods" in Kubernetes. The right way to give a pod a secret is a Kubernetes Secret object, not hard-coding it. And kubectl describe pod is a command that prints a pod's configuration — if you passed the key the wrong way, it shows up in that output for anyone who can run the command. You'll see the right way in Phase 10.

---

# PHASE 2 — The RAG Core: Embeddings + Vector Database

**Goal:** Turn documents into searchable vectors and retrieve the right chunks. This is the beating heart of the whole system — spend real time understanding it.

### Concept first

- **Chunking:** You can't embed a whole 50-page PDF as one vector — you split it into overlapping chunks (e.g. ~500–1000 tokens, ~10–15% overlap). _Why overlap?_ So a sentence split across a boundary still has context on both sides. Chunking strategy quietly determines RAG quality.
- **Embeddings:** A model maps each chunk to a vector (a list of numbers) where _similar meaning = nearby in space_. The current leader is Gemini Embedding v2 (and it's multimodal — it can put text and images into the _same_ space, which matters in Phase 3). Knowing how to _swap_ embedding models cleanly is the real skill.
- **Vector DB:** Qdrant stores vectors and does fast nearest-neighbor search. When a question comes in, you embed the question and ask Qdrant for the closest chunks.

### Steps

1. **Run Qdrant** in Docker (one command — your first taste of "containerized infra"):
   **_ We need Incremental Ingestion - do not embed files twice _**

- first check docker running : docker ps
  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```
  Open `http://localhost:6333/dashboard` to see it live.

2. **Write the ingestion script** (`backend/ingest.py`). The flow:
   - Load each document (use a PDF/text loader).
   - Split into chunks with overlap.
   - Embed each chunk (call the embedding model).
   - Upsert vectors + the original text + metadata (filename, page) into a Qdrant collection.
3. **Write the retrieval function** (`backend/retrieve.py`): embed a query string → ask Qdrant for top-k chunks → return their text + sources.
4. **Test it manually:** ingest your docs, then run a query like "what is the refund policy?" and print the retrieved chunks. _Look at what comes back._ If it's junk, your chunking or embedding choice is wrong — this debugging instinct is gold.
5. **Make the embedding model swappable:** write a tiny `get_embedder()` function so switching from Gemini to a local model is a one-line config change. Now you can _demonstrate_ the cloud-vs-on-prem tradeoff.

> **Interview hook:** "RAG quality is mostly chunking + embedding choice + retrieval quality, not the LLM. I made the embedder swappable so we can run Gemini in the cloud or an open model on-prem for data-residency requirements." That's a senior-sounding answer.

"""
Cleanup / teardown (when you want to remove everything)
Nothing here bills you while idle — the only charges were the embedding calls you already ran. To remove it all:
Stop and delete the Qdrant container:
bashdocker stop qdrant
`docker rm qdrant`
Delete the stored vectors and the bookkeeping file:
bash `rm -rf qdrant_storage`
`rm -f backend/ingested.json`
Optional — remove the Qdrant image to reclaim disk space:
bash `docker rmi qdrant/qdrant`
If you'd rather pause than delete: just `docker stop qdrant`. Your vectors are preserved by the volume, and `docker start qdrant` brings everything back with no re-embedding needed.
"""

---

# PHASE 3 — Multimodal: Reading Images, Tables & Scanned Pages (VLM)

**Goal:** Handle documents that aren't pure text. The JD says "multi-modal preferred" — this phase earns that line on your resume.

### Concept first

A **VLM (Vision-Language Model)** can _look at_ an image and describe/answer about it. Two patterns:

- **Image-to-text at ingestion:** when a page is a scan or contains a chart, pass it to a VLM, get a text description/transcription, and embed _that text_ alongside the rest. Simple and robust.
- **Multimodal embeddings:** with a multimodal embedding model (like Gemini Embedding), you can embed the image _itself_ into the same vector space as text — so an image can be retrieved by a text query directly. More advanced, very current.

Start with the first pattern; mention the second as the "next step."

### Steps

1. **Detect non-text pages** during ingestion (e.g. a PDF page with little extractable text but an embedded image).
2. **Send the page image to a VLM** (your cloud model's vision mode is easiest) with a prompt like _"Transcribe all text and describe any charts/tables in detail."_
3. **Embed and store the returned text** like any other chunk, tagged `source_type: image`.
4. **Test:** ask a question that can _only_ be answered from a chart/scanned page. If it works, you've built a genuinely multimodal RAG system.

> **Interview hook:** "For multimodal docs I used a VLM to transcribe charts and scans into searchable text at ingestion, and the architecture supports true multimodal embeddings where images live in the same vector space as text."

---

# PHASE 4 — Local SLM Serving (On-Prem / Edge)

**Goal:** Run a _small_ model entirely on your machine. This covers "SLMs" and the "on-premise/edge" deployment requirement — and teaches you the cloud-vs-local tradeoff hands-on.

### Concept first

- **SLM (Small Language Model):** a few hundred million to ~8B parameters. Weaker than frontier models but **cheap, fast, private, and runs without internet**. Perfect for edge devices and privacy-sensitive on-prem deployments — exactly Lenovo's world (they sell edge hardware).
- **Ollama** is the easiest way to run SLMs locally. It serves an **OpenAI-compatible API**, which is the key trick: your app code doesn't care whether it's talking to Ollama or OpenAI — same interface.

### Steps

1. **Install Ollama** and pull a small model:
   ```bash
   ollama pull llama3.2:3b      # or a qwen / phi small model
   ollama run llama3.2:3b "Say hello"
   ```
2. **Hit its API** to confirm it speaks the OpenAI format (`http://localhost:11434/v1/...`).
3. **Wire it as a provider** in your app (you'll formalize this in Phase 5).
4. **Compare:** run the same RAG query through the local SLM and the cloud LLM. Notice the quality, speed, and cost differences _yourself_. That lived experience is what you talk about in the interview.

> **Interview hook:** "I used Ollama to serve an SLM locally for the on-prem/edge path — same OpenAI-compatible interface as the cloud, so routing between them is a config decision driven by cost, latency, and data-sensitivity, not a code rewrite."

---

# PHASE 5 — Cloud LLM + Provider Abstraction (the LLM Ecosystem)

**Goal:** Call a frontier model in the cloud, and build a clean abstraction so providers are swappable. This directly covers "LLM ecosystems (OpenAI, Anthropic, Llama, Hugging Face, Bedrock)."

### Concept first

You never want your business logic hard-coded to one vendor. The pattern: a thin **`generate(messages, model)` interface** with implementations for each provider (Anthropic, OpenAI, Ollama, and conceptually Bedrock/HF). The rest of your app calls the interface, not the vendor.

### Steps

1. **Create `backend/llm.py`** with a single `chat(provider, messages)` function that dispatches to:
   - Anthropic (cloud frontier)
   - OpenAI (cloud frontier)
   - Ollama (local SLM, from Phase 4)
2. **Add a routing rule** (this is a great talking point): e.g. "use the local SLM for simple/cheap tasks, escalate to the cloud LLM for complex reasoning." Even a simple keyword/length heuristic demonstrates the _concept of model routing_.
3. **Name-drop accurately:** you don't need a Bedrock account — but understand that **Bedrock** is AWS's managed gateway to many models (Anthropic, Llama, etc.) and **Hugging Face** hosts open weights you can self-host. Be ready to say _when_ you'd choose each.

> **Interview hook:** "I abstracted the provider behind one interface and added cost-based routing — small model first, escalate to frontier only when needed. In production I'd put Bedrock in front of the cloud models for governance and a single billing/security boundary."

---

# PHASE 6 — From RAG to Agent (LangGraph)

**Goal:** Turn the pipeline into a real **agent** that decides and uses tools. This is the single most impressive concept to demo and discuss. Spend a full day here.

### Concept first

A RAG pipeline is a straight line. An **agent** is a _graph with a loop_: the model reasons, optionally calls a **tool** (search the docs, do a calculation, look something up), observes the result, and decides whether to loop again or answer. **LangGraph** models this as nodes (steps) and edges (transitions), with state carried between them. It's the current production standard because you get checkpointing, streaming, and human-in-the-loop control.

Your agent's tools:

- `search_documents(query)` → your Phase 2 retrieval function.
- `summarize(text)` → calls an LLM.
- (optional) `calculator`, `clarify_with_user`.

### Steps

1. **Install LangGraph** and define an **agent state** (the conversation, retrieved context, scratchpad).
2. **Define nodes:** an "agent/brain" node (the LLM deciding what to do) and "tool" nodes (search, summarize).
3. **Define edges:** conditional edge from the brain — "if the model asked for a tool, go run it; else finish."
4. **Wrap your retrieval as a tool** so the _agent decides_ when to search instead of always searching. This is the leap from RAG to agentic RAG.
5. **Add memory/checkpointing** so multi-turn conversations keep context.
6. **Trace one run end to end** and watch the decision loop. This "watch it think" moment is what you describe in the interview.

> **Interview hook:** "I built it as a LangGraph agent, not a fixed RAG chain — the model decides when to retrieve, when to summarize, when to ask for clarification. LangGraph gives me checkpointing and human-in-the-loop, which matters for auditability in regulated settings."

---

# PHASE 7 — Backend API (FastAPI / Python)

**Goal:** Expose the agent over HTTP so a frontend (and the outside world) can use it. Covers "Python," "AI architectures," and "integrate into existing platforms."

### Steps

1. **Create `backend/main.py`** with FastAPI. Key endpoints:
   - `POST /ingest` — upload/process documents.
   - `POST /chat` — send a question, stream back the agent's answer.
   - `GET /health` — returns 200 (Kubernetes will need this in Phase 10 — note it now).
2. **Add streaming** for the chat endpoint (server-sent events) so answers appear token by token.
3. **Add basic auth** — even a simple API-key header check. This is your "access control" security checkbox and it's genuinely easy.
4. **Run it:** `uvicorn main:app --reload`, test with the auto-generated docs at `/docs`.

> **Interview hook:** "FastAPI gives me async streaming and a typed, self-documenting API. The `/health` endpoint exists specifically so Kubernetes liveness/readiness probes can manage the pod."

---

# PHASE 8 — Frontend (React)

**Goal:** A simple chat UI. The JD explicitly lists React, so even a minimal interface matters.

### Steps

1. **Scaffold with Vite:** `npm create vite@latest frontend -- --template react`.
2. **Build three things only** (keep it minimal — this isn't a design contest):
   - A file-upload control that calls `/ingest`.
   - A chat box that calls `/chat` and renders the streaming response.
   - A "sources" panel showing which document chunks the answer came from (this _citations_ touch impresses people — it shows you understand grounding/hallucination).
3. **Handle the stream** so text appears live.
4. **Run both together:** frontend on one port, backend on another, talking to each other.

> **Interview hook:** "The UI shows source citations for every answer, so users can verify the grounding — that's my first line of defense against hallucination."

---

# PHASE 9 — Containerization (Docker + docker-compose)

**Goal:** Package each piece into containers that run identically anywhere. This is the first half of your stated gap — and it's very learnable in a day.

### Concept first

A **container** bundles your code + its exact dependencies into an image that runs the same on your laptop, a server, or the cloud. _"Works on my machine" disappears._ **docker-compose** runs several containers together (your backend, frontend, Qdrant, Ollama) with one command and a shared network.

### Steps

1. **Write a `Dockerfile` for the backend:** start from a slim Python image, copy code, install deps, expose the port, run uvicorn. Use a **multi-stage build** and a **non-root user** — both are security best practices interviewers love.
2. **Write a `Dockerfile` for the frontend:** build the React app, serve the static files with nginx.
3. **Write `docker-compose.yml`** tying together backend, frontend, and Qdrant on one network, with environment variables and volumes.
4. **Run the whole stack:** `docker compose up`. Your entire app now boots with one command. _Feel that._
5. **Understand image layers and `.dockerignore`** — why order matters for caching, why you exclude `.env` and `node_modules`.

> **Interview hook:** "Each service is a container — multi-stage builds, non-root users, no secrets baked in. docker-compose runs the full stack locally; the same images deploy to Kubernetes unchanged. That's the whole point of containers: build once, run anywhere."

---

# PHASE 10 — Kubernetes (Local `kind` Cluster)

**Goal:** Deploy your containers to a _real_ Kubernetes cluster running locally. This is the biggest gap you named — and you can absolutely get the _concepts_ solid in a day.

### Concept first (learn these five words and you can hold a conversation)

- **Pod:** the smallest unit — one or more containers running together.
- **Deployment:** declares "I want N replicas of this pod, keep them running, roll updates safely."
- **Service:** a stable network address/load-balancer in front of pods (pods come and go; the Service stays).
- **ConfigMap / Secret:** configuration and sensitive values injected into pods (your API keys go in a **Secret**, never in the image).
- **Ingress:** routes outside HTTP traffic to the right Service.

The mental model: **you declare the desired state in YAML; Kubernetes constantly works to make reality match it.** That "declarative, self-healing" idea is the thing to articulate.

### Steps

1. **Install `kind` and `kubectl`.** Create a cluster: `kind create cluster --name documind`. You now have a real Kubernetes cluster _inside Docker Desktop_, free.
2. **Load your images** into kind: `kind load docker-image documind-backend --name documind`.
3. **Write Kubernetes manifests** (`infra/k8s/`):
   - `Deployment` + `Service` for backend, frontend, and Qdrant.
   - A `Secret` holding your API keys (created with `kubectl create secret ...` — _not_ committed to git).
   - **Liveness and readiness probes** pointing at the `/health` endpoint from Phase 7 (now you see why it exists).
4. **Apply them:** `kubectl apply -f infra/k8s/`. Watch pods come up with `kubectl get pods -w`.
5. **Access the app:** `kubectl port-forward` to reach it in your browser.
6. **Break something on purpose:** delete a pod (`kubectl delete pod ...`) and watch Kubernetes recreate it. _That_ is self-healing — and a fantastic thing to describe having witnessed.

> **Interview hook:** "I deployed to a local kind cluster — Deployments for replicas and rolling updates, Services for stable networking, Secrets for keys, liveness/readiness probes on a `/health` endpoint. I killed a pod and watched the Deployment recreate it. Same manifests would run on EKS/GKE or on-prem — that's the cloud/on-prem portability the role needs."

---

# PHASE 11 — Observability & Evaluation (MLOps / LLMOps / AIOps)

**Goal:** See inside the system and measure quality. This covers "MLOps/LLMOps/AIOps" and "AI development best practices for experimentation and production" — your other named gap.

### Concept first

Traditional apps: a 200-OK means success. **LLM apps are different** — a fast, successful HTTP response can still contain a wrong or hallucinated answer. So you need **LLM observability**: trace every step (prompt → retrieval → tool calls → response), capture tokens/latency/cost, and **evaluate quality** (faithfulness, relevance, hallucination). **Langfuse** is the leading open-source, self-hostable tool for exactly this.

The MLOps/LLMOps mindset to articulate: _experiment → measure → version prompts → deploy → monitor in production → catch quality drift → improve → repeat._ It's a loop, not a launch.

### Steps

1. **Run Langfuse via Docker** (self-hosted, open source).
2. **Instrument your app** — wrap LLM and retrieval calls (Langfuse has decorators/SDKs and integrates with LangChain/LangGraph). Now every request produces a **trace** you can inspect: exact prompt, retrieved chunks, tool calls, tokens, latency, cost.
3. **Build a tiny eval set:** 10–15 question/expected-answer pairs from your documents.
4. **Run automated evaluation:** use an **LLM-as-a-judge** to score answers for faithfulness/relevance, log the scores to Langfuse.
5. **Use prompt management:** version your prompts in Langfuse so you can change them without redeploying — and roll back if quality drops.

> **Interview hook:** "I treat LLMOps as a loop: Langfuse traces every prompt, retrieval, and tool call with token/latency/cost, and I run LLM-as-a-judge evals on a fixed test set to catch quality regressions before users do. Prompts are versioned so I can roll back. A 200-OK isn't success for an LLM app — _faithfulness_ is."

---

# PHASE 12 — CI/CD + Security

**Goal:** Automate build/test/deploy and scan for vulnerabilities. Covers "CI/CD pipelines (Jenkins, GitLab)" and the "security standards (encryption, access control, vulnerability management)" requirement.

### Concept first

- **CI (Continuous Integration):** every git push automatically runs tests + builds the container images. No more "it broke and nobody noticed."
- **CD (Continuous Delivery/Deployment):** automatically ships those images (to a registry, then to the cluster).
- **The pipeline is just YAML** describing steps. GitHub Actions, GitLab CI, and Jenkins all express the same idea differently — understand the _concept_ and you can map it to any of them.
- **Security in the pipeline ("shift left"):** scan images for known vulnerabilities (**Trivy**) _before_ they ship, manage secrets properly, enforce access control.

### Steps

1. **Write `.github/workflows/ci.yml`** with jobs that, on every push:
   - Check out code, install deps, **run your tests** (write a couple of simple ones).
   - **Build the Docker images.**
   - **Run Trivy** to scan the images for vulnerabilities — fail the build on critical findings.
   - (Optional) push images to a registry (GitHub Container Registry).
2. **Know the GitLab/Jenkins mapping:** the same stages live in a `.gitlab-ci.yml` (`stages: [test, build, scan, deploy]`) or a Jenkins `Jenkinsfile`. Be able to say "same pipeline, different syntax."
3. **Write your security one-pager** (in `docs/`), covering the three pillars the JD names:
   - **Encryption:** TLS in transit (HTTPS/Ingress), encryption at rest for the vector DB volume.
   - **Access control:** API-key/JWT auth on the backend, Kubernetes Secrets and RBAC.
   - **Vulnerability management:** Trivy image scanning in CI, dependency updates, non-root containers, minimal base images.

> **Interview hook:** "CI runs tests, builds images, and Trivy-scans them on every push — security shifts left into the pipeline. The same stages map cleanly to GitLab CI or a Jenkinsfile. Secrets live in Kubernetes Secrets, traffic is TLS, containers run non-root. That's encryption, access control, and vuln management — the three pillars in the JD."

---

# BONUS — NVIDIA / GPU Serving (Talking Points You Don't Have to Run)

The JD lists NVIDIA (GPUs, CUDA, TensorRT) as _an advantage_, not a requirement — and you can't realistically run it on a laptop. So **learn to discuss it intelligently** instead. Be able to say:

- **Why GPUs:** LLM inference is matrix math; GPUs parallelize it massively. CUDA is NVIDIA's programming layer that frameworks build on.
- **vLLM:** the leading open-source inference _server_ for LLMs. Its key innovation is **PagedAttention** — it manages the KV-cache like virtual memory, eliminating waste, so you serve far more requests per GPU. OpenAI-compatible API, so it drops into the same provider interface you built in Phase 5.
- **TensorRT-LLM:** NVIDIA's compiler that optimizes a model for a _specific_ GPU for maximum throughput/lowest latency — faster than vLLM at peak, but with model-specific compilation and tighter vendor lock-in.
- **The honest tradeoff:** "For self-hosting open models at scale on NVIDIA hardware, I'd serve with vLLM for flexibility, or TensorRT-LLM when I need to squeeze maximum throughput out of fixed GPUs. On Kubernetes you'd add the NVIDIA GPU Operator and request `nvidia.com/gpu` resources in the pod spec."

That paragraph alone will make you sound like you've thought about production GPU serving — which you now have.

---

# Your Interview-Day Cheat Sheet

When they ask "tell me about a project," walk the architecture **in this order** (it mirrors a request flowing through the system):

1. **Problem:** "Enterprises have documents — including images and scans — that LLMs can't answer about and tend to hallucinate on."
2. **Ingestion:** chunking → multimodal (VLM for images) → embeddings (swappable Gemini/local) → Qdrant.
3. **Query path:** it's an **agent** (LangGraph), not a fixed chain — it decides when to retrieve, summarize, or clarify.
4. **Models:** **provider-abstracted** — local **SLM via Ollama** for cheap/private, **cloud LLM** for hard reasoning, routed by cost/sensitivity. (Mention Bedrock/HF/vLLM as the production options.)
5. **Serving:** FastAPI backend + React frontend with **source citations**.
6. **Deployment:** **containerized** (Docker), running on **Kubernetes** (Deployments/Services/Secrets/probes), portable across cloud and on-prem.
7. **LLMOps:** **Langfuse** traces + **LLM-as-judge evals** catch quality drift; prompts versioned.
8. **CI/CD + security:** **GitHub Actions** builds, tests, and **Trivy-scans** images; secrets, TLS, non-root, RBAC.

### Honesty that builds trust

You have real experience with LLMs, SLMs, Python, and data science, and _less_ with Kubernetes/MLOps/CI/CD. The winning move is **not** to pretend. Say: _"I'm strongest in the model and Python layers. To go deeper on the infra side I built this end-to-end project — containerized it, deployed it to a local Kubernetes cluster, added Langfuse observability and a GitHub Actions pipeline with Trivy scanning. Here's what I learned and what I'd do differently in production."_ Interviewers trust **"I learned it by building it"** far more than a memorized buzzword. Your week of hands-on work _is_ your answer.

---

# If You're Truly Short on Time

Do these four phases deeply and you'll still interview well: **Phase 2 (RAG)**, **Phase 6 (Agent)**, **Phase 9 (Docker)**, **Phase 10 (Kubernetes)**. Then _read and understand_ the concept sections of Phases 11–12 even if you don't fully build them. Understanding > completeness, every time.

You've got this. Build it, break it, watch it heal, and tell that story.
