import os
import shutil

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from config import ROOT
from agent import build_agent
import ingest as ingest_module


# ----- Authentication (access control) -----
API_KEY = os.environ.get("APP_API_KEY", "dev-secret-key")

# --- Langfuse tracing (optional; activates only if keys are present) ---
LANGFUSE_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
langfuse_handler = None
if LANGFUSE_ON:
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()
    print("Langfuse tracing enabled.")

def require_api_key(x_api_key: str = Header(default=None)):
    """A FastAPI dependency: rejects requests without the correct X-API-Key header."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

# ----- App setup -----
app = FastAPI(title="DocuMind API")

# CORS: allow the React frontend (Phase 8) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the agent ONCE at startup and reuse it for every request.
AGENT = build_agent("openai")
DOCS_DIR = ROOT / "docs"


class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default"   # different ids = separate conversation memories


@app.get("/")
def root():
    return {"service": "DocuMind", "endpoints": ["/health", "/chat", "/ingest", "/reindex"]}


@app.get("/health")
def health():
    """Kubernetes liveness/readiness target. Must stay public (no auth)."""
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    """Stream the agent's answer back token by token."""
    config = {"configurable": {"thread_id": req.thread_id}}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {"langfuse_session_id": req.thread_id}

    def token_stream():
        # stream_mode="messages" yields (token, metadata) as the model generates.
        # We forward only tokens from the 'assistant' node that carry text — so
        # tool-call planning (empty text) and raw tool output are not streamed,
        # only the final answer reaches the user.
        for token, meta in AGENT.stream(
            {"messages": [HumanMessage(content=req.question)]},
            config,
            stream_mode="messages",
        ):
            if meta.get("langgraph_node") == "assistant" and getattr(token, "content", ""):
                yield token.content

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest_file(file: UploadFile = File(...)):
    """Upload a document, save it to docs/, and embed it (incremental)."""
    dest = DOCS_DIR / file.filename
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    ingest_module.main()   # only the new file gets embedded
    return {"status": "ingested", "filename": file.filename}


@app.post("/reindex", dependencies=[Depends(require_api_key)])
def reindex():
    """Re-scan docs/ and embed anything new or changed."""
    ingest_module.main()
    return {"status": "reindexed"}
