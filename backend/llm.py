import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # needed when this file is imported or run directly

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# --- Provider registry ---
# Every provider below speaks the OpenAI chat format, so ONE client shape
# handles all of them — only base_url, key, and model differ. Switching
# providers is configuration, not code.
PROVIDERS = {
    "local": {
        "base_url": OLLAMA_BASE,
        "api_key": "ollama",
        "model": "llama3.1:8b",
    },
    "local-tiny": {
        "base_url": OLLAMA_BASE,
        "api_key": "ollama",
        "model": "qwen3.5:0.8b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-3.5-flash",         # cloud frontier (free tier)
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",                  # set to a model your account has; mini/nano are cheaper
    },
}

_clients = {}

def _client_for(provider):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(PROVIDERS)}")
    if provider not in _clients:
        cfg = PROVIDERS[provider]
        api_key = cfg.get("api_key") or os.environ[cfg["api_key_env"]]
        _clients[provider] = OpenAI(base_url=cfg["base_url"], api_key=api_key)
    return _clients[provider]


def chat(messages, provider="local", model=None, temperature=None):
    """
    Send chat messages to ANY provider through one interface.

    `messages` is the standard list of {"role": ..., "content": ...} dicts.
    We only pass temperature if explicitly set, because some cloud reasoning
    models reject a custom temperature — leaving it unset uses each model's
    sensible default and keeps the interface universal.
    """
    client = _client_for(provider)
    model = model or PROVIDERS[provider]["model"]

    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def choose_provider(question):
    """
    Illustrative routing: send simple questions to the local SLM (free, private,
    fast) and harder/analytical ones to the cloud frontier model (stronger
    reasoning). In production you'd route on more signals, but the idea is the
    same — match model capability and cost to the task.
    """
    q = question.lower()
    looks_hard = (
        len(question.split()) > 25
        or any(w in q for w in ["why", "compare", "analyze", "trade-off", "explain in detail"])
    )
    return "gemini" if looks_hard else "local"

# Self-test: `python backend/llm.py`
if __name__ == "__main__":
    msg = [{"role": "user", "content": "Say hello in one short sentence."}]
    print("LOCAL :", chat(msg, provider="local"))
    print("GEMINI:", chat(msg, provider="gemini"))