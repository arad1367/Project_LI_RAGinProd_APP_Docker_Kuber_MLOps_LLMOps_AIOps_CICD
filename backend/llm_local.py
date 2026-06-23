from openai import OpenAI

# If you pulled a different model in Step 2, change this name to match.
OLLAMA_MODEL = "llama3.1:8b"

# The OpenAI SDK pointed at Ollama. api_key is required by the SDK but ignored
# by Ollama (it's local), so any non-empty string works.
_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2):
    """Send chat messages to the local SLM and return its text reply."""
    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,   # low temp = more focused, less random
    )
    return resp.choices[0].message.content


# Self-test: `python backend/llm_local.py`
if __name__ == "__main__":
    reply = ollama_chat([{"role": "user", "content": "Say hello in one sentence."}])
    print(reply)