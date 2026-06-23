import sys

from retrieve import retrieve
from llm import chat, choose_provider

SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Be concise and mention which source filenames you used."
)


def build_prompt(question, hits):
    blocks = []
    for h in hits:
        loc = h["source"] + (f" p{h['page']}" if h["page"] else "")
        if h["type"] == "image":
            blocks.append(f"[{loc}] (an image matched this query)")
        else:
            blocks.append(f"[{loc}]\n{h['text']}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def rag_answer(question, provider="local", k=5):
    hits = retrieve(question, k=k)
    prompt = build_prompt(question, hits)
    answer = chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        provider=provider,
    )
    return answer, hits


if __name__ == "__main__":
    # Usage:
    #   python backend/rag.py what is the refund policy            (defaults to local)
    #   python backend/rag.py gemini what is the refund policy     (force a provider)
    #   python backend/rag.py smart why did revenue change         (auto-route)
    args = sys.argv[1:]
    provider = "local"
    if args and args[0] in ("local", "local-tiny", "gemini", "openai", "smart"):
        provider = args[0]
        args = args[1:]

    question = " ".join(args) or "What is this document about?"

    if provider == "smart":
        provider = choose_provider(question)
        print(f"(smart routing chose: {provider})")

    print(f"Q: {question}\n")
    answer, hits = rag_answer(question, provider=provider)
    print(f"ANSWER (via {provider}):\n{answer}\n")
    print("SOURCES USED:")
    for h in hits:
        page = f" p{h['page']}" if h["page"] else ""
        print(f"  - {h['source']}{page}  ({h['type']}, score {h['score']:.3f})")