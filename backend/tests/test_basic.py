import os
# Set a dummy key so importing modules that construct model clients won't fail.
os.environ.setdefault("OPENAI_API_KEY", "dummy-for-import")

from ingest import chunk_text
from embeddings import prepare_query, prepare_document
from agent import calculator


def test_chunk_text_overlaps():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 100 for c in chunks)
    # The last 20 words of chunk 0 should reappear at the start of chunk 1.
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_calculator_tool():
    assert calculator.invoke({"expression": "2 + 2 * 5"}) == "12"
    assert calculator.invoke({"expression": "(3 + 4) * 2"}) == "14"


def test_embedding_prefixes():
    assert prepare_query("hello").startswith("task: search result")
    assert prepare_document("hello").startswith("title:")