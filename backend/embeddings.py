import os
import time
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv
load_dotenv()

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
    """Return ONE vector per input string (rate-limit aware)."""
    if isinstance(texts, str):
        texts = [texts]

    client = _get_client()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        contents = [
            types.Content(parts=[types.Part.from_text(text=t)])
            for t in batch
        ]
        while True:
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=contents,
                )
                vectors.extend(e.values for e in result.embeddings)
                break
            except errors.ClientError as e:
                if getattr(e, "code", None) == 429:
                    print("  Rate limited — waiting 30s before retrying this batch...")
                    time.sleep(30)
                else:
                    raise

        done = min(i + batch_size, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks...")
        if done < len(texts):
            time.sleep(pause_seconds)
    return vectors


def embed_image(path):
    """
    Embed an image FILE directly into the same 3072-dim space as text.
    A single image input returns a single image vector, so a text query
    can retrieve it directly (cross-modal search) — no transcription needed.
    """
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        image_bytes = f.read()

    client = _get_client()
    while True:
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime)],
            )
            return result.embeddings[0].values
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                print("  Image embed rate limited — waiting 30s...")
                time.sleep(30)
            else:
                raise


# Connectivity self-test: `python backend/embeddings.py`
if __name__ == "__main__":
    v = embed_texts("hello world")[0]
    print(f"Got an embedding with {len(v)} dimensions.")