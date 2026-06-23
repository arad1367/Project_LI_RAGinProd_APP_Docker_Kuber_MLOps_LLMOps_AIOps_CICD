import os
import time
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv
load_dotenv()

VLM_MODEL = "gemini-3.5-flash"   # current GA vision model

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client

PROMPT = (
    "You are reading a document image. Transcribe ALL visible text exactly. "
    "Then describe any charts, tables, or diagrams in detail, including specific "
    "numbers, labels, axis names, and trends. Be thorough and factual; do not "
    "guess at anything you cannot see."
)


def _mime_for(path):
    return "image/png" if str(path).lower().endswith(".png") else "image/jpeg"


def describe_image(path):
    """Use a VLM to 'read' an image and return a detailed text description."""
    with open(path, "rb") as f:
        image_bytes = f.read()

    client = _get_client()
    while True:  # retry on rate limit
        try:
            resp = client.models.generate_content(
                model=VLM_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=_mime_for(path)),
                    types.Part.from_text(text=PROMPT),
                ],
            )
            return resp.text
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                print("  VLM rate limited — waiting 30s...")
                time.sleep(30)
            else:
                raise


# Self-test: `python backend/vlm.py docs/Crypto.png`
if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "docs/Crypto.png"
    print(describe_image(p))