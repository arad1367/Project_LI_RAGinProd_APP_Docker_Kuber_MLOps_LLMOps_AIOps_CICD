import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Configurable so the container can point at the 'qdrant' service by name;
# falls back to localhost when you run uvicorn directly on your host.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documind"
EMBED_DIM = 3072