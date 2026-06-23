import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

from config import ROOT, QDRANT_URL, COLLECTION_NAME, EMBED_DIM
from embeddings import embed_texts, embed_image, prepare_document
from vlm import describe_image

client = QdrantClient(url=QDRANT_URL)

STATE_FILE = ROOT / "backend" / "ingested.json"

TEXT_EXTS = {".txt", ".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def read_pdf(path):
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        yield i + 1, (page.extract_text() or "")


def file_fingerprint(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'.")


def text_chunks_for_file(path):
    """Return text chunks for a .txt or .pdf file."""
    suffix = path.suffix.lower()
    out = []
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        for c in chunk_text(text):
            out.append({"text": c, "source": path.name, "page": None, "type": "text"})
    elif suffix == ".pdf":
        for page_num, page_text in read_pdf(path):
            if not page_text.strip():
                print(f"  (skipping image-only page {page_num} of {path.name})")
                continue
            for c in chunk_text(page_text):
                out.append({"text": c, "source": path.name, "page": page_num, "type": "text"})
    return out


def delete_existing_points_for(source_name):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_name))]
        ),
    )


def next_start_id():
    return client.count(collection_name=COLLECTION_NAME, exact=True).count


def main():
    docs_dir = ROOT / "docs"
    ensure_collection()
    state = load_state()

    # 1) Which files are new or changed?
    to_process = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in (TEXT_EXTS | IMAGE_EXTS):
            print(f"  (skipping {path.name} — unsupported type)")
            continue
        fp = file_fingerprint(path)
        if state.get(path.name) == fp:
            print(f"  (unchanged, skipping) {path.name}")
            continue
        to_process.append((path, fp))

    if not to_process:
        print("Nothing new to embed. Everything is already up to date.")
        return

    # 2) Clean up old vectors for any file that changed.
    for path, fp in to_process:
        if path.name in state:
            delete_existing_points_for(path.name)

    # 3) Build items. Text items get embedded as text; images get BOTH a
    #    VLM description (text) and a direct image embedding.
    text_items = []     # embedded via embed_texts
    image_items = []    # embedded via embed_image (one per image)

    for path, fp in to_process:
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTS:
            text_items.extend(text_chunks_for_file(path))
        elif suffix in IMAGE_EXTS:
            print(f"  Reading image {path.name} with the VLM...")
            description = describe_image(path)
            # (a) the VLM description, embedded as text
            text_items.append({
                "text": description, "source": path.name, "page": None,
                "type": "image_caption", "image_path": str(path),
            })
            # (b) the raw image, embedded directly into the shared space
            image_items.append({
                "text": f"[image] {path.name}", "source": path.name, "page": None,
                "type": "image", "image_path": str(path),
            })

    # 4) Embed everything.
    print(f"Embedding {len(text_items)} text item(s) and {len(image_items)} image(s)...")
    text_vectors = embed_texts([prepare_document(c["text"]) for c in text_items]) if text_items else []
    image_vectors = [embed_image(p["image_path"]) for p in image_items]

    # 5) Combine and store.
    all_items = text_items + image_items
    all_vectors = text_vectors + image_vectors
    start_id = next_start_id()
    points = [
        PointStruct(id=start_id + i, vector=all_vectors[i], payload=all_items[i])
        for i in range(len(all_items))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    # 6) Record fingerprints.
    for path, fp in to_process:
        state[path.name] = fp
    save_state(state)

    print(f"Done. Added {len(points)} points. Files tracked: {len(state)}.")


if __name__ == "__main__":
    main()