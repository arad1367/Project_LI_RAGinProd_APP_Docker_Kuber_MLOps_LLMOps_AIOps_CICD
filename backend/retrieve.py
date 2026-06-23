import sys

from qdrant_client import QdrantClient

from config import QDRANT_URL, COLLECTION_NAME
from embeddings import embed_texts, prepare_query

client = QdrantClient(url=QDRANT_URL)


def retrieve(query, k=5):
    query_vector = embed_texts(prepare_query(query))[0]
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
    )
    hits = []
    for point in result.points:
        p = point.payload
        hits.append({
            "score": point.score,
            "type": p.get("type", "text"),
            "text": p.get("text", ""),
            "source": p.get("source"),
            "page": p.get("page"),
            "image_path": p.get("image_path"),
        })
    return hits


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is this document about?"
    print(f"Question: {question}\n")
    for h in retrieve(question):
        page = f" p{h['page']}" if h["page"] else ""
        tag = h["type"].upper()
        print(f"[{h['score']:.3f}] ({tag}) {h['source']}{page}")
        if h["type"] == "image":
            print(f"    -> matched the IMAGE itself: {h['image_path']}\n")
        else:
            print(f"    {h['text'][:200]}...\n")