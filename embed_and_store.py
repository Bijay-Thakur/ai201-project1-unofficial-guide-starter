"""
Pipeline stage 3: Embedding + Vector Store

Loads every chunk from documents/chunks/*.json, embeds with all-MiniLM-L6-v2
(via ChromaDB's built-in ONNX runtime — no PyTorch required), and persists to
a ChromaDB collection at ./chroma_db.

Run this after ingest_and_chunk.py has populated documents/chunks/.
Re-running recreates the collection from scratch so the store stays in sync.
"""

import json
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ---------------------------------------------------------------------------
# Config  (mirrors planning.md Retrieval Approach)
# ---------------------------------------------------------------------------
CHUNKS_DIR  = Path("documents/chunks")
CHROMA_PATH = "./chroma_db"
COLLECTION  = "supplements"
STORE_BATCH = 100   # docs per ChromaDB add call


# ---------------------------------------------------------------------------
# Load all chunks from disk
# ---------------------------------------------------------------------------

def load_all_chunks() -> tuple[list[str], list[dict], list[str]]:
    """Return (documents, metadatas, ids) across all brand JSON files."""
    documents, metadatas, ids = [], [], []

    json_files = sorted(CHUNKS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {CHUNKS_DIR}. "
            "Run ingest_and_chunk.py first."
        )

    for path in json_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        brand = record["source"]
        url   = record["url"]
        for i, chunk in enumerate(record["chunks"]):
            documents.append(chunk)
            metadatas.append({"source": brand, "url": url, "chunk_index": i})
            ids.append(f"{brand}_{i}")

    return documents, metadatas, ids


# ---------------------------------------------------------------------------
# Build / rebuild the ChromaDB collection
# ---------------------------------------------------------------------------

def build_collection(
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> chromadb.Collection:

    # DefaultEmbeddingFunction uses ChromaDB's bundled ONNX model (all-MiniLM-L6-v2)
    # — no PyTorch needed, no memory-mapping, no page-file issues on Windows.
    ef = DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION)
        print(f"Dropped existing '{COLLECTION}' collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(documents)
    for start in range(0, total, STORE_BATCH):
        end = min(start + STORE_BATCH, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Embedded and stored {end}/{total} chunks ...", end="\r", flush=True)

    print()
    return collection


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading chunks from {CHUNKS_DIR} ...")
    documents, metadatas, ids = load_all_chunks()

    counts = Counter(m["source"] for m in metadatas)
    print(f"\nSources found ({len(counts)}):")
    for brand, n in sorted(counts.items()):
        print(f"  {brand:30s}  {n} chunks")
    print(f"\nTotal: {len(documents)} chunks across {len(counts)} sources\n")

    print("Embedding with all-MiniLM-L6-v2 (ONNX) and storing in ChromaDB ...")
    collection = build_collection(documents, metadatas, ids)

    stored = collection.count()
    print(f"\nDone.  {stored} vectors in the '{COLLECTION}' collection.")

    if stored < 50:
        print("WARNING: fewer than 50 vectors — re-check ingest output.")
    elif stored > 2000:
        print("NOTE: more than 2,000 vectors — consider increasing chunk size.")
    else:
        print(f"OK: {stored} is within the healthy 50–2,000 range.")


if __name__ == "__main__":
    main()
