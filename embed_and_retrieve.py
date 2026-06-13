import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = BASE_DIR / "documents" / "chunks"
VECTOR_DIR = BASE_DIR / "documents" / "vectors"
VECTOR_COLLECTION_NAME = "supplements_catalog"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 5


def load_chunk_records(chunks_dir: Path) -> List[Dict[str, Any]]:
    """Load all chunk JSON records and flatten them into one record-per-chunk list."""
    chunk_files = sorted(chunks_dir.glob("*.json"))
    records: List[Dict[str, Any]] = []

    for chunk_file in chunk_files:
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        source = payload.get("source", chunk_file.stem)
        url = payload.get("url", "")
        chunks = payload.get("chunks", [])

        for idx, text in enumerate(chunks, start=1):
            records.append({
                "id": f"{source}-{idx}",
                "source": source,
                "url": url,
                "chunk_index": idx,
                "text": text,
                "chunk_count": len(chunks),
            })

    return records


class SentenceTransformerEmbedding:
    """A wrapper so Chroma can call sentence-transformers embeddings."""

    def __init__(self, model_name: str):
        try:
            self.model = SentenceTransformer(
                model_name,
                device="cpu",
                model_kwargs={
                    "dtype": "float16",
                    "low_cpu_mem_usage": True,
                },
            )
        except OSError as first_exc:
            fallback_model = "paraphrase-albert-small-v2"
            print(
                f"Warning: failed to load {model_name} due to memory limits; "
                f"falling back to {fallback_model}.",
                file=sys.stderr,
            )
            self.model = SentenceTransformer(
                fallback_model,
                device="cpu",
                model_kwargs={
                    "dtype": "float16",
                    "low_cpu_mem_usage": True,
                },
            )

    def __call__(self, texts: Iterable[str]) -> List[List[float]]:
        return self.model.encode(
            list(texts),
            convert_to_numpy=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()


def get_chroma_client(persist_directory: Path) -> chromadb.Client:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.Client(Settings(
        persist_directory=str(persist_directory),
        is_persistent=True,
    ))


def build_vector_store(
    chunks_dir: Path,
    vector_dir: Path,
    model_name: str,
    collection_name: str,
) -> Tuple[chromadb.api.models.Collection.Collection, int]:
    records = load_chunk_records(chunks_dir)
    if not records:
        raise ValueError(f"No chunk records found in {chunks_dir}")

    embedding_fn = SentenceTransformerEmbedding(model_name)
    client = get_chroma_client(vector_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    ids = [record["id"] for record in records]
    documents = [record["text"] for record in records]
    metadatas = [
        {
            "source": record["source"],
            "url": record["url"],
            "chunk_index": record["chunk_index"],
            "chunk_count": record["chunk_count"],
        }
        for record in records
    ]

    # Replace any existing collection contents for a fresh index.
    if collection.count() > 0:
        print(f"Clearing existing collection '{collection_name}' before reindexing...")
        collection.delete()  # delete the existing collection
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    client.persist()
    return collection, len(records)


def retrieve(
    collection: chromadb.api.models.Collection.Collection,
    query: str,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances'],
    )

    hits = []
    for idx, doc in enumerate(results["documents"][0]):
        hits.append({
            "rank": idx + 1,
            "document": doc,
            "metadata": results["metadatas"][0][idx],
            "distance": results["distances"][0][idx],
        })
    return hits


def test_retrieval(collection: chromadb.api.models.Collection.Collection) -> None:
    sample_queries = [
        "Which brands in the catalog sell a fish oil supplement?",
        "Does Solgar offer a Vitamin D3 supplement, and if so what form does it come in?",
        "Which brands offer a magnesium supplement and what forms do they use?",
    ]

    for query in sample_queries:
        print(f"\nQuery: {query}")
        hits = retrieve(collection, query, top_k=TOP_K_DEFAULT)
        for hit in hits:
            metadata = hit["metadata"]
            print(
                f"Rank {hit['rank']} | source={metadata.get('source')} "
                f"| chunk={metadata.get('chunk_index')}/{metadata.get('chunk_count')} "
                f"| distance={hit['distance']:.4f}"
            )
            print(hit["document"][:400].replace('\n', ' ').strip())
            print("---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Chroma vector store from document chunks and run retrieval tests."
    )
    parser.add_argument(
        "command",
        choices=["build", "query", "test"],
        help="Action to perform: build the vector store, query it, or run sample retrieval tests.",
    )
    parser.add_argument("--query", "-q", help="Query text to search against the vector store.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help="Number of results to return for the query.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNKS_DIR,
        help="Directory containing chunk JSON files.",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=VECTOR_DIR,
        help="Directory for persistent Chroma storage.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="Sentence Transformers embedding model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        print("Building embedding store from chunk files...")
        collection, count = build_vector_store(
            chunks_dir=args.chunks_dir,
            vector_dir=args.vector_dir,
            model_name=args.model,
            collection_name=VECTOR_COLLECTION_NAME,
        )
        print(f"Built vector store with {count} chunks in collection '{collection.name}'.")

    else:
        client = get_chroma_client(args.vector_dir)
        collection = client.get_collection(VECTOR_COLLECTION_NAME)
        if collection is None:
            raise RuntimeError(
                f"Collection '{VECTOR_COLLECTION_NAME}' not found. Run 'build' first."
            )

        if args.command == "query":
            if not args.query:
                raise ValueError("--query is required for the query command.")
            hits = retrieve(collection, args.query, top_k=args.top_k)
            for hit in hits:
                metadata = hit["metadata"]
                print(
                    f"Rank {hit['rank']} | source={metadata.get('source')} "
                    f"| chunk={metadata.get('chunk_index')}/{metadata.get('chunk_count')} "
                    f"| distance={hit['distance']:.4f}"
                )
                print(hit["document"])
                print("\n" + "=" * 80 + "\n")
        elif args.command == "test":
            test_retrieval(collection)


if __name__ == "__main__":
    main()
