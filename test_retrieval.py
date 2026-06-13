"""
Pipeline stage 4: Retrieval test  (no generation)

Runs each of the 5 evaluation-plan questions against the ChromaDB collection
and prints the top-k results so you can judge retrieval quality before wiring
in the LLM.  Fill in the README Evaluation Report table after reviewing this output.

Distance metric: cosine (lower = more similar; 0.0 = identical, 2.0 = opposite).
A result below ~0.4 is typically a good match for domain-specific catalog queries.

Usage:
    python test_retrieval.py
    python test_retrieval.py --query "your custom query here"
"""

import sys
import textwrap

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ---------------------------------------------------------------------------
# Config  (mirrors planning.md Retrieval Approach)
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION  = "supplements"
TOP_K       = 5          # number of chunks to retrieve per query
PREVIEW_LEN = 300        # characters of chunk text shown in output

# ---------------------------------------------------------------------------
# Evaluation plan questions  (from planning.md)
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    "Which brands in the catalog sell a fish oil supplement?",
    "Does Solgar offer a Vitamin D3 supplement, and if so what form does it come in (softgel, tablet, liquid)?",
    "Which brands offer a magnesium supplement, and what forms of magnesium do they use (glycinate, citrate, oxide)?",
    "Does MegaFood sell a B-complex product?",
    "Which brand offers the widest variety of probiotic products based on the catalog pages?",
]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def load_collection() -> chromadb.Collection:
    # DefaultEmbeddingFunction uses ChromaDB's bundled ONNX model — no PyTorch.
    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION, embedding_function=ef)


def retrieve(collection: chromadb.Collection, query: str) -> dict:
    return collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

DIVIDER = "-" * 70

def print_results(query: str, results: dict, query_num: int | None = None) -> None:
    label = f"Query {query_num}: " if query_num else ""
    print(f"\n{DIVIDER}")
    print(f"{label}{query}")
    print(DIVIDER)

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        print("  (no results)")
        return

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        source  = meta.get("source", "?")
        url     = meta.get("url", "")
        ci      = meta.get("chunk_index", "?")
        preview = textwrap.shorten(doc, width=PREVIEW_LEN, placeholder=" ...")
        # Strip non-ASCII so Windows CP1252 console doesn't error on symbols
        preview = preview.encode("ascii", errors="replace").decode("ascii")

        print(f"\n  [{rank}] source={source}  chunk={ci}  distance={dist:.4f}")
        print(f"      url: {url}")
        print(f"      {preview}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    custom_query = None
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        if idx + 1 < len(sys.argv):
            custom_query = sys.argv[idx + 1]

    print(f"Connecting to ChromaDB at '{CHROMA_PATH}' ...")
    try:
        collection = load_collection()
    except Exception as exc:
        print(f"ERROR: could not load collection '{COLLECTION}': {exc}")
        print("Run embed_and_store.py first.")
        sys.exit(1)

    print(f"Collection '{COLLECTION}' has {collection.count()} vectors.")
    print(f"Embedding model: all-MiniLM-L6-v2 (ONNX)   top-k: {TOP_K}")

    if custom_query:
        results = retrieve(collection, custom_query)
        print_results(custom_query, results)
    else:
        for i, question in enumerate(EVAL_QUESTIONS, start=1):
            results = retrieve(collection, question)
            print_results(question, results, query_num=i)

    print(f"\n{DIVIDER}")
    print("Retrieval test complete.")
    print(
        "Review distances above:\n"
        "  < 0.30  — strong match\n"
        "  0.30–0.50 — relevant\n"
        "  > 0.50  — weak / off-target\n"
        "Record source names and quality in the README Evaluation Report table."
    )


if __name__ == "__main__":
    main()
