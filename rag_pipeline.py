"""
Core RAG pipeline: Retrieval + Grounded Generation

Stages 4 & 5 of the pipeline:
  ChromaDB similarity search  →  Groq LLM (llama-3.3-70b-versatile)

Grounding contract
------------------
The system prompt instructs the model to answer ONLY from the retrieved
context chunks.  If the answer is not present in those chunks the model
must say so — it must not fall back on general supplement knowledge.
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION  = "supplements"
TOP_K       = 5
GROQ_MODEL  = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are a supplement product assistant.  Your ONLY source of knowledge is the
product catalog excerpts provided in each message — you have no other information.

Rules you must follow without exception:
1. Answer using ONLY facts stated in the provided excerpts.
2. If the excerpts do not contain enough information to answer, respond with:
   "The catalog data I have doesn't include that information."
3. Do NOT use general supplement knowledge, personal opinions, or anything
   not explicitly stated in the excerpts.
4. Keep answers concise — 2 to 4 sentences unless a list is clearer.
"""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        ef = DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            _collection = client.get_collection(name=COLLECTION, embedding_function=ef)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ChromaDB collection '{COLLECTION}'. "
                "Run embed_and_store.py first."
            ) from exc
    return _collection


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return top-k chunks as list of {text, source, url, chunk_index, distance}."""
    results = get_collection().query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":        doc,
            "source":      meta.get("source", "unknown"),
            "url":         meta.get("url", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "distance":    dist,
        })
    return chunks


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
_groq_client: Groq | None = None


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered excerpts for the LLM."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} — {c['source']}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def generate(query: str, chunks: list[dict]) -> str:
    """Call Groq with retrieved context and return the grounded answer."""
    context = _format_context(chunks)
    user_message = (
        f"Product catalog excerpts:\n\n{context}\n\n"
        f"Question: {query}"
    )
    response = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,   # deterministic — no hallucination drift
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def ask(query: str, k: int = TOP_K) -> dict:
    """
    Full RAG round-trip.
    Returns {answer, chunks} so callers can display sources separately.
    """
    chunks = retrieve(query, k=k)
    answer = generate(query, chunks)
    return {"answer": answer, "chunks": chunks}
