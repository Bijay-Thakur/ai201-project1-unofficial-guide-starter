"""
Supplement RAG — interactive CLI interface  (Pipeline stage 5)

Usage:
    python app.py

Commands inside the session:
    <question>   Ask anything about the 10 supplement brands in the catalog
    /sources     Toggle showing retrieved chunk previews after each answer
    /quit        Exit
"""

import textwrap

from rag_pipeline import ask, retrieve

BANNER = """
============================================================
  Supplement Catalog Assistant
  Powered by ChromaDB + Groq (llama-3.3-70b-versatile)
  Answers are grounded in scraped product catalog data only.
  Type /quit to exit   |   /sources to toggle chunk display
============================================================
"""

WRAP_WIDTH  = 80
PREVIEW_LEN = 200
show_sources = False   # toggled by /sources


def _attribution_footer(chunks: list[dict]) -> str:
    """
    Build a verified source list from the actual retrieved chunks.
    Deduplicates by brand so each source appears once, ordered by best distance.
    """
    seen: dict[str, str] = {}   # brand -> url (first/closest occurrence)
    for c in chunks:
        brand = c["source"]
        if brand not in seen:
            seen[brand] = c["url"]
    parts = [f"{brand} <{url}>" for brand, url in seen.items()]
    return "  Retrieved from: " + " | ".join(parts)


def print_answer(result: dict) -> None:
    answer = result["answer"].encode("ascii", errors="replace").decode("ascii")
    print()
    for line in textwrap.wrap(answer, width=WRAP_WIDTH):
        print(" ", line)

    # Always show verified attribution — built from chunk metadata, not LLM output
    print()
    footer = _attribution_footer(result["chunks"])
    print(footer.encode("ascii", errors="replace").decode("ascii"))

    if show_sources:
        print()
        print("  --- Retrieved chunks ---")
        for i, c in enumerate(result["chunks"], start=1):
            preview = textwrap.shorten(c["text"], width=PREVIEW_LEN, placeholder=" ...")
            preview = preview.encode("ascii", errors="replace").decode("ascii")
            print(f"  [{i}] {c['source']}  dist={c['distance']:.3f}")
            print(f"      {preview}")


def main() -> None:
    global show_sources

    print(BANNER)

    # Warm up ChromaDB + embedding model on first query
    print("  Loading vector store ...", end=" ", flush=True)
    try:
        retrieve("test", k=1)   # triggers lazy collection load
        print("ready.\n")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        return

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye.")
            break

        if query.lower() == "/sources":
            show_sources = not show_sources
            state = "ON" if show_sources else "OFF"
            print(f"  [chunk display {state}]")
            continue

        print("  thinking ...", end="\r", flush=True)
        try:
            result = ask(query)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        print(" " * 20, end="\r")   # clear "thinking ..." line
        print_answer(result)
        print()


if __name__ == "__main__":
    main()
