"""
Pipeline stages 1 & 2: Document Ingestion → Chunking

Sources   : documents/URLs.txt  (10 supplement brand product catalog pages)
Chunk size: 2048 tokens → ~8192 chars  (RecursiveCharacterTextSplitter is char-based;
Overlap   : 410  tokens → ~1640 chars   converted at ~4 chars / token)
Output    : documents/raw/<brand>.txt      — cleaned plain text per source
            documents/chunks/<brand>.json  — chunk list + metadata

Ingestion strategy
------------------
Most sources are Shopify stores.  Their catalog pages render product listings
via JavaScript, so requests + BeautifulSoup only returns the page shell (~3-8 k chars).
Shopify exposes a static JSON API at  /products.json?limit=250&page=N  that returns
full product data (title, description, tags, variants) with no JS required.

  Shopify stores  → fetch /products.json, paginate, format as text blocks
  Other stores    → fetch HTML, strip noise tags, extract visible text
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
URLS_FILE = Path("documents/URLs.txt")
RAW_DIR   = Path("documents/raw")
CHUNKS_DIR = Path("documents/chunks")

# ---------------------------------------------------------------------------
# Chunking constants  (from planning.md Chunking Strategy)
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 8192   # 2048 tokens × ~4 chars/token
CHUNK_OVERLAP = 1640   # 410  tokens × ~4 chars/token

# ---------------------------------------------------------------------------
# HTTP settings
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 30
POLITE_DELAY    = 1.5   # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NOISE_TAGS = {
    "script", "style", "noscript", "nav", "footer", "header",
    "aside", "svg", "img", "iframe", "form", "button",
}

# Shopify stores identified from the URL list.
# They expose /products.json regardless of the catalog sub-path in the URL.
SHOPIFY_HOSTS = {
    "countrylifevitamins.com",
    "megafood.com",
    "solaray.com",
    "jarrow.com",
    "sourceofnature.eu",
    "naturesplus.com",
    "gardenoflife.com",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def brand_slug(url: str) -> str:
    """'www.solgar.com' → 'solgar',  'countrylifevitamins.com' → 'countrylifevitamins'"""
    host = url.split("//")[-1].split("/")[0]
    parts = [p for p in host.split(".") if p not in ("www", "com", "eu", "net", "org")]
    return parts[0] if parts else host


def origin(url: str) -> str:
    """'https://foo.com/bar' → 'https://foo.com'"""
    parts = url.split("//")
    scheme = parts[0]
    host = parts[1].split("/")[0]
    return f"{scheme}//{host}"


def is_shopify(url: str) -> bool:
    host = url.split("//")[-1].split("/")[0].lstrip("www.")
    return any(host == s or host.endswith("." + s) for s in SHOPIFY_HOSTS)


def strip_html(raw: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Stage 1a: Shopify /products.json ingestion
# ---------------------------------------------------------------------------

def fetch_shopify_products(base_url: str) -> list[dict]:
    """Paginate through /products.json and return all product dicts."""
    products = []
    page = 1
    while True:
        url = f"{base_url}/products.json?limit=250&page={page}"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        batch = resp.json().get("products", [])
        if not batch:
            break
        products.extend(batch)
        if len(batch) < 250:
            break
        page += 1
        time.sleep(0.5)
    return products


def shopify_products_to_text(brand: str, products: list[dict]) -> str:
    """
    Format each product as a short text block so chunk boundaries
    fall between products rather than splitting one product entry.
    """
    blocks = []
    for p in products:
        title = p.get("title", "").strip()
        desc  = strip_html(p.get("body_html") or "").strip()
        tags  = ", ".join(p.get("tags", [])) if p.get("tags") else ""
        types = p.get("product_type", "").strip()
        variants = p.get("variants", [])
        options = "; ".join(
            f"{v.get('title','')} — ${v.get('price','')}"
            for v in variants[:3]            # first 3 variants is enough
        )

        lines = [f"Brand: {brand}", f"Product: {title}"]
        if types:
            lines.append(f"Type: {types}")
        if tags:
            lines.append(f"Tags: {tags}")
        if desc:
            lines.append(f"Description: {desc}")
        if options:
            lines.append(f"Variants: {options}")
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Stage 1b: Plain HTML ingestion  (non-Shopify stores)
# ---------------------------------------------------------------------------

def fetch_html_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


# ---------------------------------------------------------------------------
# Stage 2: Chunking
# ---------------------------------------------------------------------------

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n---\n\n", "\n\n", "\n", ". ", " ", ""],
)


def make_chunks(text: str) -> list[str]:
    return SPLITTER.split_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    urls = [
        line.strip()
        for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    total_chunks = 0
    failed = []

    for url in urls:
        brand = brand_slug(url)
        print(f"[{brand}]", end=" ", flush=True)

        try:
            if is_shopify(url):
                print("Shopify /products.json ...", end=" ", flush=True)
                base = origin(url)
                products = fetch_shopify_products(base)
                text = shopify_products_to_text(brand, products)
                print(f"{len(products)} products →", end=" ", flush=True)
            else:
                print("HTML scrape ...", end=" ", flush=True)
                text = fetch_html_text(url)

        except Exception as exc:
            print(f"FAILED — {exc}")
            failed.append((brand, url, str(exc)))
            continue

        if not text.strip():
            print("EMPTY — skipping")
            failed.append((brand, url, "empty content"))
            continue

        RAW_DIR.joinpath(f"{brand}.txt").write_text(text, encoding="utf-8")

        chunks = make_chunks(text)
        total_chunks += len(chunks)

        record = {
            "source": brand,
            "url": url,
            "char_count": len(text),
            "chunk_count": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunks": chunks,
        }
        CHUNKS_DIR.joinpath(f"{brand}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"{len(chunks)} chunks  ({len(text):,} chars)")
        time.sleep(POLITE_DELAY)

    print(f"\n{'='*55}")
    print(f"Total chunks : {total_chunks}")
    print(f"Sources OK   : {len(urls) - len(failed)} / {len(urls)}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for b, u, err in failed:
            print(f"  {b}: {err}")

    # Chunk-count health check
    print()
    if total_chunks < 50:
        print("WARNING: fewer than 50 chunks — content may still be too thin.")
        print("  Consider reducing CHUNK_SIZE or verifying more sources fetched correctly.")
    elif total_chunks > 2000:
        print("WARNING: more than 2,000 chunks — chunks may be too small.")
        print("  Consider increasing CHUNK_SIZE to carry more meaning per embedding.")
    else:
        print(f"OK: {total_chunks} chunks is within the healthy 50–2,000 range.")


if __name__ == "__main__":
    main()
