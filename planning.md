# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
I choose popular supplements company in US. Almost all of them have similar kinds of products, so it will be easier for consumer to compare between different brands even if they are looking for one particular product, for example: fish oil.
---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Solgar | Product catalog (website) | <https://www.solgar.com/solgar-products> |
| 2 | Life Extension | Product catalog (website) | <https://www.lifeextension.com/vitamins-supplements/essentials> |
| 3 | Garden of Life | Product catalog (website) | <https://www.gardenoflife.com/products> |
| 4 | Country Life Vitamins | Product catalog (website) | <https://countrylifevitamins.com/collections/all> |
| 5 | MegaFood | Product catalog (website) | <https://megafood.com/collections/all> |
| 6 | Solaray | Product catalog (website) | <https://solaray.com/collections/all-products> |
| 7 | NOW Foods | Product catalog (website) | <https://www.nowfoods.com/products/supplements/all-products> |
| 8 | Source of Nature | Product catalog (website) | <https://sourceofnature.eu/collections/all-products> |
| 9 | Nature's Plus | Product catalog (website) | <https://naturesplus.com/collections/source-of-life> |
| 10 | Jarrow Formulas | Product catalog (website) | <https://jarrow.com/collections/all> |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

Chunk size: 2048 tokens (~8,192 chars — RecursiveCharacterTextSplitter is char-based, converted at ~4 chars/token)

Overlap: 410 tokens (~1,640 chars)

Reasoning: Since we are doing fixed size chunking here, we need to have overlapping context so that the model doesnt miss any relevant context.

**Ingestion note:** Most sources are Shopify stores. Their catalog pages render product listings via JavaScript, so requests + BeautifulSoup only returns the page shell (~3–8 k chars) — not the actual products. The updated ingest_and_chunk.py targets `/products.json?limit=250&page=N` for Shopify hosts instead, which returns full product data (title, description, tags, variants) as static JSON without JavaScript. Non-Shopify sources (solgar, lifeextension, nowfoods) fall back to HTML scraping.


---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers. Loaded by ChromaDB's `SentenceTransformerEmbeddingFunction`; vectors stored with cosine similarity (`hnsw:space: cosine`).

**Top-k:** 5

**Production tradeoff reflection:** Since the products of these companies are not too many, using a high-capability API-hosted model (e.g. text-embedding-3-large) would be overkill on cost. all-MiniLM-L6-v2 runs locally, is fast, and handles English product catalog text well. For a real deployment the main tradeoffs to weigh would be: (1) context length — all-MiniLM tops out at 256 tokens, so very long product descriptions get truncated; a model with a 512+ token window (e.g. all-mpnet-base-v2) would be safer; (2) domain specificity — a fine-tuned biomedical or retail embedding model might score closer matches on supplement jargon; (3) latency — local inference is fast at this corpus size but would need batching or a hosted endpoint under concurrent load.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Which brands in the catalog sell a fish oil supplement? | Should name at least two specific brands from the 10 sources (e.g., Solgar, NOW Foods, Jarrow Formulas) — not a generic "many brands do." |
| 2 | Does Solgar offer a Vitamin D3 supplement, and if so what form does it come in (e.g., softgel, tablet, liquid)? | Yes — Solgar lists a Vitamin D3 product; the expected form is softgel. Answer is wrong if it says "unknown" or names a different brand. |
| 3 | Which brands offer a magnesium supplement, and what forms of magnesium do they use (e.g., glycinate, citrate, oxide)? | Should identify at least two brands and name at least two distinct magnesium forms pulled from the catalog pages — not generic chemistry knowledge. |
| 4 | Does MegaFood sell a B-complex product? | Yes or No, verifiable directly from the MegaFood catalog page. The answer is wrong if the system hedges without checking the source. |
| 5 | Which brand offers the widest variety of probiotic products based on the catalog pages scraped? | Should name a specific brand (e.g., Garden of Life or Jarrow Formulas) with a count or list of probiotic SKUs as justification — not a guess or a general statement about probiotics. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **JavaScript-rendered catalog pages.** Most of the 10 sources are Shopify stores that inject product listings via client-side JavaScript. A plain `requests` + BeautifulSoup scraper will only see the page shell — roughly 3–8 KB of navigation and marketing text with no product data. This means the ingestion stage could silently produce near-empty chunks without raising an error, and retrieval would fail quietly at query time. Mitigation: detect Shopify hosts (check for `cdn.shopify.com` in HTML or a `Shopify` header) and switch to the `/products.json` API endpoint, which returns full product JSON without JavaScript.

2. **Missing sources creating silent corpus gaps.** Three of the 10 planned sources (Solgar, Garden of Life, NOW Foods) use non-Shopify platforms that return minimal or JavaScript-gated HTML. If these sources fail silently, the vector store will simply have no chunks for those brands and the system will retrieve from unrelated brands instead — without telling the user why. The grounding rule ("if the excerpt doesn't contain the answer, say so") prevents hallucination, but the user may not realize that the missing information is a scraper failure rather than a genuine absence of products.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

```text
  ┌─────────────────────────┐
  │   Document Ingestion    │
  │  (requests / BeautifulSoup)  │
  │  10 supplement brand    │
  │  product catalog pages  │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │        Chunking         │
  │  (LangChain             │
  │   RecursiveCharacter-   │
  │   TextSplitter)         │
  │  size=2048 / overlap=410│
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  Embedding + Vector     │
  │       Store             │
  │  Embedding:             │
  │  all-MiniLM-L6-v2       │
  │  (sentence-transformers)│
  │  Store: ChromaDB        │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │        Retrieval        │
  │  (ChromaDB similarity   │
  │   search, top-k=5)      │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │       Generation        │
  │  (Groq API —            │
  │   llama-3.3-70b-        │
  │   versatile)            │
  │  grounded to retrieved  │
  │  chunks only            │
  └─────────────────────────┘
```

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

Used Claude with the Documents table, Chunking Strategy section, and Architecture diagram as input. Asked it to implement a script that loads the 10 URLs, fetches HTML, strips noise tags, and splits with LangChain RecursiveCharacterTextSplitter at size=8192/overlap=1640. It produced `ingest_and_chunk.py`. I then verified it by checking the output JSON files in `documents/chunks/` and confirming chunk counts were in the expected range. When the initial run returned only 1 chunk per Shopify-based source (JS rendering issue), I directed Claude to switch to the Shopify `/products.json` API endpoint, which returned full product data without JavaScript.

**Milestone 4 — Embedding and retrieval:**

Used Claude with the Retrieval Approach section and Architecture diagram. Asked it to implement `embed_and_store.py` (ChromaDB + all-MiniLM-L6-v2) and `test_retrieval.py` (runs the 5 eval questions, prints distances). I verified by running `test_retrieval.py` and checking that distances were below 0.50 for most eval questions. Hit a Windows page-file/memory-mapping crash when using the PyTorch-based `SentenceTransformerEmbeddingFunction`; directed Claude to switch to ChromaDB's built-in ONNX `DefaultEmbeddingFunction`, which resolved the issue.

**Milestone 5 — Generation and interface:**

Used Claude with the full pipeline diagram, the Groq API key setup in `.env`, and the grounding requirement. Asked it to implement `rag_pipeline.py` (retrieval + Groq generation with a strict grounding system prompt) and `app.py` (CLI loop with `/sources` toggle). Verified by piping a test question through `app.py` and confirming the answer cited only brands present in the retrieved chunks, with a `Sources:` line at the end.
