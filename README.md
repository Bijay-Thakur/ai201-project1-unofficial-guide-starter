# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

This system covers the product catalogs of 10 popular US supplement brands — including NaturesPlus, Jarrow Formulas, MegaFood, Solaray, Country Life Vitamins, and Source of Nature. It lets a user ask cross-brand comparison questions ("which brands sell a magnesium glycinate supplement?", "does MegaFood make a B-complex?") and get a grounded answer drawn directly from scraped catalog data.

This knowledge is valuable because supplement shoppers often need to compare products across brands, but each brand's website only shows its own catalog. Review aggregators like iHerb and Amazon mix products from all brands but rank by sales and ads, not by ingredient form or brand coverage. There is no neutral, structured source that answers "which of these 10 brands carries X?" in one place — this RAG system fills that gap by indexing all 10 catalogs into a single queryable vector store.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

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

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** 8,192 characters (~2,048 tokens at approximately 4 characters per token). LangChain's `RecursiveCharacterTextSplitter` operates in characters, so the token target from planning.md was converted at this ratio.

**Overlap:** 1,640 characters (~410 tokens). Each chunk shares its last ~410 tokens with the start of the next chunk, so a product entry that straddles a boundary still appears in full in at least one chunk.

**Why these choices fit your documents:** Most of the sources are Shopify product catalogs fetched via `/products.json`. Each product is formatted as a short structured text block (Brand / Product / Tags / Description / Variants), roughly 150–400 characters. A chunk of ~8,000 characters holds approximately 20–40 product entries at once, which is appropriate for comparison queries like "which brands sell fish oil" — a single chunk can contain several brands' entries together. The overlap protects against a product description being split right at the chunk boundary, since a 410-token overlap is larger than any single product block.

**Preprocessing:** For Shopify sources, the HTML layer was bypassed entirely — products were fetched as JSON from `/products.json?limit=250&page=N` and formatted into plain text blocks. For non-Shopify sources, BeautifulSoup stripped all noise tags (`script`, `style`, `nav`, `footer`, `header`, `svg`, `iframe`) before extracting visible text, and all whitespace was collapsed to single spaces. No stemming, lowercasing, or stopword removal was applied before chunking.

**Final chunk count:** 163 chunks across 7 sources. Three of the 10 planned sources (Solgar, Garden of Life, NOW Foods) did not produce chunk files because their pages returned empty or near-empty content — see Failure Case Analysis.

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** `all-MiniLM-L6-v2`, served through ChromaDB's built-in ONNX runtime (`DefaultEmbeddingFunction`). The model was chosen because it runs entirely locally (no API key or internet connection needed at query time), is fast on CPU via ONNX, produces 384-dimensional vectors, and performs well on short English-language product text. Using ChromaDB's ONNX wrapper rather than the PyTorch-based `sentence-transformers` library was a practical necessity on this machine — the PyTorch loader uses memory-mapped `safetensors` files, which triggered a Windows virtual-memory error (OS error 1455) when the system page file was too small.

**Production tradeoff reflection:** For real users, the main tradeoffs to evaluate would be: (1) **Context length** — `all-MiniLM-L6-v2` truncates inputs at 256 tokens, which is shorter than some detailed product descriptions; `all-mpnet-base-v2` handles 512 tokens and would reduce truncation losses. (2) **Domain specificity** — a model fine-tuned on retail or biomedical text (e.g., `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`) would embed supplement terminology more precisely and improve recall for ingredient-level queries. (3) **Multilingual support** — if the catalog expands to non-English brands, `multilingual-e5-base` would be needed. (4) **Latency vs. quality** — at this corpus size (163 vectors), local ONNX inference is fast; at 1M+ vectors with concurrent users, an API-hosted model like `text-embedding-3-small` with server-side batching would be more practical, at the cost of per-query pricing.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**

The model receives this system prompt on every request (from `rag_pipeline.py`):

```text
You are a supplement product assistant.  Your ONLY source of knowledge is the
product catalog excerpts provided in each message — you have no other information.

Rules you must follow without exception:
1. Answer using ONLY facts stated in the provided excerpts.
2. If the excerpts do not contain enough information to answer, respond with:
   "The catalog data I have doesn't include that information."
3. Do NOT use general supplement knowledge, personal opinions, or anything
   not explicitly stated in the excerpts.
4. Keep answers concise — 2 to 4 sentences unless a list is clearer.
```

The context window also contains the top-5 retrieved chunks formatted as numbered excerpts with brand labels (`[Excerpt 1 — naturesplus]`), so the model can trace which brand each fact comes from. Temperature is set to 0.0 to eliminate random drift from the grounded answer.

**How source attribution is surfaced in the response:**

Source attribution is **programmatic, not model-generated**. After every answer, `app.py` builds a `Retrieved from:` footer directly from the `chunks` list returned by ChromaDB — deduplicating by brand and showing the source URL. This footer is derived from chunk metadata, so it cannot be hallucinated or omitted by the LLM. The model is explicitly not asked to produce a Sources line; instead the verified ground-truth list always appears below the answer. A `/sources` toggle is also available to print the full chunk text and cosine distance for each retrieved excerpt.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Which brands in the catalog sell a fish oil supplement? | At least two specific brands named (e.g. Solgar, NOW Foods, Jarrow) | Named 4 brands: NaturesPlus, MegaFood, Jarrow, Country Life Vitamins — grounded in retrieved chunks (e.g. Country Life DHA Gummies mentions "500 mg of fish oils per serving") | Relevant | Partially accurate — correct for the 7 brands in corpus; Solgar, Garden of Life, and NOW Foods were never scraped so the answer is incomplete |
| 2 | Does Solgar offer a Vitamin D3 supplement, and if so what form (softgel, tablet, liquid)? | Yes — Solgar Vitamin D3, softgel form | "The catalog data I have doesn't include that information." Retrieved chunks were from NaturesPlus and Country Life only | Off-target | Inaccurate — correct answer is Yes/softgel, but Solgar was never successfully ingested so the corpus has no Solgar data; the model correctly refused to guess, which is grounding working as intended |
| 3 | Which brands offer a magnesium supplement, and what forms do they use? | At least two brands, at least two distinct magnesium forms from the catalog | Named 3 brands: Source of Nature (Oxide, Citrate, Taurate, Bisglycinate, Malate, Aspartate via 6-in-1), NaturesPlus (Citrate), Solaray (Citrate) | Relevant | Accurate — multiple brands and specific forms all grounded in retrieved product data |
| 4 | Does MegaFood sell a B-complex product? | Yes or No, directly from MegaFood catalog | "Yes — MegaFood sells Balanced B Complex, made with all eight B vitamins including Methyl Folate and Methyl B12, delivered with an organic food blend of brown rice, kale, and broccoli." | Relevant | Accurate — correct yes answer with specific product name and details pulled from the MegaFood chunk |
| 5 | Which brand offers the widest variety of probiotic products? | Specific brand named with a count or list of probiotic SKUs | Named MegaFood with 5 specific products: MegaFlora Women's, MegaFlora with Turmeric, MegaFlora Plus, MegaFlora Kids, Women's Prebiotic + Probiotic Gummies | Relevant | Accurate — specific brand named with a list of SKUs; comparison is limited to 7 brands in corpus but the reasoning is grounded |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** "Does Solgar offer a Vitamin D3 supplement, and if so what form does it come in (softgel, tablet, liquid)?"

**What the system returned:** "The catalog data I have doesn't include that information." The retrieved chunks came from NaturesPlus (Vitamin D3 softgels) and Country Life — both relevant by topic, but neither is Solgar. The programmatic attribution footer confirmed the sources were `naturesplus` and `countrylifevitamins`, not Solgar.

**Root cause (tied to a specific pipeline stage):** The failure originates at **Stage 1 — Document Ingestion**, not at retrieval or generation. Solgar's website (`solgar.com`) is not a Shopify store, so the scraper fell back to HTML extraction. The page returned only navigation chrome and marketing text with no machine-readable product listings — approximately 2,000 characters of content, none of it product data. Because no usable text was extracted, no chunks were written to `documents/chunks/solgar.json`, and no Solgar vectors were ever stored in ChromaDB. When the Vitamin D3 query ran, the embedding model returned the semantically nearest neighbors it had — NaturesPlus and Country Life Vitamin D3 chunks — which are topically correct but brand-wrong. The model then correctly applied the grounding rule and refused to answer, because no Solgar-branded text appeared anywhere in the top-5 chunks. The failure is a corpus gap, not a retrieval error or a hallucination.

**What you would change to fix it:** Solgar's product data would need a dedicated scraper. Two options: (1) check whether Solgar exposes an undocumented product JSON endpoint (some non-Shopify stores do), or (2) use a JavaScript-capable headless browser like Playwright to fully render the page before extracting text. A third option is to find a structured data source for Solgar — for example, their products appear on iHerb and Amazon with structured listings that can be scraped more reliably. Once Solgar chunks exist in the vector store, the same embedding and retrieval code requires no changes.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:** The Architecture diagram in planning.md gave a clear stage-by-stage breakdown with tool names attached to each box. When the ingestion stage failed for Shopify stores (returning near-empty HTML), the diagram made it immediately obvious that this was an isolated Stage 1 problem — the chunking, embedding, and retrieval stages were unchanged. Without that structure I might have tried to compensate downstream (e.g., reducing chunk size or tuning retrieval) when the real fix was entirely in the ingestion function. The diagram also made it easy to hand each stage to an AI tool as a scoped prompt, because each box had a defined input and output.

**One way your implementation diverged from the spec, and why:** The spec specified `all-MiniLM-L6-v2` via `sentence-transformers` (PyTorch-based) as the embedding model, but the actual implementation uses the same model through ChromaDB's built-in ONNX runtime (`DefaultEmbeddingFunction`). The reason is a Windows-specific constraint: the PyTorch `safetensors` loader memory-maps model weights into virtual address space at load time, and on a machine with a nearly-full disk and a small page file this triggers OS error 1455 and kills the process. The ONNX runtime loads the model sequentially without memory mapping, which worked without any configuration changes. The vectors produced are identical (384 dimensions, same model weights) so there is no accuracy difference — only the runtime differs.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

### Instance 1 — Ingestion and chunking code

- *What I gave the AI:* The Documents table (10 URLs), the Chunking Strategy section (size 2048 tokens, overlap 410 tokens), and the Architecture diagram showing requests/BeautifulSoup for ingestion and LangChain RecursiveCharacterTextSplitter for chunking.
- *What it produced:* `ingest_and_chunk.py` — a script that fetched each URL with requests, stripped noise tags with BeautifulSoup, and split the resulting text with RecursiveCharacterTextSplitter at the specified size and overlap.
- *What I changed or overrode:* The initial version used HTML scraping for all 10 sources. When I ran it, most Shopify-based sources (MegaFood, Solaray, etc.) returned only 2,000–8,000 characters of page shell with no product data, because their catalog listings are injected by JavaScript at runtime. I directed the AI to detect Shopify hosts and switch to fetching `/products.json?limit=250&page=N` — a static JSON API that all Shopify stores expose — which returned full product data without JavaScript. I also directed it to format each product as a named text block (Brand / Product / Tags / Description / Variants) so chunk boundaries would fall between products rather than splitting a single product entry.

### Instance 2 — Embedding, storage, and retrieval code

- *What I gave the AI:* The Retrieval Approach section (all-MiniLM-L6-v2, ChromaDB, top-k=5), the Architecture diagram (cosine similarity, HNSW index), and the 5 evaluation questions from planning.md.
- *What it produced:* `embed_and_store.py` using ChromaDB's `SentenceTransformerEmbeddingFunction` (PyTorch-based wrapper), and `test_retrieval.py` that queried the collection with `query_texts` and printed top-k results with distances.
- *What I changed or overrode:* The `SentenceTransformerEmbeddingFunction` crashed with OS error 1455 (Windows paging file too small) when trying to load the model via `safetensors` memory mapping. I directed the AI to switch to `DefaultEmbeddingFunction` — ChromaDB's built-in ONNX runtime for the same model — which loads weights sequentially and does not require virtual memory backing. I also directed it to change `query_texts` to `query_embeddings` in the retrieval call to match, and to replace Unicode box-drawing characters in the output with plain ASCII so the Windows CP1252 console would not throw a `UnicodeEncodeError`.
