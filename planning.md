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

Chunk size:2048 tokens

Overlap:410

Reasoning: Since we are doing fixed size chunking here, we need to have overlapping context so that the model doesnt miss any relevant context.


---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

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

1.

2.

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
  │  (Claude claude-sonnet- │
  │   4-6 via Anthropic API)│
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

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
