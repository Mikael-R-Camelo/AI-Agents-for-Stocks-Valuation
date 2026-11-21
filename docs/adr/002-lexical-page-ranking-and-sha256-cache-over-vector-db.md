# ADR-002: Lexical Page Ranking and Post-Validation SHA-256 Cache Over a Vector DB

- **Status:** Accepted
- **Date:** 2026-11-21
- **Affected Components:** `agents/extrator.py`, `graph/nodes.py`, `config.py`, `main.py`

## Context

Each ticker analysis processes 1 to 4 quarterly PDF filings (`1T`–`4T`) per fiscal year. In practice, input documents fall into two profiles:

1. **Compact Earnings Releases (Press Releases):** Typically 2–30 pages (e.g., `reports/BBDC3/` filings are 2 pages, ~14–17k characters each), fitting comfortably inside modern LLM context windows.
2. **Full IFRS Financial Statements:** Occasionally 80–200+ pages (>150k characters), where key tables (Balance Sheet, Cash Flow Statement, and Share Capital Notes containing `numero_total_acoes`) frequently appear in the second half of the document.

Additionally, re-running the batch pipeline on a new trading day to refresh market prices previously re-parsed and re-extracted immutable historical PDFs, wasting tokens and latency.

A naive instinct in LLM projects is to introduce a vector database (RAG) for PDF ingestion. However, every added infrastructure component increases operational complexity, failure modes, and retrieval variance.

## Decision

Avoid introducing a Vector DB for this single-ticker quarterly extraction workflow, and instead implement a **two-layer lightweight data strategy**:

1. **Smart Lexical Page Ranking (`selecionar_texto_relevante_pdf` in `agents/extrator.py`):**
   - If the full extracted PDF text fits within `LIMITE_CARACTERES_DOCUMENTO` (`150,000` chars), pass it intact with zero information loss.
   - Only when a PDF exceeds the limit, score pages using weighted accounting/financial keywords (DRE, recurring net income, equity, ROE/ROAE, share capital/share count, DFC/cash, management guidance, numeric table density, plus a cover/highlights bonus), greedily select the highest-scoring pages within budget, and **re-sort selected pages into their original document order**.
2. **Post-Validation SHA-256 Local JSON Cache (`agents/extrator.py` + `graph/nodes.py`):**
   - Key cache entries by `SHA-256(pdf_bytes) + VERSAO_SCHEMA_EXTRATOR + gemini_model`, validated via Pydantic (`EntradaCacheExtracao`).
   - **Strict persistence invariant:** Only write cache entries inside `validador_node` **after** a PDF passes all mathematical checks with zero errors—never cache failed or unverified extractions.
   - Allow deterministic bypass via `USAR_CACHE_PDF=false` or CLI flag `--no-cache`, and always bypass cache during corrective reflection retries.

## Alternatives Considered

- **Vector Database + Embeddings (Chroma/Qdrant/pgvector):** Rejected for this use case. Quarterly release extraction needs a fixed structured schema (`DadosTrimestraisPDF`) from 1–4 self-contained files rather than open-ended search across thousands of documents. Chunking dense financial tables into embeddings often splits table headers from numeric columns, hurting extraction accuracy while adding embedding API costs and infra setup.
- **Blind Prefix Truncation (`texto[:150000]`):** Rejected (replaced). Silently drops Balance Sheet, Cash Flow, and Share Count tables located near the end of long IFRS reports.
- **Caching raw LLM output immediately inside `extrair_dados_do_pdf`:** Rejected. Would permanently poison the cache with hallucinated or mis-scaled numbers whenever validation fails.

## Trade-offs & Consequences

- **Pros:**
  - Zero external database dependencies (`pip install -r requirements.txt` works out of the box).
  - Full-document fidelity for normal Press Releases; high-signal preservation of late-document tables for long IFRS reports.
  - Near-zero latency and zero extraction token cost on repeated runs for immutable historical quarters, with automatic invalidation on PDF edits, schema bumps, or model switches.
- **Cons:**
  - Lexical keyword scoring is tailored to Portuguese corporate filings and does not perform cross-document semantic search across multi-year archives or dozens of companies simultaneously.
  - Cache is local to disk (`.cache/extrator/`), not shared across distributed workers.

## Revisit Triggers

- Migrate to **Hybrid RAG (BM25 + dense vector search + reranking)** with structure-aware table chunking if the product scope expands to free-form Q&A across multi-year filing archives, earnings call transcripts, and regulatory notes spanning hundreds of documents.
- Replace local JSON files with object storage (S3/GCS) or Redis if deployed as a multi-worker cloud service.
