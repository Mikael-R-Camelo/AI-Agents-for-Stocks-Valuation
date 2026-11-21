# 📈 AI Agents for Stocks Valuation

Autonomous multi-agent pipeline built in Python with **LangGraph**, **Pydantic**, and **Google Gemini** for fundamental equity analysis and automated valuation of Brazilian B3 stocks from quarterly earnings filings (*Press Releases*) and live market data.

The architecture pairs LLM contextual document reading with **deterministic Python financial engineering**, applying the **Reflection (Self-Correction)** pattern across two independent audit gates: per-PDF accounting validation before valuation and a deterministic post-generation compliance guardrail before report persistence.

> **Engineering Design Note:** Core architectural trade-offs—zero LLM math, lexical page ranking + SHA-256 caching instead of a Vector DB, and clause-scoped compliance auditing—are documented in the [Architecture Decision Records (`docs/adr/`)](docs/adr/README.md).

---

## 🏗️ System Architecture (LangGraph)

The pipeline runs as a typed `StateGraph` (`AgentState`) orchestrating extraction, accounting verification, deterministic financial tool execution, and compliance verification:

```text
  [ Start / Batch Folder Scan ]
                 │
                 ▼ (Resolve Sector via Yahoo Finance)
         (Node) EXTRATOR  <──────────────────────────────────────────────┐
                 │                                                       │
                 ▼ (Structured Pydantic JSON per PDF)                    │
        (Node) VALIDADOR (Pure Python)                                   │
                 │                                                       │
                 ▼                                                       │
  (Conditional Edge) ROTEADOR ──(Re-extract failing PDFs only, < 3)──────┘
                 │
         (Chronologically Sorted Validated History)
                 │
                 ▼
       (Node) CALCULADORA (Pure Python — Zero LLM Tokens)
         ├─ Runs yfinance + projections + multiples + 2-Stage DDM
         ├─ Logs tabular summary & saves Valuation_<TICKER>_quant.json
                 │
                 ▼
         (Node) ANALISTA (Single-Shot LLM Synthesis)
         └─ Transient 503/429 -> optional GEMINI_FALLBACK_MODEL -> deterministic MD fallback
                 │
                 ▼
       (Node) COMPLIANCE (Pure Python) ──(Violation / <= 2 & LLM up)──► back to ANALISTA
                 │
          (Approved, Fallback, or Explicit Alert Banner)
                 │
                 ▼
   [ Save Valuation_<TICKER>.md ]
```

### Graph Components

1. **Dynamic Sector Resolver (`tools/utils.py` + `agents/sector_protocols.py`):** Queries Yahoo Finance (`yfinance`) using an in-memory per-execution `Ticker` cache to map the company's sector to its internal analysis protocol (`banco`, `energia`/`eletrica`, `holding`, `commodities`, `imobiliario`, `varejo`, `saude`, `industria`, or `geral`).
2. **Extractor Node (`extrator_node`):** Reads quarterly PDF filings (`1T`–`4T`) via `PyPDFLoader` and enforces structured output through **Pydantic** (`DadosTrimestraisPDF`). Cash-flow statement (DFC) fields (`caixa_inicial`, `caixa_final`, `variacao_liquida_caixa`, `efeito_cambial_caixa`) are optional (`float | None`) to support bank earnings releases that omit full DFC tables. On retries, it **surgically reprocesses only failing PDFs**, injecting exact mathematical error feedback into the prompt ([extrator_prompt.py](agents/prompts/extrator_prompt.py)).
3. **Validator Node (`validador_node`) & Reflection Loop:** Zero-token deterministic Python audit running over every extracted quarter:
   * **Structural Integrity:** Validates `numero_total_acoes > 0`, `patrimonio_liquido != 0`, and decimal scale bounds (`roe_reportado`, `payout_estimado`).
   * **Annualized ROE Cross-Check:** Annualizes quarterly ROE ($\frac{\text{Quarterly Net Income}}{\text{Equity}}$ under both compound and linear conventions) and verifies convergence against reported annualized `roe_reportado`.
   * **Conditional Cash Flow Reconciliation:** Runs whenever DFC fields are present, validating $\text{Beginning Cash} + \text{Net Change} + \text{FX Effect} \approx \text{Ending Cash}$.
   * **True Chronological Ordering:** Sorts consolidated quarters by `(year, quarter)` via `chave_ordenacao_trimestre`, preserving accuracy across fiscal year boundaries (e.g., `4T25 -> 1T26 -> 2T26`).
   * **Bounded Retries & Safe Caching:** Increments `tentativas_extracao` on each pass (capped at `MAX_TENTATIVAS_EXTRACAO = 3`) and persists entries to the local SHA-256 cache **strictly after validation passes**.
4. **Deterministic Calculator Node (`calculadora_node`):** Pure-Python node (zero LLM tokens) that classifies `momento_operacional` from the chronological earnings trajectory, invokes all market-data, multiples, and 2-Stage DDM functions into a typed Pydantic `ResultadoQuantitativo`, logs a clean tabular summary to the terminal, and persists `reports/<TICKER>/Valuation_<TICKER>_quant.json` before invoking the Analyst LLM.
5. **Single-Shot Analyst Node (`analista_node`) & Graceful Fallback:** Loads the dedicated system prompt from [analista_prompt.py](agents/prompts/analista_prompt.py) and performs a **single LLM call** (no ReAct tool loop) to synthesize the qualitative narrative around the pre-computed numbers. On transient API errors (`503`, `429`, timeouts), it optionally tries `GEMINI_FALLBACK_MODEL` (when explicitly configured) and otherwise degrades gracefully via `gerar_relatorio_fallback_quantitativo`, delivering a complete quantitative Markdown report without aborting the pipeline.
6. **Compliance Node (`compliance_node`):** Deterministic post-generation guardrail ([compliance.py](agents/compliance.py)) that audits the final report prior to saving `Valuation_<TICKER>.md` (and never loops back to an unavailable LLM in fallback mode).

---

## ⚙️ Financial Engineering & Deterministic Tools (`@tool`)

To eliminate LLM arithmetic hallucinations, **100% of financial calculations execute in typed, unit-tested Python tools**:

* **Dual-Scenario Annualized Earnings (`projetar_lucro_anual_cenarios`):** Projects remaining quarters for in-progress fiscal years (e.g., 1Q–3Q reported) across three operational regimes (`recuperacao`, `estavel`, `deteriorando`), applying proper downside stress to both profits and losses while enforcing the invariant `lucro_anual_conservador <= lucro_anual_mediano`:
  * **Conservative Scenario:** Stresses remaining quarters using the worst reported quarter (or YTD mean during turnaround), deepening negative values during losses.
  * **Median Scenario:** Uses the YTD average, latest-quarter run-rate, or worst quarter under deterioration.
* **Historical Anchoring & B3 ON/PN Safeguards (`yfinance`):**
  * `buscar_dados_mercado_yfinance`: Fetches **exclusively** current share price (`preco_atual`, with `fast_info` fallback) and currency on B3. **Structural safeguard:** intentionally omits `sharesOutstanding` and `bookValue` from the payload to prevent ON/PN share-class distortions—`numero_total_acoes` and `patrimonio_liquido` are sourced strictly from official PDF filings.
  * `buscar_medias_historicas_yfinance`: Computes **4-year historical average ROE** across Income Statement and Balance Sheet with IFRS accounting line-item aliases.
  * `buscar_payout_medio_historico_yfinance`: Computes **4-year historical average Payout** across Income Statement and Cash Flow (summing ON+PN dividends when split), filters out non-recurring distortions `> 150%`, applies an **automatic fallback to `info["payoutRatio"]`** when cash-flow dividends are zero/missing (common for Brazilian banks distributing via **JCP**), and explicitly returns both `payout_medio_observado` and `payout_sustentavel` (capped at `1.0` for DDM compatibility).
* **Multiples & Valuation (`calcular_multiplos_basicos` & `fluxo_de_caixa_futuro_buffet`):**
  * Computes EPS (`LPA`), BVPS (`VPA`), P/E (`P_L`), P/B (`P_VP`), and implied ROE—returning `None` for non-applicable multiples under losses or negative equity.
  * Implements a **Consistent 2-Stage Dividend Discount Model (DDM)** with strict numeric/domain guards and centralized defaults in `config.py`:
    * **Stage 1 (Years 1 to $N$):** Projects earnings growth at sustainable rate $g_{\text{sust}} = (1 - \text{payout}) \times \text{ROE}$ and discounts effective shareholder dividends ($\text{Earnings}_t \times \text{payout}$).
    * **Stage 2 (Terminal Value / Perpetuity):** Assumes competitive equilibrium ($\text{ROE}_{\text{terminal}} = K_e$), applying maturity sustainable payout $\text{payout}_{\text{terminal}} = 1 - \frac{g_{\text{perpetual}}}{K_e}$ in Gordon Growth to avoid artificial value destruction.
    * **Default Parameters:** Cost of Equity **$K_e = 14.5\%$** (`TAXA_DESCONTO_PADRAO = 0.145`) and Perpetual Growth **$g_{\text{perpetual}} = 4.0\%$** (`CRESCIMENTO_PERPETUO_PADRAO = 0.04`), yielding $\text{payout}_{\text{terminal}} = 72.41\%$.

---

## 🧠 Valuation Methodologies

* **Consistent 2-Stage DDM (Implemented):** Projects sustainable earnings retention over a 4-year explicit horizon, discounts distributed dividends at $K_e = 14.5\%$, and applies competitive-convergence terminal value at $g_{\text{perpetual}} = 4.0\%$.
* *(Roadmap)* **Dividend Yield Ceiling (Décio Bazin Method):** Leverages `buscar_payout_medio_historico_yfinance` (including JCP fallback) to estimate price ceilings targeting a minimum 6% dividend yield.
* *(Roadmap)* **Benjamin Graham Intrinsic Value Formula.**

---

## 🛡️ Compliance & Guardrails

The pipeline combines System Prompt constraints with **deterministic post-generation auditing** ([agents/compliance.py](agents/compliance.py)):

* **Zero Mental Math:** The Analyst is prohibited from estimating or computing financial metrics outside tool payloads.
* **Clause-Scoped Compliance Guardrail:** Before writing `Valuation_<TICKER>.md`, `compliance_node` scans normalized text (stripping inline Markdown such as `**Buy**`) for explicit investment recommendation patterns (e.g., `Recomendação: Compra/Venda/Manter`, `Recomendamos comprar/vender`, `Rating: Buy/Sell/Hold`, `Sinal de compra/venda`) while **avoiding false positives** on legitimate corporate finance prose (*"venda de ativos"*, *"compra de carteira"*) and negative legal disclaimers (*"não constitui recomendação de compra ou venda"*).
* **Bounded Corrective Rewrite:** Detected violations trigger a clean rewrite loop back to the Analyst (capped at `MAX_TENTATIVAS_COMPLIANCE = 2`); if violations persist at the cap, an explicit automated compliance alert banner is prepended to the saved report.

---

## 📐 Architecture Decision Records (ADRs)

Key engineering trade-offs are documented in [docs/adr/README.md](docs/adr/README.md):

* [ADR-001: Deterministic Python for Valuation and Accounting Validation](docs/adr/001-deterministic-valuation-and-validation.md)
* [ADR-002: Lexical Page Ranking and Post-Validation SHA-256 Cache Over a Vector DB](docs/adr/002-lexical-page-ranking-and-sha256-cache-over-vector-db.md)
* [ADR-003: Post-Generation Compliance Guardrail with Clause-Scoped Negation](docs/adr/003-clause-scoped-compliance-guardrail.md)
* [ADR-004: Deterministic Calculator Node and Graceful LLM Fallback Over ReAct Tool Loop](docs/adr/004-deterministic-calculator-and-llm-fallback.md)

---

## 📂 Project Structure

```text
AI-Agents-for-Stocks-Valuation/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI (ruff check . + pytest -v on every push/PR)
├── docs/
│   └── adr/                      # Architecture Decision Records (engineering trade-offs)
│       ├── README.md
│       ├── 001-deterministic-valuation-and-validation.md
│       ├── 002-lexical-page-ranking-and-sha256-cache-over-vector-db.md
│       ├── 003-clause-scoped-compliance-guardrail.md
│       └── 004-deterministic-calculator-and-llm-fallback.md
├── config.py                     # Centralized config (Ke, g, retry limits, cache, fallback model, LLM factory)
├── graph/
│   ├── __init__.py               # Graph exports
│   ├── state.py                  # AgentState TypedDict & ResultadoQuantitativo Pydantic schemas
│   ├── nodes.py                  # Nodes (extrator, validador, calculadora, analista, compliance) & fallback generator
│   └── workflow.py               # build_app(llm=None, fallback_llm=None) with dependency injection
├── agents/
│   ├── prompts/
│   │   ├── __init__.py           # Prompt builder exports
│   │   ├── analista_prompt.py    # Single-shot Analyst System Prompt (pre-computed numbers, no tool loop)
│   │   └── extrator_prompt.py    # Extractor Prompt with corrective feedback injection
│   ├── compliance.py             # Deterministic post-generation compliance guardrail (Pydantic)
│   ├── extrator.py               # Pydantic schema, smart page ranking, and SHA-256 cache
│   └── sector_protocols.py       # Sector guidelines aligned with Yahoo Finance mappings
├── tools/
│   ├── market_data.py            # Yahoo Finance tools (with JCP fallback), multiples & projections
│   ├── utils.py                  # B3 ticker normalization, yfinance cache, quarter sorting & sector
│   └── valuation.py              # Consistent 2-Stage DDM (Ke=14.5%, g=4.0%)
├── tests/
│   ├── conftest.py               # Deterministic cache and env isolation fixtures
│   ├── test_valuation.py         # 2-stage DDM math & domain guards
│   ├── test_market_data.py       # Projections, multiples, yfinance mocks, JCP fallback & cache
│   ├── test_graph_validation.py  # ROE/DFC audit, calculadora_node, 503 fallback, retries & SHA-256 cache
│   └── test_contracts.py         # Sector contracts, prompt interpolation, model fallback & compliance guardrail
├── reports/                      # Batch input/output directory (grouped by ticker)
│   └── BBDC3/
│       ├── Demonstracao-Financeira-1T26-Bradesco.pdf
│       ├── Demonstracao-Financeira-2T26-Bradesco.pdf
│       ├── Demonstracao-Financeira-3T26-Bradesco.pdf
│       ├── Valuation_BBDC3_quant.json  # Deterministic quantitative snapshot saved before LLM call
│       └── Valuation_BBDC3.md          # Final valuation report (or deterministic quantitative fallback)
├── .cache/                       # Local SHA-256 cache of validated PDF extractions (gitignored)
├── .env                          # Environment variables (GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODEL, USAR_CACHE_PDF)
├── .gitignore
├── main.py                       # Thin CLI entry point (--no-cache) & batch runner
├── pyproject.toml                # Pytest configuration
├── requirements.txt              # Production dependencies
└── requirements-dev.txt          # Development & CI dependencies (ruff, pytest)
```

> **💡 Document Handling & Token Optimization:** Prefer concise **Earnings Press Releases** over full IFRS statements when available. For long PDFs exceeding the character budget (`150,000` chars), the extractor applies **financial keyword page ranking** (prioritizing Income Statement, Recurring Net Income, ROE, Equity, Share Count, Cash Flow, and Management Guidance) while preserving original page order. Validated extractions are cached locally by `SHA-256(pdf) + schema_version + model` in `.cache/extrator/` to eliminate redundant token spend on repeated runs.

---

## 🚀 Quickstart

### 1. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For local development, linting, and automated testing:
```bash
pip install -r requirements-dev.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the repository root with your Google AI Studio key (and optional overrides):
```env
GOOGLE_API_KEY="your_api_key_here"
# Optional:
# GEMINI_MODEL="gemini-3.8-flash"
# GEMINI_FALLBACK_MODEL=""
# USAR_CACHE_PDF="true"
```

### 3. Add Quarterly Filings
Create one subfolder per B3 ticker inside `reports/` (e.g., `reports/BBDC3/`, `reports/BBAS3/`) and place the quarterly PDF filings inside it.

### 4. Run the Pipeline
```bash
python main.py
```
To bypass the local PDF SHA-256 cache for a specific run:
```bash
python main.py --no-cache
```
Structured `logging` output reports extraction progress, accounting audit checks, selective retries, immediate deterministic valuation tables (`Valuation_<TICKER>_quant.json`), single-shot Analyst synthesis (or quantitative fallback on `503`), and compliance verification. Final reports are written to `reports/<TICKER>/Valuation_<TICKER>.md`.

---

## 🧪 Code Quality, CI & Automated Tests

The entire test suite under `tests/` is **100% deterministic, offline, and zero-token** (no Gemini API or live network calls) and runs automatically via [GitHub Actions CI](.github/workflows/ci.yml) on every push and pull request:

```bash
ruff check .
pytest -v
```
