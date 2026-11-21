# ADR-001: Deterministic Python for Valuation and Accounting Validation

- **Status:** Accepted
- **Date:** 2026-11-21
- **Affected Components:** `tools/valuation.py`, `tools/market_data.py`, `graph/nodes.py`, `config.py`

## Context

Fundamental equity valuation requires exact arithmetic, strict scale consistency (thousands/millions/billions in BRL), and domain-specific financial conventions. During early development and auditing of this pipeline on Brazilian B3 equities (e.g., `BBDC3`), relying on LLM reasoning or unverified third-party feeds exposed critical failure modes:

1. **Silent LLM arithmetic drift:** LLMs routinely mix isolated quarterly profit with year-to-date (`6M`/`9M`) cumulative figures, confuse percentage points (`16.1`) with decimals (`0.161`), or double-count retained earnings in DCF/DDM projections.
2. **Real-world market data quirks (B3):**
   - Yahoo Finance `sharesOutstanding` frequently reflects only one share class (ON vs. PN), distorting EPS (LPA), BVPS (VPA), and intrinsic fair value.
   - Brazilian banks distribute shareholder returns primarily via JCP (*Juros sobre Capital Próprio*), causing `Cash Dividends Paid` in Yahoo Finance cash-flow statements to report `0.0` across all fiscal years.

In financial analysis, an unverified hallucinated valuation is worse than no output.

## Decision

Enforce a **strict separation of concerns** between LLM reasoning and deterministic computation:

1. **Zero LLM math:** All financial algebra runs exclusively inside typed, unit-tested Python `@tool` functions (`fluxo_de_caixa_futuro_buffet`, `calcular_multiplos_basicos`, `projetar_lucro_anual_cenarios`).
2. **Consistent 2-Stage DDM (`tools/valuation.py`):** Project explicit earnings growth at $g_{\text{sust}} = (1 - \text{payout}) \times \text{ROE}$ discounting effective dividends ($\text{Earnings}_t \times \text{payout}$), and assume competitive convergence in perpetuity ($\text{ROE}_{\text{terminal}} = K_e \Rightarrow \text{payout}_{\text{terminal}} = 1 - g / K_e$) using centralized defaults (`TAXA_DESCONTO_PADRAO = 0.145`, `CRESCIMENTO_PERPETUO_PADRAO = 0.04`).
3. **Structural data safeguards (`tools/market_data.py`):**
   - Omit `sharesOutstanding` and `bookValue` from `buscar_dados_mercado_yfinance` so the agent is structurally forced to use `numero_total_acoes` and `patrimonio_liquido` extracted from official PDF filings.
   - Add an auditable fallback to `info["payoutRatio"]` when cash-flow dividends are zero/absent (JCP), exposing both `payout_medio_observado` and `payout_sustentavel` (capped at `1.0`).
4. **Pure-Python Validation Node (`graph/nodes.py`):** `validador_node` audits every extracted PDF deterministically—checking annualized quarterly ROE (compound & linear) against reported ROE, verifying scale bounds, and conditionally reconciling cash flows when a DFC is present—driving a bounded reflection loop (`MAX_TENTATIVAS_EXTRACAO = 3`) that re-extracts **only** failing PDFs with exact error feedback.

## Alternatives Considered

- **Chain-of-Thought (CoT) math inside the Analyst prompt:** Rejected. Non-deterministic, impossible to guarantee exact convergence or unit-test without API calls, and vulnerable to compounding rounding/scale errors.
- **LLM-as-a-Validator node:** Rejected. Adds token cost and latency while still risking shared blind spots (e.g., agreeing with a quarterly vs. annualized ROE mismatch).
- **Single-stage DDM keeping current payout fixed in perpetuity:** Rejected. Artificially compresses terminal value when a firm retains earnings above what is needed to sustain perpetual growth $g$.

## Trade-offs & Consequences

- **Pros:**
  - 100% reproducible valuation outputs verifiable by fast, offline unit tests (`pytest`: 84 deterministic tests in ~0.3s).
  - Structural elimination of B3 ON/PN share-count distortions and JCP zero-payout failures.
  - Zero token cost for accounting validation and targeted single-PDF retries.
- **Cons:**
  - Adding new valuation methodologies (e.g., Décio Bazin, Benjamin Graham, FCFF for non-dividend growth companies) requires implementing dedicated Python tools and domain guards rather than simply editing prompts.
  - Companies with `payout == 0` or negative earnings cannot be valued by the DDM tool and explicitly raise domain `ValueError` guards.

## Revisit Triggers

- Supporting non-dividend growth companies or leveraged industrials will require introducing an FCFE/FCFF valuation tool alongside the 2-stage DDM.
- If cross-currency filings (USD vs. BRL in commodity exporters) become frequent, currency-matching validation should be added to `validador_node`.
