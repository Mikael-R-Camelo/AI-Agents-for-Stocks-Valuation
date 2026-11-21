# ADR-003: Post-Generation Compliance Guardrail with Clause-Scoped Negation

- **Status:** Accepted
- **Date:** 2026-11-21
- **Affected Components:** `agents/compliance.py`, `graph/nodes.py`, `graph/workflow.py`, `config.py`

## Context

Equity research is a regulated domain (e.g., CVM rules in Brazil / FINRA & SEC guidelines in the US). This pipeline must produce objective fundamental analysis (fair value, margin of safety, valuation multiples, and management outlook) **without issuing explicit investment recommendations** (buy, sell, or hold ratings).

Prompt-only compliance rules (*"never recommend buy/sell/hold"*) are soft guardrails: LLMs can still slip into phrases like `**Recomendação:** **Compra**`, `Rating: Buy`, or `Recomendamos comprar o papel`—especially when margin of safety is large.

Conversely, a naive keyword blocklist banning isolated words (`"compra"`, `"venda"`, `"manter"`) breaks legitimate financial prose (e.g., *"venda de ativos não estratégicos"*, *"compra de carteira de crédito"*, *"manter o apetite de risco moderado"*) and paradoxically blocks standard legal disclaimers (*"Este relatório não constitui recomendação de compra ou venda"*).

## Decision

Implement a **deterministic post-generation compliance node (`compliance_node`)** in the LangGraph workflow (`agents/compliance.py`) that audits every final Analyst draft before saving `Valuation_<TICKER>.md`:

1. **Explicit-Recommendation Pattern Matching:** Normalize inline Markdown markers (`*`, `_`, `` ` ``) so formatted labels like `**Recomendação:** **Venda**` cannot evade detection, then match targeted syntactic patterns for explicit calls (rating headers, direct prescriptive verbs, passive voice recommendations, explicit buy/sell signals, and investor imperatives).
2. **Clause-Scoped Negation Analysis (`_esta_em_contexto_negativo`):** Before flagging a match, inspect the preceding text within the **same clause** (bounded by strong punctuation `[.;!\n]` or adversative conjunctions like `mas`, `porém`, `contudo`, `entretanto`).
   - If the same clause contains explicit negation/disclaimer cues (`não constitui`, `não se trata de`, `não recomendamos`, `sem caráter de`, `does not constitute`), treat the match as a legal disclaimer and **approve** it.
   - If a negation precedes an adversative turn (*"Não há garantia de retorno, porém recomendamos comprar a ação"*), the clause boundary resets negation scope and the violation **is still caught**.
3. **Bounded Rewrite Loop + Explicit Fallback Banner:**
   - On violation, route back to `analista_node` with exact offending snippets and instructions to rewrite cleanly without re-calling tools, capped at `MAX_TENTATIVAS_COMPLIANCE = 2`.
   - If violations persist after reaching the cap, never ship silently: prepend an explicit `ALERTA DE COMPLIANCE AUTOMATIZADO` banner to the output Markdown for mandatory human review.

## Alternatives Considered

- **System Prompt instructions alone:** Rejected. Probabilistic enforcement without automated verification does not meet regulated-industry reliability standards.
- **Naive isolated-word regex (`\b(compra|venda|manter)\b`):** Rejected. Causes unacceptable false-positive rates on standard corporate finance terminology and legal disclaimers.
- **LLM-as-a-Compliance-Judge on every run:** Rejected as primary gate. Adds extra latency and token cost on every single report and introduces non-determinism on edge cases; deterministic regex with clause negation runs in <1ms at zero token cost and is exhaustively testable.

## Trade-offs & Consequences

- **Pros:**
  - Zero-token, sub-millisecond deterministic enforcement verified by 20+ positive, negative, disclaimer, and adversative test cases in `tests/test_contracts.py`.
  - Bounded graph execution (`MAX_TENTATIVAS_COMPLIANCE = 2`) prevents infinite rewrite loops while guaranteeing explicit flagging on stubborn failures.
  - Structured Pydantic audit payload (`ResultadoCompliance`) enables future logging/telemetry aggregation.
- **Cons:**
  - Rule-based matching targets explicit recommendations and may not catch subtle, indirect investment nudges (e.g., *"this stock is an unmissable opportunity to build a position today"*).
  - Pattern maintenance is required if reports are generated in additional languages beyond Portuguese and English rating labels.

## Revisit Triggers

- Add an asynchronous **LLM-as-a-Judge semantic compliance evaluator** in an offline eval harness (or as a secondary check for external client distribution) if subtle implicit persuasion needs to be audited alongside explicit calls.
- Expand negation/pattern sets if multi-language report generation (e.g., full English or Spanish reports) is introduced.
