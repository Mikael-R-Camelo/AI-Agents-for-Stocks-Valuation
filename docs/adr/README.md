# Architecture Decision Records (ADRs)

This directory documents the core architectural trade-offs behind `AI-Agents-for-Stocks-Valuation`. Each ADR captures the business/engineering context, the chosen intervention, rejected alternatives, and explicit revisit triggers.

| ID | Title | Status | Key Trade-off |
| :--- | :--- | :---: | :--- |
| [ADR-001](001-deterministic-valuation-and-validation.md) | Deterministic Python for Valuation and Accounting Validation | Accepted | Zero LLM math hallucinations & full testability vs. rigid schema contracts |
| [ADR-002](002-lexical-page-ranking-and-sha256-cache-over-vector-db.md) | Lexical Page Ranking and Post-Validation SHA-256 Cache Over a Vector DB | Accepted | Zero retrieval infra overhead & lower latency vs. ceiling on very large corpora |
| [ADR-003](003-clause-scoped-compliance-guardrail.md) | Post-Generation Compliance Guardrail with Clause-Scoped Negation | Accepted | Deterministic zero-token enforcement & no disclaimer false positives vs. subtle implicit phrasing |
| [ADR-004](004-deterministic-calculator-and-llm-fallback.md) | Deterministic Calculator Node and Graceful LLM Fallback Over ReAct Tool Loop | Accepted | Single LLM call, guaranteed quant output & resilience to `503` outages vs. fixed tool sequence |
