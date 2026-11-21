# ADR-004: Deterministic Calculator Node and Graceful LLM Fallback Over ReAct Tool Loop

- **Status:** Accepted
- **Date:** 2026-11-21
- **Affected Components:** `graph/state.py`, `graph/nodes.py`, `graph/workflow.py`, `agents/prompts/analista_prompt.py`, `config.py`, `main.py`

## Context

In the initial LangGraph topology, once quarterly PDFs passed accounting validation, control moved to `analista_node` wired in a ReAct loop with `ToolNode(FERRAMENTAS)`. Even though every equity analysis executes the **exact same fixed sequence of six deterministic steps** (fetch price -> fetch 4-year ROE -> fetch 4-year Payout -> annualize earnings -> compute multiples -> run 2-stage DDM for Conservative and Median scenarios), the Analyst required **3 to 5 sequential LLM round-trips** just to dispatch those tools before drafting the final report.

During real-world execution against `BBDC3`, high-demand spikes on the Gemini API returned `HTTP 503 Service Unavailable` mid-loop (after ~12 minutes of extraction and tool turns), aborting the entire pipeline with **zero delivered output**—no fair value, no margin of safety, no upside/downside, and no persisted report—despite all PDF data already being extracted and validated.

## Decision

Decouple deterministic quantitative computation from LLM qualitative synthesis:

1. **Dedicated Pure-Python `calculadora_node` (`graph/nodes.py` + `graph/state.py`):**
   - Insert `calculadora_node` immediately after `validador_node` (replacing the `analista <-> ToolNode` ReAct cycle).
   - Classify `momento_operacional` (`recuperacao`, `estavel`, `deteriorando`) deterministically via `_classificar_momento_operacional` over the chronological quarterly profit series (handling losses safely via `abs(media_anterior)`).
   - Execute all six existing `@tool` functions directly (preserving their public signatures and tests), structuring all outputs in a typed Pydantic schema (`ResultadoQuantitativo`).
   - Log a formatted summary table immediately to the terminal and atomically persist `reports/<TICKER>/Valuation_<TICKER>_quant.json` **before** calling the Analyst LLM.
   - Isolate DDM domain errors (e.g., negative earnings or zero payout) per scenario so quotes, projections, and valid multiples are still produced.
2. **Single-Shot `analista_node` with Optional Model Fallback and Deterministic Report Degradation:**
   - Remove `ToolNode` and `bind_tools`; the Analyst now makes a **single LLM call** to synthesize the qualitative narrative while strictly reproducing pre-computed numbers from `resultado_quantitativo`.
   - Catch transient LLM API/network errors (`503`, `429`, `TimeoutError`, connection errors).
   - If `GEMINI_FALLBACK_MODEL` is explicitly configured in `.env` (and differs from the primary model), attempt the secondary model once; otherwise (or if it also fails), invoke `gerar_relatorio_fallback_quantitativo(...)` to produce a complete, compliance-clean Markdown report (`Valuation_<TICKER>.md`) containing tables, valuation parameters, extracted qualitative quarter summaries, and an explicit fallback notice.
   - Keep `compliance_node` active over fallback outputs while preventing `roteador_compliance` from looping back into an unavailable LLM (`analise_llm_indisponivel == True`).

## Alternatives Considered

- **Increase retry count / exponential backoff on the ReAct `ToolNode` loop:** Rejected. Still wastes 3–5 unnecessary LLM calls per ticker to orchestrate a fixed deterministic sequence and still loses all computed valuation numbers during prolonged API outages.
- **Keep `ToolNode` only for optional ad-hoc queries:** Rejected for the current scope. Adds tool-calling latency, token overhead, and non-deterministic tool ordering when every required input is already known upfront.
- **Remove `@tool` decorators from `tools/market_data.py` and `tools/valuation.py`:** Rejected. Preserving the `@tool` interfaces keeps compatibility with existing unit tests and leaves the functions ready if future interactive chat/MCP workflows need dynamic tool calling.

## Trade-offs & Consequences

- **Pros:**
  - Reduces post-validation LLM calls from **3–5 calls down to 1**, sharply lowering latency, token cost, and exposure to `503`/`429` outages.
  - Guarantees immediate visibility (terminal summary + `Valuation_<TICKER>_quant.json`) and graceful Markdown delivery (`Valuation_<TICKER>.md`) even during full LLM downtime.
  - Eliminates any residual risk of the Analyst skipping a tool call, passing wrong arguments, or mixing Conservative/Median inputs.
- **Cons:**
  - `momento_operacional` is classified solely from quarterly earnings trajectory rather than combining qualitative management tone during the calculation step (the Analyst still contextualizes management tone in the narrative and recommends which scenario to prioritize prudentially).
  - Adding dynamic, open-ended research tools in the future (e.g., web search or ad-hoc filing Q&A) will require re-introducing a scoped tool-calling sub-node only for that exploratory step.

## Revisit Triggers

- If `DadosTrimestraisPDF` schema is bumped to `v3` in the future, adding a structured `tom_gestao: Literal["otimista", "cauteloso", "pessimista"]` field can combine numerical trajectory and management tone deterministically inside `calculadora_node`.
- Re-introduce bounded tool calling only if future features require non-fixed exploratory workflows (e.g., interactive analyst Q&A or MCP server integration).
