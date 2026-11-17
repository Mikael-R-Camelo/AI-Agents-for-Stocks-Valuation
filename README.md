# 📈 AI Agents for Stocks Valuation

Pipeline multiagente autônomo desenvolvido em Python com **LangGraph** e **Google Gemini** para análise fundamentalista e *valuation* automatizado de ações da B3 a partir de relatórios trimestrais (*Press Releases*) e dados de mercado em tempo real.

O sistema combina a capacidade de leitura contextual de LLMs com a precisão determinística do Python, utilizando o padrão de arquitetura **Reflection (Self-Correction)** para auditar matematicamente os dados extraídos antes de projetar fluxos de caixa.

---

## 🏗️ Arquitetura do Sistema (LangGraph)

O projeto opera como uma máquina de estados (`StateGraph`) onde uma memória compartilhada (`AgentState`) trafega entre nós especializados:

```text
  [ Início / Varredura de Pastas ]
                 │
                 ▼ (Identifica Setor via Yahoo Finance)
        (Nó) EXTRATOR  <──────────────────────────────────────┐
                 │                                            │
                 ▼ (JSON Pydantic)                            │
        (Nó) VALIDADOR (Python Puro)                          │
                 │                                            │
                 ▼                                            │
  (Aresta Condicional) ROTEADOR ──(Erro Contábil / Tent < 3)──┘
                 │
         (Matemática OK)
                 │
                 ▼
        (Nó) ANALISTA  <──────► (Nó) FERRAMENTAS (yfinance + Valuation)
                 │
                 ▼
  [ Salva Valuation_<TICKER>.md ]