# 📈 AI Agents for Stocks Valuation

Pipeline multiagente autônomo desenvolvido em Python com **LangGraph** e **Google Gemini** para análise fundamentalista e *valuation* automatizado de ações da B3 a partir de relatórios trimestrais (*Press Releases*) e dados de mercado em tempo real.

O sistema combina a capacidade de leitura contextual de LLMs com a precisão determinística do Python, utilizando o padrão de arquitetura **Reflection (Self-Correction)** tanto na auditoria contábil da extração (por PDF) quanto no guardrail determinístico de compliance pós-geração.

---

## 🏗️ Arquitetura do Sistema (LangGraph)

O projeto opera como uma máquina de estados (`StateGraph`) onde uma memória compartilhada (`AgentState`) trafega entre nós especializados:

```text
  [ Início / Varredura de Pastas ]
                 │
                 ▼ (Identifica Setor via Yahoo Finance)
         (Nó) EXTRATOR  <───────────────────────────────────────────────┐
                 │                                                      │
                 ▼ (JSON Pydantic por PDF)                              │
        (Nó) VALIDADOR (Python Puro)                                    │
                 │                                                      │
                 ▼                                                      │
  (Aresta Condicional) ROTEADOR ──(Reprocessa só PDFs com erro < 3)─────┘
                 │
         (Histórico Consolidado em Ordem Cronológica)
                 │
                 ▼
         (Nó) ANALISTA  <──────► (Nó) FERRAMENTAS (yfinance + Valuation)
                 │
                 ▼
       (Nó) COMPLIANCE (Python Puro) ──(Violação / Tentativas <= 2)──► volta ao ANALISTA
                 │
          (Aprovado ou Alerta Explícito)
                 │
                 ▼
   [ Salva Valuation_<TICKER>.md ]
```

### Componentes do Grafo

1. **Identificador Dinâmico de Setor (`tools/utils.py` + `agents/sector_protocols.py`):** Antes de iniciar o grafo para um ativo, o sistema consulta o Yahoo Finance (`yfinance`), identifica o setor da companhia e carrega o protocolo correspondente (`banco`, `energia`/`eletrica`, `holding`, `commodities`, `imobiliario`, `varejo`, `saude`, `industria` ou `geral`).
2. **Nó Extrator (`extrator_node`):** Lê múltiplos PDFs trimestrais (`1T`, `2T`, `3T`, `4T`) utilizando `PyPDFLoader` e saída estruturada via **Pydantic** (`DadosTrimestraisPDF`). Os campos de DFC (`caixa_inicial`, `caixa_final`, `variacao_liquida_caixa` e `efeito_cambial_caixa`) são opcionais (`float | None`) para suportar *Press Releases* de bancos sem DFC. Em retentativas, reprocessa **cirurgicamente apenas os PDFs reprovados**, injetando o feedback exato da divergência no prompt ([extrator_prompt.py](agents/prompts/extrator_prompt.py)).
3. **Nó Validador (`validador_node`) & Loop de Reflexão:** Nó determinístico em Python puro (sem custo de tokens) que audita todos os trimestres extraídos:
   * **Integridade Estrutural:** valida `numero_total_acoes > 0`, `patrimonio_liquido != 0` e escalas decimais (`roe_reportado`, `payout_estimado`).
   * **Prova Real do ROE Anualizado:** anualiza o ROE trimestral ($\frac{\text{Lucro}_{\text{tri}}}{\text{PL}}$ nas convenções composta e linear) e compara com o `roe_reportado` anualizado.
   * **Conciliação de Caixa (DFC opcional):** executada apenas quando a DFC estiver presente no PDF, validando $\text{Caixa Inicial} + \text{Variação Líquida} + \text{Efeito Cambial} \approx \text{Caixa Final}$.
   * **Ordenação Cronológica Real:** consolida os trimestres por `(ano, trimestre)` via `chave_ordenacao_trimestre`, preservando a ordem correta mesmo em viradas de ano (ex.: `4T25 -> 1T26 -> 2T26`).
   * **Limite de Tentativas:** `tentativas_extracao` é incrementado a cada passagem e limita o loop a no máximo **3 tentativas** (`MAX_TENTATIVAS_EXTRACAO = 3`), encaminhando ressalvas de auditoria ao Analista caso algum erro persista.
4. **Nó Analista (`analista_node`):** Carrega o *System Prompt* dedicado em [analista_prompt.py](agents/prompts/analista_prompt.py) com os parâmetros oficiais importados diretamente de `tools/valuation.py`, avalia o momento operacional, aciona exclusivamente as ferramentas determinísticas e redige o relatório final em Markdown.
5. **Nó de Compliance (`compliance_node`):** Guardrail determinístico pós-geração ([compliance.py](agents/compliance.py)) que audita o relatório final antes de salvar o `.md` (detalhes na seção de Compliance).

---

## ⚙️ Engenharia Financeira e Ferramentas (`@tool`)

Para evitar alucinações matemáticas comuns em modelos de linguagem, **toda a álgebra financeira é executada por ferramentas determinísticas em Python**:

* **Anualização de Lucros em Duplo Cenário (`projetar_lucro_anual_cenarios`):** Quando o ano fiscal ainda está em andamento (ex.: dados até o 3T), projeta os trimestres faltantes sob duas óticas baseadas no `momento_operacional` (`recuperacao`, `estavel` ou `deteriorando`), tratando corretamente lucros e prejuízos e garantindo a invariante `lucro_anual_conservador <= lucro_anual_mediano`:
  * **Cenário Conservador:** estressa os trimestres restantes utilizando o pior trimestre (ou a média do ano em recuperação), agravando prejuízos quando negativos.
  * **Cenário Mediano:** utiliza a média do ano, o *run-rate* do último trimestre ou o pior trimestre em deterioração.
* **Ancoragem Histórica e Blindagem ON/PN (`yfinance`):**
  * `buscar_dados_mercado_yfinance`: captura **exclusivamente** a cotação atual (`preco_atual`, com fallback para `fast_info`) e a moeda na B3. **Blindagem estrutural:** não retorna `sharesOutstanding` nem `bookValue`, eliminando o risco de distorção por classe ON/PN — `numero_total_acoes` e `patrimonio_liquido` vêm exclusivamente do PDF oficial.
  * `buscar_medias_historicas_yfinance`: calcula o **ROE médio real dos últimos 4 anos** cruzando DRE e Balanço Patrimonial com suporte a aliases IFRS do Yahoo Finance.
  * `buscar_payout_medio_historico_yfinance`: calcula o **Payout médio dos últimos 4 anos** cruzando DRE e DFC (com suporte a dividendos ON+PN), descarta distorções não-recorrentes `> 150%`, aplica **fallback automático para `info["payoutRatio"]`** quando a DFC vier zerada/ausente (comum em bancos brasileiros que distribuem via **JCP**) e retorna explicitamente `payout_medio_observado` e `payout_sustentavel` (limitado ao teto de `1.0` para o DDM).
* **Múltiplos e Valuation (`calcular_multiplos_basicos` & `fluxo_de_caixa_futuro_buffet`):**
  * Calcula LPA, VPA, P/L, P/VP e ROE implícito (retornando `None` para múltiplos não aplicáveis em caso de prejuízo ou PL negativo).
  * Executa o **DDM de 2 Estágios Consistente** com guardas numéricas e parâmetros oficiais definidos em `tools/valuation.py`:
    * **Estágio 1 (Anos 1 a $N$):** lucro anual cresce à taxa sustentável $g_{\text{sust}} = (1 - \text{payout}) \times \text{ROE}$ e desconta os dividendos efetivamente distribuídos ($\text{Lucro}_t \times \text{payout}$).
    * **Estágio 2 (Perpetuidade / Valor Terminal):** assume convergência competitiva do retorno marginal ao custo de capital próprio ($ROE_{\text{terminal}} = K_e$), aplicando o payout sustentável de maturidade $\text{payout}_{\text{terminal}} = 1 - \frac{g_{\text{perpétuo}}}{K_e}$ no modelo de Gordon Growth.
    * **Parâmetros Padrão:** Taxa de Desconto **$K_e = 14{,}5\%$** (`TAXA_DESCONTO_PADRAO = 0.145`) e Crescimento Perpétuo **$g_{\text{perpétuo}} = 4{,}0\%$** (`CRESCIMENTO_PERPETUO_PADRAO = 0.04`), resultando em $\text{payout}_{\text{terminal}} = 72{,}41\%$.

---

## 🧠 Metodologias de Valuation Suportadas

O sistema é ancorado em modelos matemáticos determinísticos alimentados pelos dados extraídos e auditados:

* **DDM de 2 Estágios Consistente (Fluxo de Dividendos Descontados):** Projeta o crescimento sustentável via retenção de lucros no estágio explícito (4 anos), desconta os dividendos efetivos a $K_e = 14{,}5\%$ e aplica convergência competitiva na perpetuidade com $g_{\text{perpétuo}} = 4{,}0\%$ e $\text{payout}_{\text{terminal}}$ sustentável.
* *(Em Breve)* **Valuation por Dividendos (Método Décio Bazin):** O sistema já conta com `buscar_payout_medio_historico_yfinance` (incluindo fallback para JCP), preparando a rota alternativa baseada em *Dividend Yield* mínimo exigido de 6%.
* *(Em Breve)* **Valuation pelo método de Benjamin Graham.**

---

## 🛡️ Regras de Compliance

O pipeline combina travas de *System Prompt* com **auditoria determinística pós-geração** ([agents/compliance.py](agents/compliance.py)):

* **Zero Matemática Mental:** É proibido deduzir, arredondar ou calcular qualquer indicador fora das ferramentas Python.
* **Guardrail Determinístico contra Recomendações Explícitas:** Antes de salvar o arquivo `.md`, o nó `compliance_node` verifica padrões explícitos de recomendação de investimento (como `Recomendação: Compra/Venda/Manter`, `Recomendamos comprar/vender`, `Rating: Buy/Sell/Hold`, `Sinal de compra/venda`), **sem gerar falsos positivos** em termos financeiros legítimos (ex.: *"venda de carteira"*, *"compra de ativos"*) ou em disclaimers negativos (*"não constitui recomendação de compra ou venda"*).
* **Ação Corretiva Controlada:** Caso detecte violação, solicita reescrita limpa ao Analista (até 2 tentativas via `MAX_TENTATIVAS_COMPLIANCE = 2`); persistindo a violação no limite, anexa um banner explícito de alerta de compliance no topo do relatório salvo.

---

## 📂 Estrutura do Projeto

```text
AI-Agents-for-Stocks-Valuation/
├── agents/
│   ├── prompts/
│   │   ├── __init__.py           # Exportação dos construtores de prompts
│   │   ├── analista_prompt.py    # System Prompt do Analista (interpola Ke e g de valuation.py)
│   │   └── extrator_prompt.py    # Prompt do Extrator com suporte a feedback corretivo
│   ├── compliance.py             # Guardrail determinístico de compliance pós-geração (Pydantic)
│   ├── extrator.py               # Schema Pydantic (DadosTrimestraisPDF) e extração via LLM
│   └── sector_protocols.py       # Protocolos setoriais alinhados ao mapeamento do Yahoo Finance
├── tools/
│   ├── market_data.py            # Ferramentas do Yahoo Finance (com fallback JCP), múltiplos e projeção
│   ├── utils.py                  # Normalização de tickers B3, ordenação cronológica, setor e modelo Gemini
│   └── valuation.py              # Modelo DDM de 2 Estágios consistente (Ke=14,5%, g=4,0%)
├── reports/                      # Diretório de entrada e saída (organizado por Ticker)
│   └── BBDC3/
│       ├── Demonstracao-Financeira-1T26-Bradesco.pdf
│       ├── Demonstracao-Financeira-2T26-Bradesco.pdf
│       ├── Demonstracao-Financeira-3T26-Bradesco.pdf
│       └── Valuation_BBDC3.md    # Relatório gerado automaticamente pelo pipeline
├── .env                          # Variáveis de ambiente (GOOGLE_API_KEY e opcional GEMINI_MODEL)
├── .gitignore
├── main.py                       # Orquestrador do Grafo (LangGraph) e execução em lote
└── requirements.txt              # Dependências do projeto
```

> **💡 Boas Práticas de Documentos:** Priorize colocar os arquivos de **Press Release** (Relatório de Análise Econômica e Financeira) dentro das pastas dos ativos em vez das Demonstrações Financeiras completas (IFRS). O Press Release concentra o Lucro Líquido Recorrente, ROE gerencial e comentários da administração com menor ruído e menor consumo de tokens.

---

## 🚀 Como Executar

### 1. Configurar o Ambiente Virtual e Dependências
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto contendo sua chave do Google AI Studio (e, opcionalmente, o identificador do modelo Gemini):
```env
GOOGLE_API_KEY="sua_chave_aqui"
# Opcional (padrão funcional mantido: gemini-3.8-flash)
# GEMINI_MODEL="gemini-3.8-flash"
```

### 3. Adicionar os Relatórios
Crie uma subpasta dentro de `reports/` utilizando **exatamente o código do ticker** da ação (ex: `reports/BBDC3/`, `reports/BBAS3/`) e adicione os PDFs trimestrais dentro dela.

### 4. Iniciar o Pipeline
```bash
python main.py
```
O terminal exibirá o progresso de extração, validação contábil (com reprocessamento seletivo se houver erro), chamadas de ferramentas e auditoria de compliance. Ao concluir cada ativo, o relatório final estará salvo em `reports/<TICKER>/Valuation_<TICKER>.md`.
