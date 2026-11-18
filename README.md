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
        (Nó) EXTRATOR  <────────────────────────────────────────────┐
                 │                                                  │
                 ▼ (JSON Pydantic)                                  │
        (Nó) VALIDADOR (Python Puro)                                │
                 │                                                  │
                 ▼                                                  │
  (Aresta Condicional) ROTEADOR ──(Erro Contábil / Tentativas < 3)──┘
                 │
         (Matemática OK)
                 │
                 ▼
        (Nó) ANALISTA  <──────► (Nó) FERRAMENTAS (yfinance + Valuation)
                 │
                 ▼
  [ Salva Valuation_<TICKER>.md ]
```

### Componentes do Grafo

1. **Identificador Dinâmico de Setor:** Antes de iniciar o grafo para um ativo, o sistema consulta a API do Yahoo Finance (`yfinance`), identifica o setor de atuação da companhia e carrega automaticamente o protocolo de análise correspondente (`banco`, `energia`, `commodities`, `varejo`, etc.).
2. **Nó Extrator (`extrator_node`):** Lê múltiplos PDFs trimestrais (`1T`, `2T`, `3T`, `4T`) da pasta do ativo utilizando `PyPDFLoader` e força a saída estruturada via **Pydantic** (`DadosTrimestraisPDF`). Extrai tanto variáveis contábeis estritas quanto o contexto qualitativo (`resumo_momento_operacional` e `visao_da_gestao`).
3. **Nó Validador (`validador_node`) & Loop de Reflexão:** Nó determinístico em Python puro (sem custo de tokens) que audita a consistência contábil do JSON extraído:
   * **Prova Real do ROE:** Verifica se $\frac{\text{Lucro Líquido}}{\text{Patrimônio Líquido}}$ converge com o ROE reportado.
   * **Conciliação de Caixa:** Verifica se $\text{Caixa Inicial} + \text{Variação Líquida} = \text{Caixa Final}$ (dentro de uma margem de tolerância para variação cambial).
   * Caso haja divergência, a aresta condicional `roteador_validador` devolve o fluxo ao Extrator com um prompt corretivo detalhando o erro exato (limitado a 3 tentativas via `tentativas_extracao`).
4. **Nó Analista (`analista_node`):** Recebe o histórico trimestral consolidado e validado, avalia a tendência operacional e o tom da diretoria, aciona as ferramentas matemáticas e redige o relatório técnico final.

---

## ⚙️ Engenharia Financeira e Ferramentas (`@tool`)

Para evitar alucinações matemáticas comuns em modelos de linguagem, **toda a álgebra financeira é executada por ferramentas determinísticas em Python**:

* **Anualização de Lucros em Duplo Cenário (`projetar_lucro_anual_cenarios`):** Quando o ano fiscal ainda está em andamento (ex: dados disponíveis até o 3T), a ferramenta projeta os trimestres faltantes sob duas óticas baseadas no momento operacional (`recuperacao`, `estavel` ou `deteriorando`):
  * **Cenário Conservador:** Estressa os trimestres restantes utilizando o pior trimestre do ano ou a média recuada.
  * **Cenário Mediano:** Utiliza a média simples do ano ou o *run-rate* do último trimestre.
* **Ancoragem Histórica (`yfinance`):**
  * `buscar_dados_mercado_yfinance`: Captura a cotação atualizada na B3. *(Nota técnica: o número total de ações ON+PN é extraído diretamente do PDF oficial para evitar distorções de `sharesOutstanding` por classe de ativo no Yahoo Finance).*
  * `buscar_medias_historicas_yfinance`: Calcula o **ROE médio real dos últimos 4 anos** cruzando DRE (`Net Income`) e Balanço Patrimonial (`Stockholders Equity`).
  * `buscar_payout_medio_historico_yfinance`: Calcula o **Payout médio dos últimos 4 anos** cruzando a DRE com a Demonstração de Fluxo de Caixa (`Cash Dividends Paid`), aplicando travas contra distorções não-recorrentes (>100%).
* **Múltiplos e Valuation (`fluxo_de_caixa_futuro_buffet` & `calcular_multiplos_basicos`):**
  * Calcula LPA, VPA, P/L, P/VP e ROE implícito.
  * Executa o modelo de Fluxo de Caixa Futuro / Dividendos Descontados (estilo Warren Buffett), projetando o crescimento sustentável ($g = \text{ROE} \times (1 - \text{Payout})$) com parâmetros de rigor fundamentalista (Taxa de Desconto $K_e = 18\%$ e Crescimento Perpétuo $g_{\text{perpétuo}} = 5\%$).

---

## 🧠 Metodologias de Valuation Suportadas

O sistema não se baseia em palpites do LLM, mas em modelos matemáticos clássicos da análise fundamentalista ancorados nos dados extraídos:

*   **Fluxo de Caixa Futuro (Estilo Warren Buffett):** Projeta o crescimento sustentável da empresa com base no ROE médio histórico e na taxa de retenção de lucros, descontando os fluxos a uma taxa de custo de capital (Ke) definida (padrão de 18%).
*   *(Em Breve)* **Valuation por Dividendos (Método Décio Bazin):** O sistema já possui a ferramenta `buscar_payout_medio_historico_yfinance`, preparando o terreno para uma rota alternativa que precifique o teto da ação com base em um *Dividend Yield* mínimo exigido de 6%.

*   *(Em Breve)* **Valuation pelo método de Benjamin Graham**
---

## 🛡️ Regras de Compliance

O *System Prompt* do Agente Analista possui travas estritas:
* **Zero Matemática Mental:** É proibido deduzir, arredondar ou calcular qualquer indicador fora das ferramentas fornecidas.
* **Sem Recomendações de Investimento:** O modelo é proibido de utilizar termos como *"Compra"*, *"Venda"* ou *"Manter"*, limitando-se a apresentar a Margem de Segurança e a assimetria percentual em relação ao Preço Teto de cada cenário.

---

## 📂 Estrutura do Projeto

```text
AI-Agents-for-Stocks-Valuation/
├── agents/
│   ├── extrator.py           # Schema Pydantic e lógica de extração via LLM
│   └── sector_protocols.py   # Diretrizes específicas de análise por setor
├── tools/
│   ├── market_data.py        # Ferramentas do Yahoo Finance, múltiplos e projeção anual
│   └── valuation.py          # Modelo matemático de Fluxo de Caixa (Buffett)
│   └── utils.py              # Funções auxiliares  
├── reports/                  # Diretório de entrada e saída (organizado por Ticker)
│   └── BBDC3/
│       ├── 1T26_Release.pdf
│       ├── 2T26_Release.pdf
│       ├── 3T26_Release.pdf
│       └── Valuation_BBDC3.md  # Relatório gerado automaticamente pelo pipeline
├── .env                      # Chave da API (GOOGLE_API_KEY)
├── main.py                   # Orquestrador do Grafo (LangGraph) e execução em lote
└── requirements.txt          # Dependências do projeto
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
Crie um arquivo `.env` na raiz do projeto contendo sua chave do Google AI Studio:
```env
GOOGLE_API_KEY="sua_chave_aqui"
```

### 3. Adicionar os Relatórios
Crie uma subpasta dentro de `reports/` utilizando **exatamente o código do ticker** da ação (ex: `reports/BBDC3/`, `reports/BBAS3/`) e adicione os PDFs trimestrais dentro dela.

### 4. Iniciar o Pipeline
```bash
python main.py
```
O terminal exibirá o progresso de extração, validação e chamadas de ferramentas. Ao concluir cada ativo, o relatório final formatado com tabelas comparativas estará salvo em `reports/<TICKER>/Valuation_<TICKER>.md`.