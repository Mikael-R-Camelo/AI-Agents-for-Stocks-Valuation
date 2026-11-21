# Relatório de Análise Fundamentalista: Banco Bradesco S.A. (BBDC3)

## 1. Tabela Comparativa de Avaliação

A tabela abaixo sintetiza os resultados do modelo de desconto de dividendos (DDM de dois estágios com convergência perpétua) e os múltiplos implícitos para o exercício de 2026, confrontados com as cotações de mercado vigentes.

| Métrica | Cenário Conservador | Cenário Mediano | Referência Atual |
| :--- | :---: | :---: | :---: |
| **Preço por Ação (R$)** | **R$ 17,10** | **R$ 18,49** | R$ 12,74 |
| **Market Cap Justo (R$)** | R$ 191.160.662.765,48 | R$ 206.643.660.666,55 | R$ 142.418.238.437,02 |
| **Upside / Downside (%)** | +34,22% | +45,10% | — |
| **Margem de Segurança (%)** | 25,50% | 31,08% | — |
| **Lucro Líquido Projetado (R$)** | R$ 27.960.000.000,00 | R$ 28.079.000.000,00 | R$ 20.970.000.000,00 *(acum. 9M)* |
| **Lucro por Ação - LPA (R$)** | R$ 2,50 | R$ 2,51 | — |
| **Valor Patrimonial p/ Ação - VPA (R$)** | R$ 16,31 | R$ 16,31 | R$ 16,31 |
| **Preço sobre Lucro (P/L)** | 5,09x | 5,07x | — |
| **Preço sobre Valor Patrimonial (P/VP)**| 0,78x | 0,78x | 0,78x |
| **ROE Projetado / Calculado (%)** | 15,34% | 15,40% | — |
| **Crescimento Estágio 1 - $g_1$ (%)** | 5,86% | 7,80% | — |
| **Payout Estágio 1 Utilizado (%)** | 48,39% | 50,00% | — |
| **Payout Terminal Perpétuo (%)** | 72,41% | 72,41% | — |

---

## 2. Parâmetros Utilizados na Avaliação

Os parâmetros estruturais do modelo DDM consideram a dinâmica de capital de uma instituição financeira regulada, utilizando a convergência de retenção de capital para suportar o crescimento na perpetuidade:

* **Custo de Capital Próprio ($K_e$ / Taxa de Desconto):** 14,50% ao ano.
* **Crescimento na Perpetuidade ($g$ perpétuo):** 4,00% ao ano.
* **Payout Terminal Sustentável:** 72,41% (derivado de $1 - (g / K_e)$ sob a premissa de convergência do ROE de longo prazo ao custo de oportunidade de capital).
* **Momento Operacional Definido:** Recuperação (*Turnaround* gradual de rentabilidade).
* **Base Acionária do PDF Mais Recente:** 11.178.825.623 ações totais (conforme DRE/Balanço do 3T26).
* **Patrimônio Líquido do PDF Mais Recente:** R$ 182.282.051.282,00.
* **Fonte do Payout Histórico:** *Fallback* via indicador `yfinance_info_payoutRatio_fallback` (48,39% no conservador; 50,00% no mediano informado no release).
* **Trimestres Realizados / Projetados:** 3 trimestres realizados (soma acumulada de R$ 20,97 bilhões) e 1 trimestre projetado para a conclusão do exercício.

### Aviso de Auditoria e Ressalva de Dados
> **Aviso de Auditoria:** A linha de dividendos declarados na Demonstração dos Fluxos de Caixa (DFC) da base primária de dados de mercado retornou zerada/ausente no período — distorção comum em emissores listados na B3 que remuneram primordialmente via Juros sobre o Capital Próprio (JCP). Dessa forma, utilizou-se o *fallback* verificado de `info['payoutRatio']` de 48,39% como taxa de distribuição observada.

---

## 3. Análise dos Fundamentos e Desempenho Bancário

O Bradesco segue em trajetória de recomposição de rentabilidade ao longo dos três trimestres de 2026:

1. **Qualidade do Lucro Recorrente e Rentabilidade (ROE):**
   A instituição acumulou lucro líquido recorrente de R$ 20,97 bilhões nos primeiros nove meses de 2026 (R$ 6,81 bi no 1T26, R$ 7,05 bi no 2T26 e R$ 7,11 bi no 3T26). O ROE reportado no 3T26 alcançou 15,60%, alinhando-se aos múltiplos de saída calculados de 15,34% (conservador) e 15,40% (mediano).
2. **Margem Financeira e Eficiência Operacional:**
   A margem financeira total manteve expansão sequencial (+16,4% a/a no 1T26, +4,1% t/t no 2T26 e +2,4% t/t no 3T26), ancorada no crescimento da carteira com foco em garantias e na margem com mercado. O Índice de Eficiência Operacional consolidou melhora consistente, caindo de 46,9% (1T26) para 46,5% (2T26) e atingindo 45,4% no 3T26.
3. **Inadimplência, PDD e Qualidade de Ativos:**
   O indicador de inadimplência acima de 90 dias apresentou inflexão desfavorável, passando de 4,2% no 1T26 para 4,3% no 2T26 e 4,6% no 3T26. Esse movimento decorre de pressões em crédito para MPMEs e carteiras de capital de giro (FGI/FGO e Banco John Deere), o que acarretou um aumento de 6,6% t/t na despesa de PDD expandida no 3T26.
4. **Divisão de Seguros e Capitalização:**
   A operação seguradora seguiu como principal amortecedor de rentabilidade do conglomerado, registrando ROAE de 22,0% no 3T26. Adicionalmente, a homologação pelo Banco Central do aumento de capital de R$ 10 bilhões reforçou os índices de solvência e a margem de Basileia, garantindo capacidade de expansão colateralizada sem pressionar a estrutura de capital.

---

## 4. Perspectivas da Gestão

Ao longo de 2026, a comunicação da diretoria do Bradesco manteve um padrão de cautela operacional associado à disciplina de custos:

* **1T26:** Construtiva quanto à execução do plano de transformação e investimentos em GenAI, mas com postura moderada a conservadora na concessão em MPME e crédito rural.
* **2T26:** Otimismo quanto à recomposição do ROAE para níveis de 16,2%, balanceado por alertas expressos sobre o ambiente macroeconômico restritivo, juros elevados e taxa de endividamento das famílias.
* **3T26:** Confirmação do avanço da digitalização e celebração da homologação do aumento de capital de R$ 10 bilhões, contrapostos pelo reconhecimento do estresse na qualidade do crédito de linhas de capital de giro e pela necessidade de calibração fina nos modelos de concessão diante do avanço da NPL (>90d para 4,6%).

### Avaliação de Cenários
Embora o **Cenário Mediano (Preço Justo de R$ 18,49; Upside de 45,10%)** represente com fidelidade o ritmo corrente de eficiência (45,4%) e a resiliência das receitas de seguros e margem financeira, **o tom da administração justifica uma leitura atenta ao Cenário Conservador (Preço Justo de R$ 17,10; Upside de 34,22%)**. O repique do NPL para 4,6% no 3T26 e o crescimento da PDD demandam cautela quanto à sustentabilidade do ROE sem novos aumentos no custo do crédito nos trimestres subsequentes.

---

## 5. Nota de Isenção (Disclaimer)

Este relatório foi elaborado com fins exclusivamente informativos e analíticos, com base em dados quantitativos padronizados e demonstrações financeiras públicas do emissor. Este documento **não constitui, sob nenhuma hipótese, oferta, solicitação de oferta, aconselhamento financeiro ou recomendação expressa de compra, venda ou manutenção** de quaisquer valores mobiliários ou instrumentos financeiros referenciados. As projeções refletem premissas de modelos teóricos de avaliação sujeitas às oscilações de mercado e a fatores macroeconômicos imprevistos. A tomada de decisão de investimento cabe unicamente ao próprio investidor, conforme seu perfil de risco e objetivos financeiros.