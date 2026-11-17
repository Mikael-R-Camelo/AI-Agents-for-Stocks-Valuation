import json
import operator
import os
import sys
from pathlib import Path
from typing import Annotated, NotRequired

from dotenv import load_dotenv
from typing_extensions import TypedDict

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY não foi encontrada. "
        "Defina essa variável no arquivo .env antes de executar o programa."
    )

from agents.extrator import extrair_dados_do_pdf
from agents.sector_protocols import obter_protocolo
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from tools.market_data import (
    buscar_dados_mercado_yfinance,
    buscar_medias_historicas_yfinance,
    buscar_payout_medio_historico_yfinance,
    calcular_multiplos_basicos,
    projetar_lucro_anual_cenarios,
)
from tools.utils import identificar_setor_pelo_ticker
from tools.valuation import fluxo_de_caixa_futuro_buffet

# ==========================================
# 1. Definindo as ferramentas e o modelo
# ==========================================
ferramentas = [
    buscar_dados_mercado_yfinance,
    buscar_medias_historicas_yfinance,
    buscar_payout_medio_historico_yfinance,
    projetar_lucro_anual_cenarios,
    calcular_multiplos_basicos,
    fluxo_de_caixa_futuro_buffet,
]

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, max_retries=3)
model_with_tools = llm.bind_tools(ferramentas)


# ==========================================
# 2. Definindo o estado do agente
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    setor: NotRequired[str]
    ticker: NotRequired[str]
    caminhos_pdfs: NotRequired[list[str]]  # Lista com todos os PDFs da subpasta
    tentativas_extracao: int
    dados_temporarios: NotRequired[dict]


# ==========================================
# 3. Definindo os nós do grafo
# ==========================================
def extrator_node(state: AgentState):
    caminhos = state.get("caminhos_pdfs", [])
    ticker = state.get("ticker", "Desconhecido")

    if not caminhos:
        return {"messages": []}

    historico_trimestral = []

    for caminho in sorted(caminhos):
        nome_arq = Path(caminho).name
        print(f"[Extrator] Extraindo dados de {nome_arq}...")
        dados_tri = extrair_dados_do_pdf(caminho)
        dados_tri["arquivo_origem"] = nome_arq
        historico_trimestral.append(dados_tri)

    # Ordena cronologicamente pelo código do trimestre (ex: 1T26, 2T26, 3T26)
    historico_trimestral.sort(key=lambda x: x.get("trimestre_referencia", ""))

    json_consolidado = json.dumps(historico_trimestral, indent=2, ensure_ascii=False)
    print(
        f"\n[Extrator] Histórico Trimestral Consolidado ({ticker}):\n{json_consolidado}\n"
    )

    mensagem = HumanMessage(
        content=(
            f"Ativo: {ticker}.\n"
            f"Histórico de trimestres extraídos dos PDFs:\n{json_consolidado}\n\n"
            "Siga rigorosamente o passo a passo das ferramentas para consolidar o ano e calcular os dois cenários."
        )
    )

    return {"messages": [mensagem]}


def analista_node(state: AgentState):
    mensagens = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in mensagens):
        protocolo = obter_protocolo(state.get("setor", "banco"))
        prompt_sistema = SystemMessage(
            content=(
                f"Você é um analista fundamentalista sênior especializado em {protocolo.nome}. "
                f"{protocolo.instrucoes}\n\n"
                "ROTEIRO OBRIGATÓRIO DE EXECUÇÃO DE FERRAMENTAS:\n"
                "1. Leia a 'visao_da_gestao' e o 'resumo_momento_operacional' que vieram do extrator.\n"
                "2. Baseado no tom da diretoria (otimista/cauteloso) e nos números, decida de forma embasada se o `momento_operacional` é 'recuperacao', 'estavel' ou 'deteriorando'.\n"
                "3. Chame `buscar_dados_mercado_yfinance` APENAS para obter o `preco_atual`. "
                "Para o número de ações, use EXCLUSIVAMENTE o `numero_total_acoes` que veio extraído do PDF mais recente.\n"
                "4. Chame a ferramenta `buscar_medias_historicas_yfinance` para descobrir qual é o ROE médio real da empresa e `buscar_payout_medio_historico_yfinance` (para o Payout).\n"
                "5. Avalie a evolução dos trimestres extraídos e defina o `momento_operacional` ('recuperacao', 'estavel' ou 'deteriorando'). "
                "Em seguida, chame `projetar_lucro_anual_cenarios` passando a lista de `lucro_liquido_trimestral` em ordem cronológica.\n"
                "6. Chame `calcular_multiplos_basicos` duas vezes (uma para o lucro anual conservador e outra para o lucro anual mediano), "
                "usando o Patrimônio Líquido e o `numero_total_acoes` do PDF mais recente.\n"
                "7. Chame `fluxo_de_caixa_futuro_buffet` duas vezes:\n"
                "   - Execução A (Cenário Conservador): Use o lucro conservador, o ROE histórico e o Payout histórico para maior margem de segurança.\n"
                "   - Execução B (Cenário Mediano): Use o lucro mediano e decida se é melhor usar os parâmetros históricos ou os reportados no PDF recente.\n"
                # TODO: Pensar em uma taxa de desconto e crescimento perpetuo dinâmicos, talvez baseados no setor ou no histórico da empresa.
                # "     Talvez pedir para o usuário informar o crescimento perpetuo e a taxa de desconto, ou usar uma média histórica do setor.
                "   (Use taxa_desconto=0.18, crescimento_perpetuo=0.03, numero_total_acoes extraído do PDF e o ROE/Payout do trimestre mais recente).\n\n"
                "REGRAS ESTRITAS DE COMPLIANCE E FORMATAÇÃO:\n"
                "- JAMAIS invente, deduza ou faça contas de cabeça. Use exclusivamente os retornos das ferramentas.\n"
                "- JAMAIS use palavras de recomendação ('Compra', 'Venda', 'Manter', 'Comprar', 'Vender'). "
                "Apenas apresente a diferença percentual (Margem de Segurança / Desconto) entre o Preço Atual e o Preço Justo de cada cenário.\n"
                "- Estruture a resposta em Markdown limpo, iniciando com uma Tabela Comparativa lado a lado: "
                "Colunas: [Métrica | Cenário Conservador | Cenário Mediano | Referência Atual]."
                "- Crie uma seção 'Perspectivas da Gestão' resumindo o que a diretoria reportou nos documentos e se o tom justifica aplicar o cenário conservador ou mediano."
            )
        )
        mensagens = [prompt_sistema] + mensagens

    print("[Analista] Processando dados e acionando ferramentas...")
    resposta = model_with_tools.invoke(mensagens)
    return {"messages": [resposta]}


def validador_node(state: AgentState):
    dados = state.get("dados_temporarios", {})

    # Recupera as variáveis contábeis
    lucro = dados.get("lucro_liquido_trimestral", 0)
    pl = dados.get("patrimonio_liquido", 1)
    roe_reportado = dados.get("roe_reportado", 0)

    caixa_inicial = dados.get("caixa_inicial", 0)
    caixa_final = dados.get("caixa_final", 0)
    variacao_caixa = dados.get("variacao_liquida_caixa", 0)

    erros_encontrados = []

    # ==========================================
    # VALIDAÇÃO 1: Consistência do ROE
    # ==========================================
    roe_calculado = lucro / pl
    margem_erro_roe = abs(roe_calculado - roe_reportado)

    # Tolera até 2 pontos percentuais de diferença
    if margem_erro_roe > 0.02:
        erros_encontrados.append(
            f"- ROE: Lucro de {lucro} e PL de {pl} geram um ROE de {roe_calculado:.2%}, "
            f"mas você extraiu {roe_reportado:.2%}."
        )

    # ==========================================
    # VALIDAÇÃO 2: Consistência do Fluxo de Caixa
    # ==========================================
    caixa_calculado = caixa_inicial + variacao_caixa

    # Evita divisão por zero
    referencia = abs(caixa_final) if caixa_final != 0 else 1
    erro_percentual_caixa = abs(caixa_calculado - caixa_final) / referencia

    # Tolera até 2% de diferença (cobre variação cambial e arredondamentos)
    if erro_percentual_caixa > 0.02:
        erros_encontrados.append(
            f"- CAIXA: Inicial ({caixa_inicial}) + Variação ({variacao_caixa}) "
            f"resulta em {caixa_calculado}, mas o Final extraído foi {caixa_final}. "
            "Procure pelo efeito da variação cambial ou corrija os saldos."
        )

    # ==========================================
    # ROTEAMENTO
    # ==========================================
    if erros_encontrados:
        print(
            "[Validador] Falhas matemáticas detectadas. Gerando prompt de correção..."
        )
        texto_erros = "\n".join(erros_encontrados)

        msg_correcao = HumanMessage(
            content=(
                "ERRO DE VALIDAÇÃO MATEMÁTICA NA EXTRAÇÃO. Revise o documento e corrija os seguintes pontos:\n"
                f"{texto_erros}\n"
                "Atenção à escala dos números (milhões/bilhões) e não confunda valores trimestrais com os acumulados do ano."
            )
        )
        return {"messages": [msg_correcao]}

    print("[Validador] Sucesso! ROE e Conciliação de Caixa validados matematicamente.")
    return {"messages": []}


def roteador_validador(state: AgentState) -> str:
    mensagens = state["messages"]
    tentativas = state.get("tentativas_extracao", 0)

    # Se a última mensagem for ERRO (HumanMessage com a palavra ERRO)
    # E ainda não estourou o limite de tentativas, volta pro extrator.
    if mensagens and "ERRO DE VALIDAÇÃO" in mensagens[-1].content:
        if tentativas < 3:
            print(f"[Loop] Retornando ao Extrator (Tentativa {tentativas + 1}/3)...")
            return "extrator"
        else:
            print("[Loop] Limite de tentativas atingido. Seguindo com erro tolerado.")
            return "analista"

    # Se passou sem erros, segue o pipeline normal
    return "analista"


# ==========================================
# 4. Construindo e compilando o Grafo
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("extrator", extrator_node)
workflow.add_node("validador", validador_node)
workflow.add_node("analista", analista_node)
workflow.add_node("ferramentas", ToolNode(ferramentas))

workflow.add_edge(START, "extrator")

# Do extrator vai OBRIGATORIAMENTE para o Validador
workflow.add_edge("extrator", "validador")

# Do validador,  ou VOLTA pro extrator, ou AVANÇA pro analista
workflow.add_conditional_edges(
    "validador", roteador_validador, {"extrator": "extrator", "analista": "analista"}
)

workflow.add_conditional_edges(
    "analista", tools_condition, {"tools": "ferramentas", "__end__": END}
)
workflow.add_edge("ferramentas", "analista")

app = workflow.compile()

# Mostra o grafo visualmente (opcional)
# display(Image(app.get_graph().draw_mermaid_png()))

# ==========================================
# 5. Execução em Lote (Agrupada por Ativo)
# ==========================================
if __name__ == "__main__":
    # Garante que o caminho funcione rodando de dentro ou de fora da pasta do projeto
    pasta_base = Path(__file__).parent
    pasta_reports = pasta_base / "reports"

    if not pasta_reports.exists():
        pasta_reports.mkdir(parents=True)
        print(
            f"Pasta '{pasta_reports}' criada! Adicione subpastas (ex: reports/BBDC3/) com os PDFs."
        )
        sys.exit()

    print("\nIniciando Varredura de Relatórios Trimestrais...\n")

    for pasta_ticker in sorted(pasta_reports.iterdir()):
        if pasta_ticker.is_dir():
            ticker_atual = pasta_ticker.name
            arquivos_pdf = sorted(str(p) for p in pasta_ticker.glob("*.pdf"))

            if not arquivos_pdf:
                continue

            setor_dinamico = identificar_setor_pelo_ticker(ticker_atual)

            print(f"{'=' * 60}")
            print(
                f"Ativo: {ticker_atual} | Setor: {setor_dinamico} | {len(arquivos_pdf)} PDF(s) encontrado(s)"
            )
            print(f"{'=' * 60}")

            try:
                resultado_final = app.invoke(
                    {
                        "messages": [],
                        "setor": setor_dinamico,
                        "ticker": ticker_atual,
                        "caminhos_pdfs": arquivos_pdf,
                        "tentativas_extracao": 0,
                    }
                )

                # Extrai o texto puro (tratando caso o Gemini devolva lista de blocos ou string direta)
                conteudo_bruto = resultado_final["messages"][-1].content
                if isinstance(conteudo_bruto, list):
                    texto_relatorio = "\n".join(
                        bloco.get("text", "")
                        for bloco in conteudo_bruto
                        if isinstance(bloco, dict)
                    )
                else:
                    texto_relatorio = str(conteudo_bruto)

                # Salva o relatório em Markdown dentro da pasta da ação
                caminho_md = pasta_ticker / f"Valuation_{ticker_atual}.md"
                with open(caminho_md, "w", encoding="utf-8") as f:
                    f.write(texto_relatorio)

                print(f"\n[SUCESSO] Relatório salvo em: {caminho_md}\n")

            except Exception as e:
                print(f"[ERRO] Falha ao processar {ticker_atual}: {e!s}\n")
