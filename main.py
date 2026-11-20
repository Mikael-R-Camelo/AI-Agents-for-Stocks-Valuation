import json
import operator
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.extrator import extrair_dados_do_pdf
from agents.sector_protocols import obter_protocolo
from tools.market_data import (
    buscar_dados_mercado_yfinance,
    buscar_medias_historicas_yfinance,
    buscar_payout_medio_historico_yfinance,
    calcular_multiplos_basicos,
    projetar_lucro_anual_cenarios,
)
from tools.utils import (
    chave_ordenacao_trimestre,
    identificar_setor_pelo_ticker,
    obter_modelo_gemini,
)
from tools.valuation import (
    CRESCIMENTO_PERPETUO_PADRAO,
    TAXA_DESCONTO_PADRAO,
    fluxo_de_caixa_futuro_buffet,
)

load_dotenv()

MAX_TENTATIVAS_EXTRACAO = 3
TOLERANCIA_ROE = 0.025
TOLERANCIA_CAIXA = 0.02

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


def _garantir_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY não foi encontrada. "
            "Defina essa variável no arquivo .env antes de executar o programa."
        )


@lru_cache(maxsize=1)
def _obter_llm() -> ChatGoogleGenerativeAI:
    _garantir_api_key()
    return ChatGoogleGenerativeAI(
        model=obter_modelo_gemini(),
        temperature=0,
        max_retries=3,
    )


@lru_cache(maxsize=1)
def _obter_model_with_tools():
    return _obter_llm().bind_tools(ferramentas)


# ==========================================
# 2. Definindo o estado do agente
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    setor: NotRequired[str]
    ticker: NotRequired[str]
    caminhos_pdfs: NotRequired[list[str]]  # Lista com todos os PDFs da subpasta
    tentativas_extracao: int
    dados_extraidos: NotRequired[dict[str, dict[str, Any]]]
    erros_validacao: NotRequired[dict[str, list[str]]]
    validacao_concluida: NotRequired[bool]


# ==========================================
# 3. Funções auxiliares de validação contábil
# ==========================================
def _validar_trimestre(dados: dict[str, Any]) -> list[str]:
    """Audita matematicamente um trimestre extraído (ROE anualizado e DFC opcional)."""
    erros: list[str] = []

    lucro = dados.get("lucro_liquido_trimestral")
    pl = dados.get("patrimonio_liquido")
    roe_reportado = dados.get("roe_reportado")
    payout = dados.get("payout_estimado")
    num_acoes = dados.get("numero_total_acoes")

    # ==========================================
    # VALIDAÇÃO 0: Campos estruturais obrigatórios
    # ==========================================
    if num_acoes is None or num_acoes <= 0:
        erros.append(
            f"- AÇÕES: numero_total_acoes extraído ({num_acoes}) é inválido. "
            "Extraia o total absoluto de ações (ON + PN)."
        )

    if pl is None or pl == 0:
        erros.append(
            f"- PATRIMÔNIO LÍQUIDO: valor extraído ({pl}) é nulo ou zero, "
            "impedindo a verificação contábil do ROE."
        )

    if payout is not None and not (0.0 <= payout <= 1.5):
        erros.append(
            f"- PAYOUT: valor extraído ({payout}) está fora da escala decimal esperada "
            "(ex: 0.50 para 50%)."
        )

    # ==========================================
    # VALIDAÇÃO 1: Consistência do ROE (Trimestral anualizado vs ROE reportado)
    # ==========================================
    if roe_reportado is not None and abs(roe_reportado) > 2.0:
        erros.append(
            f"- ROE (ESCALA): roe_reportado ({roe_reportado}) parece estar em pontos percentuais "
            "em vez de decimal (ex: use 0.161 para 16,1%)."
        )
    elif lucro is not None and pl is not None and pl > 0 and roe_reportado is not None:
        roe_trimestral = lucro / pl
        roe_anual_linear = roe_trimestral * 4.0
        roe_anual_composto = (
            ((1.0 + roe_trimestral) ** 4) - 1.0
            if roe_trimestral > -1.0
            else roe_anual_linear
        )

        erro_composto = abs(roe_anual_composto - roe_reportado)
        erro_linear = abs(roe_anual_linear - roe_reportado)

        if min(erro_composto, erro_linear) > TOLERANCIA_ROE:
            erros.append(
                f"- ROE: Lucro trimestral de {lucro:,.2f} e PL de {pl:,.2f} geram "
                f"ROE trimestral de {roe_trimestral:.2%} (anualizado: {roe_anual_composto:.2%} composto / "
                f"{roe_anual_linear:.2%} linear), mas roe_reportado extraído foi {roe_reportado:.2%}. "
                "Verifique se o lucro não está acumulado (6M/9M) ou se há erro de escala."
            )

    # ==========================================
    # VALIDAÇÃO 2: Consistência do Fluxo de Caixa (apenas quando DFC existir no PDF)
    # ==========================================
    caixa_inicial = dados.get("caixa_inicial")
    caixa_final = dados.get("caixa_final")
    variacao_caixa = dados.get("variacao_liquida_caixa")
    efeito_cambial = dados.get("efeito_cambial_caixa")

    campos_dfc = (caixa_inicial, caixa_final, variacao_caixa)
    if any(c is not None for c in campos_dfc):
        if any(c is None for c in campos_dfc):
            erros.append(
                "- CAIXA: A DFC foi extraída parcialmente. Preencha caixa_inicial, caixa_final "
                "e variacao_liquida_caixa em conjunto, ou defina todos como null se não houver DFC no PDF."
            )
        else:
            ajuste_cambio = float(efeito_cambial) if efeito_cambial is not None else 0.0
            caixa_calculado = (
                float(caixa_inicial) + float(variacao_caixa) + ajuste_cambio
            )
            referencia = max(abs(float(caixa_final)), abs(float(caixa_inicial)), 1.0)
            erro_percentual_caixa = abs(caixa_calculado - float(caixa_final)) / referencia

            if erro_percentual_caixa > TOLERANCIA_CAIXA:
                erros.append(
                    f"- CAIXA: Inicial ({caixa_inicial:,.2f}) + Variação ({variacao_caixa:,.2f}) "
                    f"+ Efeito Cambial ({ajuste_cambio:,.2f}) resulta em {caixa_calculado:,.2f}, "
                    f"mas o Final extraído foi {caixa_final:,.2f}. "
                    "Verifique o efeito cambial na DFC ou corrija os saldos."
                )

    return erros


# ==========================================
# 4. Definindo os nós do grafo
# ==========================================
def extrator_node(state: AgentState) -> dict[str, Any]:
    caminhos = state.get("caminhos_pdfs", [])
    dados_existentes = dict(state.get("dados_extraidos") or {})
    erros_anteriores = state.get("erros_validacao") or {}
    tentativas = state.get("tentativas_extracao", 0)

    if not caminhos:
        return {
            "dados_extraidos": {},
            "erros_validacao": {},
            "validacao_concluida": True,
        }

    # Em retry, reprocessa APENAS os PDFs reprovados no validador
    if erros_anteriores:
        alvos = [c for c in sorted(caminhos) if c in erros_anteriores]
    else:
        alvos = sorted(caminhos)

    llm = _obter_llm()

    for caminho in alvos:
        nome_arq = Path(caminho).name
        feedback = (
            "\n".join(erros_anteriores[caminho])
            if caminho in erros_anteriores
            else None
        )
        if feedback:
            print(f"[Extrator] Reprocessando {nome_arq} com feedback do validador...")
        else:
            print(f"[Extrator] Extraindo dados de {nome_arq}...")

        dados_tri = extrair_dados_do_pdf(
            caminho,
            feedback_correcao=feedback,
            llm=llm,
        )
        dados_tri["arquivo_origem"] = nome_arq
        dados_existentes[caminho] = dados_tri

    return {
        "dados_extraidos": dados_existentes,
        "erros_validacao": {},
        "tentativas_extracao": tentativas + 1,
        "validacao_concluida": False,
    }


def validador_node(state: AgentState) -> dict[str, Any]:
    dados_extraidos = state.get("dados_extraidos") or {}
    ticker = state.get("ticker", "Desconhecido")
    tentativas = state.get("tentativas_extracao", 0)

    if not dados_extraidos:
        print("[Validador] Nenhum PDF extraído para validar.")
        return {
            "erros_validacao": {},
            "validacao_concluida": True,
        }

    erros_por_arquivo: dict[str, list[str]] = {}
    for caminho, dados in dados_extraidos.items():
        erros_pdf = _validar_trimestre(dados)
        if erros_pdf:
            erros_por_arquivo[caminho] = erros_pdf

    # Se há erros e ainda restam tentativas, devolve ao extrator sem poluir state["messages"]
    if erros_por_arquivo and tentativas < MAX_TENTATIVAS_EXTRACAO:
        print(
            f"[Validador] Falhas matemáticas em {len(erros_por_arquivo)} PDF(s) "
            f"(tentativa {tentativas}/{MAX_TENTATIVAS_EXTRACAO}):"
        )
        for caminho, lista_erros in erros_por_arquivo.items():
            print(f"  -> {Path(caminho).name}:")
            for erro in lista_erros:
                print(f"     {erro}")

        return {
            "erros_validacao": erros_por_arquivo,
            "validacao_concluida": False,
        }

    # Consolida em ordem cronológica real (ano, trimestre) antes de entregar ao Analista
    historico_trimestral = sorted(
        dados_extraidos.values(),
        key=lambda x: chave_ordenacao_trimestre(str(x.get("trimestre_referencia", ""))),
    )
    json_consolidado = json.dumps(historico_trimestral, indent=2, ensure_ascii=False)
    print(
        f"\n[Extrator] Histórico Trimestral Consolidado ({ticker}):\n{json_consolidado}\n"
    )

    bloco_ressalva = ""
    if erros_por_arquivo:
        resumo_ressalvas = "\n".join(
            f"Arquivo {Path(caminho).name}:\n" + "\n".join(lista)
            for caminho, lista in erros_por_arquivo.items()
        )
        bloco_ressalva = (
            "\n\nAVISO DE AUDITORIA: Após o limite máximo de tentativas de re-extração, "
            "as seguintes ressalvas contábeis permaneceram e devem ser citadas no relatório:\n"
            f"{resumo_ressalvas}"
        )
    else:
        print(
            "[Validador] Sucesso! Todos os trimestres passaram na auditoria matemática."
        )

    mensagem = HumanMessage(
        content=(
            f"Ativo: {ticker}.\n"
            f"Histórico de trimestres extraídos e auditados dos PDFs:\n{json_consolidado}"
            f"{bloco_ressalva}\n\n"
            "Siga rigorosamente o passo a passo das ferramentas para consolidar o ano e calcular os dois cenários."
        )
    )

    return {
        "messages": [mensagem],
        "erros_validacao": erros_por_arquivo,
        "validacao_concluida": True,
    }


def roteador_validador(state: AgentState) -> str:
    dados_extraidos = state.get("dados_extraidos") or {}
    if not dados_extraidos:
        return "fim"

    erros = state.get("erros_validacao") or {}
    tentativas = state.get("tentativas_extracao", 0)
    concluida = state.get("validacao_concluida", False)

    if erros and not concluida and tentativas < MAX_TENTATIVAS_EXTRACAO:
        print(
            f"[Loop] Retornando ao Extrator para corrigir {len(erros)} PDF(s) "
            f"(Tentativa {tentativas + 1}/{MAX_TENTATIVAS_EXTRACAO})..."
        )
        return "extrator"

    if erros:
        print(
            "[Loop] Limite de tentativas atingido. Seguindo ao Analista com ressalvas registradas."
        )

    return "analista"


def analista_node(state: AgentState) -> dict[str, Any]:
    mensagens = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in mensagens):
        protocolo = obter_protocolo(state.get("setor", "geral"))
        prompt_sistema = SystemMessage(
            content=(
                f"Você é um analista fundamentalista sênior especializado em {protocolo.nome}. "
                f"{protocolo.instrucoes}\n\n"
                "ROTEIRO OBRIGATÓRIO DE EXECUÇÃO DE FERRAMENTAS:\n"
                "1. Leia a 'visao_da_gestao' e o 'resumo_momento_operacional' que vieram do extrator.\n"
                "2. Baseado no tom da diretoria (otimista/cauteloso) e na evolução dos trimestres, "
                "defina de forma embasada o `momento_operacional` ('recuperacao', 'estavel' ou 'deteriorando').\n"
                "3. Chame `buscar_dados_mercado_yfinance` para obter o `preco_atual`. "
                "Para o número de ações e Patrimônio Líquido, use EXCLUSIVAMENTE `numero_total_acoes` e "
                "`patrimonio_liquido` extraídos do PDF mais recente.\n"
                "4. Chame `buscar_medias_historicas_yfinance` (ROE médio histórico) e "
                "`buscar_payout_medio_historico_yfinance` (que retorna `payout_medio_observado` e `payout_sustentavel` limitado a 1.0).\n"
                "5. Chame `projetar_lucro_anual_cenarios` passando a lista de `lucro_liquido_trimestral` "
                "em ordem cronológica e o `momento_operacional` definido no passo 2.\n"
                "6. Chame `calcular_multiplos_basicos` duas vezes (uma para o lucro anual conservador e outra para o mediano), "
                "usando `patrimonio_liquido` e `numero_total_acoes` do PDF mais recente.\n"
                "7. Chame `fluxo_de_caixa_futuro_buffet` (modelo DDM de 2 estágios com convergência na perpetuidade) duas vezes, sempre com "
                f"`taxa_desconto={TAXA_DESCONTO_PADRAO}`, `crescimento_perpetuo={CRESCIMENTO_PERPETUO_PADRAO}` "
                "e `num_acoes` igual ao `numero_total_acoes` do PDF mais recente:\n"
                "   - Execução A (Cenário Conservador): Use `lucro_anual_conservador`, `roe_medio_historico` e `payout_sustentavel` "
                "(sempre limitado a 1.0 no DDM; caso algum dado histórico falhe no Yahoo Finance ou `payout_sustentavel` seja 0, "
                "use `roe_reportado`/`payout_estimado` do PDF mais recente limitado a 1.0 e informe).\n"
                "   - Execução B (Cenário Mediano): Use `lucro_anual_mediano` e escolha justificadamente entre os parâmetros históricos "
                "(usando `payout_sustentavel`) ou `roe_reportado`/`payout_estimado` do PDF mais recente (limitado a 1.0).\n\n"
                "REGRAS ESTRITAS DE COMPLIANCE E FORMATAÇÃO:\n"
                "- JAMAIS invente, deduza ou faça contas de cabeça. Use exclusivamente os retornos das ferramentas.\n"
                "- JAMAIS use palavras de recomendação ('Compra', 'Venda', 'Manter', 'Comprar', 'Vender'). "
                "Apenas apresente a diferença percentual (Margem de Segurança / Desconto) entre o Preço Atual e o Preço Justo de cada cenário.\n"
                "- Estruture a resposta em Markdown limpo, iniciando com uma Tabela Comparativa lado a lado: "
                "Colunas: [Métrica | Cenário Conservador | Cenário Mediano | Referência Atual].\n"
                "- Crie uma seção 'Perspectivas da Gestão' resumindo o que a diretoria reportou nos documentos "
                "e se o tom justifica aplicar o cenário conservador ou mediano."
            )
        )
        mensagens = [prompt_sistema, *mensagens]

    print("[Analista] Processando dados e acionando ferramentas...")
    resposta = _obter_model_with_tools().invoke(mensagens)
    return {"messages": [resposta]}


# ==========================================
# 5. Construindo e compilando o Grafo
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("extrator", extrator_node)
workflow.add_node("validador", validador_node)
workflow.add_node("analista", analista_node)
workflow.add_node("ferramentas", ToolNode(ferramentas))

workflow.add_edge(START, "extrator")

# Do extrator vai OBRIGATORIAMENTE para o Validador
workflow.add_edge("extrator", "validador")

# Do validador, volta pro extrator, avança pro analista ou encerra se não houver PDFs
workflow.add_conditional_edges(
    "validador",
    roteador_validador,
    {"extrator": "extrator", "analista": "analista", "fim": END},
)

workflow.add_conditional_edges(
    "analista", tools_condition, {"tools": "ferramentas", "__end__": END}
)
workflow.add_edge("ferramentas", "analista")

app = workflow.compile()

# ==========================================
# 6. Execução em Lote (Agrupada por Ativo)
# ==========================================
if __name__ == "__main__":
    _garantir_api_key()

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
                        "dados_extraidos": {},
                        "erros_validacao": {},
                        "validacao_concluida": False,
                    }
                )

                mensagens_finais = resultado_final.get("messages") or []
                if not mensagens_finais:
                    print(
                        f"[AVISO] Nenhuma resposta gerada pelo pipeline para {ticker_atual}.\n"
                    )
                    continue

                # Extrai o texto puro (tratando caso o Gemini devolva lista de blocos ou string direta)
                conteudo_bruto = mensagens_finais[-1].content
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
