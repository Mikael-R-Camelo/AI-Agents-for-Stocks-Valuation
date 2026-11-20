import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.compliance import (
    anexar_alerta_compliance,
    auditar_compliance_relatorio,
    extrair_texto_mensagem,
    montar_feedback_compliance,
)
from agents.extrator import (
    carregar_cache_extracao,
    extrair_dados_do_pdf,
    salvar_cache_extracao,
)
from agents.prompts import montar_prompt_sistema_analista
from agents.sector_protocols import obter_protocolo
from config import (
    MAX_TENTATIVAS_COMPLIANCE,
    MAX_TENTATIVAS_EXTRACAO,
    TOLERANCIA_CAIXA,
    TOLERANCIA_ROE,
    cache_pdf_habilitado,
    obter_llm,
)
from graph.state import AgentState
from tools.utils import chave_ordenacao_trimestre

logger = logging.getLogger(__name__)


def validar_trimestre(dados: dict[str, Any]) -> list[str]:
    """Audita matematicamente um trimestre extraído (ROE anualizado e DFC opcional)."""
    erros: list[str] = []

    lucro = dados.get("lucro_liquido_trimestral")
    pl = dados.get("patrimonio_liquido")
    roe_reportado = dados.get("roe_reportado")
    payout = dados.get("payout_estimado")
    num_acoes = dados.get("numero_total_acoes")

    # VALIDAÇÃO 0: Campos estruturais obrigatórios
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

    # VALIDAÇÃO 1: Consistência do ROE (Trimestral anualizado vs ROE reportado)
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

    # VALIDAÇÃO 2: Consistência do Fluxo de Caixa (apenas quando DFC existir no PDF)
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
            erro_percentual_caixa = (
                abs(caixa_calculado - float(caixa_final)) / referencia
            )

            if erro_percentual_caixa > TOLERANCIA_CAIXA:
                erros.append(
                    f"- CAIXA: Inicial ({caixa_inicial:,.2f}) + Variação ({variacao_caixa:,.2f}) "
                    f"+ Efeito Cambial ({ajuste_cambio:,.2f}) resulta em {caixa_calculado:,.2f}, "
                    f"mas o Final extraído foi {caixa_final:,.2f}. "
                    "Verifique o efeito cambial na DFC ou corrija os saldos."
                )

    return erros


def extrator_node(state: AgentState, llm: Any = None) -> dict[str, Any]:
    """Extrai dados estruturados dos PDFs (com reprocessamento seletivo e leitura de cache)."""
    caminhos = state.get("caminhos_pdfs", [])
    dados_existentes = dict(state.get("dados_extraidos") or {})
    origem_cache = dict(state.get("origem_cache") or {})
    erros_anteriores = state.get("erros_validacao") or {}
    tentativas = state.get("tentativas_extracao", 0)
    usar_cache = state.get("usar_cache")
    if usar_cache is None:
        usar_cache = cache_pdf_habilitado()

    if not caminhos:
        return {
            "dados_extraidos": {},
            "erros_validacao": {},
            "origem_cache": {},
            "validacao_concluida": True,
        }

    # Em retry, reprocessa APENAS os PDFs reprovados no validador
    if erros_anteriores:
        alvos = [c for c in sorted(caminhos) if c in erros_anteriores]
    else:
        alvos = sorted(caminhos)

    modelo_llm = llm

    for caminho in alvos:
        nome_arq = Path(caminho).name
        feedback = (
            "\n".join(erros_anteriores[caminho])
            if caminho in erros_anteriores
            else None
        )

        # Leitura de cache apenas quando habilitado e fora de retry corretivo
        if usar_cache and feedback is None:
            dados_cacheados = carregar_cache_extracao(caminho)
            if dados_cacheados is not None:
                logger.info("[Extrator] Cache HIT para %s.", nome_arq)
                dados_cacheados["arquivo_origem"] = nome_arq
                dados_existentes[caminho] = dados_cacheados
                origem_cache[caminho] = True
                continue

        if modelo_llm is None:
            modelo_llm = obter_llm()

        if feedback:
            logger.info(
                "[Extrator] Reprocessando %s com feedback do validador...", nome_arq
            )
        else:
            logger.info("[Extrator] Extraindo dados de %s...", nome_arq)

        dados_tri = extrair_dados_do_pdf(
            caminho,
            feedback_correcao=feedback,
            llm=modelo_llm,
        )
        dados_tri["arquivo_origem"] = nome_arq
        dados_existentes[caminho] = dados_tri
        origem_cache[caminho] = False

    return {
        "dados_extraidos": dados_existentes,
        "erros_validacao": {},
        "origem_cache": origem_cache,
        "tentativas_extracao": tentativas + 1,
        "validacao_concluida": False,
    }


def validador_node(state: AgentState) -> dict[str, Any]:
    """Audita os trimestres extraídos e persiste no cache apenas os aprovados."""
    dados_extraidos = state.get("dados_extraidos") or {}
    origem_cache = dict(state.get("origem_cache") or {})
    ticker = state.get("ticker", "Desconhecido")
    tentativas = state.get("tentativas_extracao", 0)
    usar_cache = state.get("usar_cache")
    if usar_cache is None:
        usar_cache = cache_pdf_habilitado()

    if not dados_extraidos:
        logger.warning("[Validador] Nenhum PDF extraído para validar.")
        return {
            "erros_validacao": {},
            "validacao_concluida": True,
        }

    erros_por_arquivo: dict[str, list[str]] = {}
    for caminho, dados in dados_extraidos.items():
        erros_pdf = validar_trimestre(dados)
        if erros_pdf:
            erros_por_arquivo[caminho] = erros_pdf
        elif (
            usar_cache
            and not origem_cache.get(caminho, False)
            and Path(caminho).is_file()
        ):
            # Regra obrigatória: só grava no cache extrações APROVADAS pelo validador
            salvar_cache_extracao(caminho, dados)
            origem_cache[caminho] = True

    # Se há erros e ainda restam tentativas, devolve ao extrator sem poluir state["messages"]
    if erros_por_arquivo and tentativas < MAX_TENTATIVAS_EXTRACAO:
        logger.warning(
            "[Validador] Falhas matemáticas em %d PDF(s) (tentativa %d/%d).",
            len(erros_por_arquivo),
            tentativas,
            MAX_TENTATIVAS_EXTRACAO,
        )
        for caminho, lista_erros in erros_por_arquivo.items():
            logger.warning(
                "  -> %s:\n%s",
                Path(caminho).name,
                "\n".join(f"     {e}" for e in lista_erros),
            )

        return {
            "erros_validacao": erros_por_arquivo,
            "origem_cache": origem_cache,
            "validacao_concluida": False,
        }

    # Consolida em ordem cronológica real (ano, trimestre) antes de entregar ao Analista
    historico_trimestral = sorted(
        dados_extraidos.values(),
        key=lambda x: chave_ordenacao_trimestre(str(x.get("trimestre_referencia", ""))),
    )
    json_consolidado = json.dumps(historico_trimestral, indent=2, ensure_ascii=False)
    logger.info(
        "[Extrator] Histórico Trimestral Consolidado (%s):\n%s",
        ticker,
        json_consolidado,
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
        logger.info(
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
        "origem_cache": origem_cache,
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
        logger.info(
            "[Loop] Retornando ao Extrator para corrigir %d PDF(s) (Tentativa %d/%d)...",
            len(erros),
            tentativas + 1,
            MAX_TENTATIVAS_EXTRACAO,
        )
        return "extrator"

    if erros:
        logger.warning(
            "[Loop] Limite de tentativas atingido. Seguindo ao Analista com ressalvas registradas."
        )

    return "analista"


def analista_node(state: AgentState, model_with_tools: Any = None) -> dict[str, Any]:
    mensagens = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in mensagens):
        protocolo = obter_protocolo(state.get("setor", "geral"))
        prompt_sistema = SystemMessage(
            content=montar_prompt_sistema_analista(protocolo)
        )
        mensagens = [prompt_sistema, *mensagens]

    logger.info("[Analista] Processando dados e acionando ferramentas...")
    if model_with_tools is None:
        raise RuntimeError("model_with_tools não foi configurado no workflow.")

    resposta = model_with_tools.invoke(mensagens)
    return {"messages": [resposta]}


def compliance_node(state: AgentState) -> dict[str, Any]:
    """Audita deterministicamente o relatório final contra recomendações explícitas."""
    mensagens = state.get("messages") or []
    if not mensagens:
        return {"violacoes_compliance": []}

    texto_relatorio = extrair_texto_mensagem(mensagens[-1].content)
    resultado = auditar_compliance_relatorio(texto_relatorio)

    if resultado.aprovado:
        logger.info("[Compliance] Relatório aprovado sem recomendações explícitas.")
        return {"violacoes_compliance": []}

    tentativas = state.get("tentativas_compliance", 0) + 1
    if tentativas <= MAX_TENTATIVAS_COMPLIANCE:
        logger.warning(
            "[Compliance] Violação detectada (%d). Solicitando reescrita limpa (Tentativa %d/%d)...",
            len(resultado.violacoes),
            tentativas,
            MAX_TENTATIVAS_COMPLIANCE,
        )
        msg_correcao = HumanMessage(
            content=montar_feedback_compliance(resultado.violacoes)
        )
        return {
            "messages": [msg_correcao],
            "tentativas_compliance": tentativas,
            "violacoes_compliance": resultado.violacoes,
        }

    logger.warning(
        "[Compliance] Limite de reescritas atingido com violações pendentes. "
        "Registrando alerta explícito no topo do relatório."
    )
    texto_com_alerta = anexar_alerta_compliance(texto_relatorio, resultado.violacoes)
    return {
        "messages": [AIMessage(content=texto_com_alerta)],
        "tentativas_compliance": tentativas,
        "violacoes_compliance": resultado.violacoes,
    }


def roteador_compliance(state: AgentState) -> str:
    violacoes = state.get("violacoes_compliance") or []
    tentativas = state.get("tentativas_compliance", 0)

    if violacoes and tentativas <= MAX_TENTATIVAS_COMPLIANCE:
        return "analista"
    return "fim"
