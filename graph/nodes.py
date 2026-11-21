import itertools
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
    CRESCIMENTO_PERPETUO_PADRAO,
    MAX_TENTATIVAS_COMPLIANCE,
    MAX_TENTATIVAS_EXTRACAO,
    TAXA_DESCONTO_PADRAO,
    TOLERANCIA_CAIXA,
    TOLERANCIA_ROE,
    cache_pdf_habilitado,
    obter_llm,
    obter_modelo_gemini_fallback,
)
from graph.state import (
    AgentState,
    MomentoOperacional,
    MultiplosCenario,
    ParametrosAvaliacao,
    ProjecaoLucrosAnuais,
    ResultadoQuantitativo,
    ValuationCenario,
)
from tools.market_data import (
    buscar_dados_mercado_yfinance,
    buscar_medias_historicas_yfinance,
    buscar_payout_medio_historico_yfinance,
    calcular_multiplos_basicos,
    projetar_lucro_anual_cenarios,
)
from tools.utils import chave_ordenacao_trimestre
from tools.valuation import fluxo_de_caixa_futuro_buffet

logger = logging.getLogger(__name__)

_LIMIAR_VARIACAO_MOMENTO = 0.02
_MARCADORES_ERRO_TRANSITORIO_LLM = (
    "503",
    "unavailable",
    "429",
    "resource_exhausted",
    "rate limit",
    "high demand",
    "overloaded",
    "service unavailable",
    "timeout",
    "timed out",
    "connection",
    "bad gateway",
    "502",
    "504",
    "500",
    "internal error",
)


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


def _classificar_momento_operacional(lucros: list[float]) -> MomentoOperacional:
    """
    Classifica deterministicamente o momento operacional pela trajetória cronológica
    dos lucros trimestrais, preservando direção correta mesmo em prejuízos.
    """
    if len(lucros) <= 1:
        return "estavel"

    anteriores = lucros[:-1]
    ultimo = lucros[-1]
    media_anterior = sum(anteriores) / len(anteriores)

    if media_anterior == 0:
        if ultimo > 0:
            return "recuperacao"
        if ultimo < 0:
            return "deteriorando"
        return "estavel"

    variacao = (ultimo - media_anterior) / abs(media_anterior)
    expansao_consistente = len(lucros) >= 3 and all(
        atual > anterior for anterior, atual in itertools.pairwise(lucros)
    )
    queda_consistente = len(lucros) >= 3 and all(
        atual < anterior for anterior, atual in itertools.pairwise(lucros)
    )

    if variacao > _LIMIAR_VARIACAO_MOMENTO or (expansao_consistente and variacao > 0):
        return "recuperacao"
    if variacao < -_LIMIAR_VARIACAO_MOMENTO or (queda_consistente and variacao < 0):
        return "deteriorando"
    return "estavel"


def _executar_ddm_seguro(
    *,
    lucro_base: float,
    payout: float | None,
    roe: float,
    num_acoes: float,
    preco_atual: float | None,
) -> tuple[ValuationCenario | None, str | None]:
    """Executa o DDM de 2 Estágios capturando erros de domínio (ex.: prejuízo ou payout zero)."""
    if preco_atual is None or preco_atual <= 0:
        return None, "Cotação atual indisponível no Yahoo Finance."
    if payout is None or payout <= 0:
        return None, "Payout positivo indisponível para aplicação do modelo DDM."

    try:
        bruto = fluxo_de_caixa_futuro_buffet.invoke(
            {
                "lucro_base": lucro_base,
                "payout": min(float(payout), 1.0),
                "roe": roe,
                "num_acoes": num_acoes,
                "preco_atual": preco_atual,
                "taxa_desconto": TAXA_DESCONTO_PADRAO,
                "crescimento_perpetuo": CRESCIMENTO_PERPETUO_PADRAO,
            }
        )
    except ValueError as exc:
        return None, str(exc)

    return (
        ValuationCenario(
            preco_justo=bruto["Preço Teto / Justo (R$)"],
            market_cap_justo=bruto["Market Cap Justo (R$)"],
            margem_seguranca=bruto["Margem de Segurança (%)"],
            upside_downside=bruto["Upside Real (%)"],
            crescimento_estagio_1=bruto["Crescimento Sustentável (%)"],
            payout_terminal=bruto["Payout Terminal Sustentável (%)"],
            taxa_desconto_pct=bruto["Taxa de Desconto Ke (%)"],
            crescimento_perpetuo_pct=bruto["Crescimento Perpétuo (%)"],
        ),
        None,
    )


def _formatar_moeda(valor: float | None, moeda: str = "R$") -> str:
    if valor is None:
        return "N/A"
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{moeda} {texto}"


def _formatar_pct(valor: float | None, sinal: bool = False) -> str:
    if valor is None:
        return "N/A"
    fmt = f"{valor:+.2f}%" if sinal else f"{valor:.2f}%"
    return fmt.replace(".", ",")


def _formatar_numero(valor: float | None, casas: int = 2) -> str:
    if valor is None:
        return "N/A"
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _logar_resumo_quantitativo(resultado: ResultadoQuantitativo) -> None:
    """Imprime resumo tabular determinístico no log antes de acionar o Analista."""
    vc = resultado.valuation_conservador
    vm = resultado.valuation_mediano
    mc = resultado.multiplos_conservador
    mm = resultado.multiplos_mediano
    p = resultado.parametros

    linhas = [
        f"[Calculadora] Resumo Quantitativo ({resultado.ticker} | Setor: {resultado.setor})",
        (
            f"  Cotação Atual : {_formatar_moeda(resultado.preco_atual)} ({resultado.moeda}) | "
            f"Momento: {p.momento_operacional} | Ke={p.taxa_desconto:.1%} | g={p.crescimento_perpetuo:.1%}"
        ),
        (
            f"  Base PDF ({p.trimestre_mais_recente}): Ações={p.num_acoes_pdf:,.0f} | "
            f"PL={_formatar_moeda(p.patrimonio_liquido_pdf)} | Fonte Payout: {p.fonte_payout}"
        ),
        "  ------------------------------------------------------------------------",
        f"  {'Métrica':<24} | {'Conservador':>20} | {'Mediano':>20}",
        "  ------------------------------------------------------------------------",
        f"  {'Lucro Anual Proj.':<24} | {_formatar_moeda(resultado.projecao_lucros.lucro_anual_conservador):>20} | {_formatar_moeda(resultado.projecao_lucros.lucro_anual_mediano):>20}",
        f"  {'Preço Justo / Teto':<24} | {_formatar_moeda(vc.preco_justo if vc else None):>20} | {_formatar_moeda(vm.preco_justo if vm else None):>20}",
        f"  {'Margem de Segurança':<24} | {_formatar_pct(vc.margem_seguranca if vc else None, sinal=True):>20} | {_formatar_pct(vm.margem_seguranca if vm else None, sinal=True):>20}",
        f"  {'Upside / Downside':<24} | {_formatar_pct(vc.upside_downside if vc else None, sinal=True):>20} | {_formatar_pct(vm.upside_downside if vm else None, sinal=True):>20}",
        f"  {'LPA':<24} | {_formatar_moeda(mc.LPA if mc else None):>20} | {_formatar_moeda(mm.LPA if mm else None):>20}",
        f"  {'VPA':<24} | {_formatar_moeda(mc.VPA if mc else None):>20} | {_formatar_moeda(mm.VPA if mm else None):>20}",
        f"  {'P/L':<24} | {_formatar_numero(mc.P_L if mc else None):>20} | {_formatar_numero(mm.P_L if mm else None):>20}",
        f"  {'P/VP':<24} | {_formatar_numero(mc.P_VP if mc else None):>20} | {_formatar_numero(mm.P_VP if mm else None):>20}",
        "  ------------------------------------------------------------------------",
    ]
    logger.info("\n%s", "\n".join(linhas))


def _salvar_json_quantitativo(
    resultado: ResultadoQuantitativo, caminhos_pdfs: list[str]
) -> Path | None:
    """Persiste `Valuation_<TICKER>_quant.json` na pasta do ativo quando os PDFs existem em disco."""
    if not caminhos_pdfs:
        return None

    primeiro_pdf = Path(caminhos_pdfs[0])
    if not primeiro_pdf.is_file():
        return None

    destino = primeiro_pdf.parent / f"Valuation_{resultado.ticker}_quant.json"
    try:
        caminho_tmp = destino.with_suffix(".tmp")
        caminho_tmp.write_text(
            resultado.model_dump_json(indent=2),
            encoding="utf-8",
        )
        caminho_tmp.replace(destino)
        logger.info("[Calculadora] Resultado quantitativo salvo em: %s", destino)
        return destino
    except OSError as exc:
        logger.warning(
            "[Calculadora] Não foi possível salvar JSON quantitativo em '%s': %s",
            destino,
            exc,
        )
        return None


def _eh_erro_transitorio_llm(exc: Exception) -> bool:
    """Detecta erros transitórios de API/rede do LLM (503, 429, timeout, conexão)."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 429, 500, 502, 503, 504}:
        return True

    nome_classe = type(exc).__name__.lower()
    if any(
        termo in nome_classe
        for termo in ("servererror", "serviceunavailable", "timeout", "httperror", "ratelimit")
    ):
        return True

    mensagem = str(exc).lower()
    return any(marcador in mensagem for marcador in _MARCADORES_ERRO_TRANSITORIO_LLM)


def gerar_relatorio_fallback_quantitativo(
    *,
    resultado_quantitativo: dict[str, Any],
    dados_extraidos: dict[str, dict[str, Any]],
    erro_analista: str,
    erros_validacao: dict[str, list[str]] | None = None,
) -> str:
    """Gera relatório Markdown completo e em compliance quando o LLM está indisponível."""
    resultado = ResultadoQuantitativo.model_validate(resultado_quantitativo)
    vc = resultado.valuation_conservador
    vm = resultado.valuation_mediano
    mc = resultado.multiplos_conservador
    mm = resultado.multiplos_mediano
    p = resultado.parametros
    proj = resultado.projecao_lucros

    historico_ordenado = sorted(
        dados_extraidos.values(),
        key=lambda x: chave_ordenacao_trimestre(str(x.get("trimestre_referencia", ""))),
    )

    linhas = [
        (
            "> **AVISO — RELATÓRIO EM MODO DE FALLBACK QUANTITATIVO:** "
            "A síntese qualitativa do Agente Analista ficou pendente devido à indisponibilidade "
            f"temporária do serviço de LLM (`{erro_analista}`). Todos os cálculos e tabelas abaixo "
            "foram gerados integralmente pelo motor determinístico em Python."
        ),
        "",
        f"### Tabela Comparativa de Valuation e Múltiplos ({resultado.ticker})",
        "",
        "| Métrica | Cenário Conservador | Cenário Mediano | Referência Atual |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Lucro Líquido Projetado (R$)** | {_formatar_moeda(proj.lucro_anual_conservador)} | {_formatar_moeda(proj.lucro_anual_mediano)} | {_formatar_moeda(proj.soma_acumulada_ano)} *({proj.trimestres_realizados}T acum.)* |",
        f"| **Preço Justo / Teto (R$)** | {_formatar_moeda(vc.preco_justo if vc else None)} | {_formatar_moeda(vm.preco_justo if vm else None)} | {_formatar_moeda(resultado.preco_atual)} |",
        f"| **Margem de Segurança (%)** | {_formatar_pct(vc.margem_seguranca if vc else None, sinal=True)} | {_formatar_pct(vm.margem_seguranca if vm else None, sinal=True)} | — |",
        f"| **Upside / Downside (%)** | {_formatar_pct(vc.upside_downside if vc else None, sinal=True)} | {_formatar_pct(vm.upside_downside if vm else None, sinal=True)} | — |",
        f"| **LPA (R$)** | {_formatar_moeda(mc.LPA if mc else None)} | {_formatar_moeda(mm.LPA if mm else None)} | — |",
        f"| **VPA (R$)** | {_formatar_moeda(mc.VPA if mc else None)} | {_formatar_moeda(mm.VPA if mm else None)} | {_formatar_moeda(mm.VPA if mm else None)} |",
        f"| **P/L** | {_formatar_numero(mc.P_L if mc else None)} | {_formatar_numero(mm.P_L if mm else None)} | — |",
        f"| **P/VP** | {_formatar_numero(mc.P_VP if mc else None)} | {_formatar_numero(mm.P_VP if mm else None)} | {_formatar_numero(mm.P_VP if mm else None)} |",
        f"| **ROE Calculado (%)** | {_formatar_pct(mc.ROE_calculado * 100 if mc and mc.ROE_calculado is not None else None)} | {_formatar_pct(mm.ROE_calculado * 100 if mm and mm.ROE_calculado is not None else None)} | {_formatar_pct(p.roe_pdf_recente * 100)} *({p.trimestre_mais_recente})* |",
        f"| **Crescimento Estágio 1 (%)** | {_formatar_pct(vc.crescimento_estagio_1 if vc else None)} | {_formatar_pct(vm.crescimento_estagio_1 if vm else None)} | — |",
        f"| **Payout Terminal Sustentável (%)** | {_formatar_pct(vc.payout_terminal if vc else None)} | {_formatar_pct(vm.payout_terminal if vm else None)} | — |",
        "",
        "---",
        "",
        "### Parâmetros Utilizados na Avaliação",
        "",
        f"- **Cotação Atual:** {_formatar_moeda(resultado.preco_atual)} ({resultado.moeda})",
        f"- **Setor / Protocolo:** `{resultado.setor}`",
        f"- **Momento Operacional (classificação determinística):** `{p.momento_operacional}`",
        f"- **Base Acionária (PDF {p.trimestre_mais_recente}):** {_formatar_numero(p.num_acoes_pdf, 0)} ações",
        f"- **Patrimônio Líquido (PDF {p.trimestre_mais_recente}):** {_formatar_moeda(p.patrimonio_liquido_pdf)}",
        f"- **Taxa de Desconto ($K_e$):** {_formatar_pct(p.taxa_desconto * 100)}",
        f"- **Crescimento Perpétuo ($g$):** {_formatar_pct(p.crescimento_perpetuo * 100)}",
        f"- **ROE Conservador / Mediano:** {_formatar_pct(p.roe_usado_conservador * 100)} / {_formatar_pct(p.roe_usado_mediano * 100)}",
        f"- **Payout Conservador / Mediano:** {_formatar_pct(p.payout_usado_conservador * 100 if p.payout_usado_conservador is not None else None)} / {_formatar_pct(p.payout_usado_mediano * 100 if p.payout_usado_mediano is not None else None)} *(fonte histórica: `{p.fonte_payout}`)*",
    ]

    if resultado.erro_dados_mercado:
        linhas.append(f"- **Alerta de Mercado:** {resultado.erro_dados_mercado}")
    if resultado.erro_valuation_conservador:
        linhas.append(
            f"- **Alerta Valuation Conservador:** {resultado.erro_valuation_conservador}"
        )
    if resultado.erro_valuation_mediano:
        linhas.append(
            f"- **Alerta Valuation Mediano:** {resultado.erro_valuation_mediano}"
        )
    for aviso in resultado.avisos:
        linhas.append(f"- **Aviso:** {aviso}")

    if erros_validacao:
        linhas.extend(["", "### Ressalvas da Auditoria Contábil", ""])
        for caminho, lista in erros_validacao.items():
            linhas.append(f"- **{Path(caminho).name}:** " + " ".join(lista))

    if historico_ordenado:
        linhas.extend(["", "---", "", "### Contexto Qualitativo Extraído dos PDFs", ""])
        for tri in historico_ordenado:
            ref = tri.get("trimestre_referencia", "N/A")
            resumo = tri.get("resumo_momento_operacional", "")
            visao = tri.get("visao_da_gestao", "")
            linhas.append(f"- **{ref} — Momento Operacional:** {resumo}")
            linhas.append(f"  - **Visão da Gestão:** {visao}")

    linhas.extend(
        [
            "",
            "---",
            "",
            (
                "*Aviso legal: este relatório tem caráter exclusivamente informativo e técnico, "
                "não constitui recomendação de compra, venda ou manutenção de valores mobiliários.*"
            ),
        ]
    )
    return "\n".join(linhas)


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

    # Se há erros e ainda restam tentativas, devolve ao extrator
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

    if not erros_por_arquivo:
        logger.info(
            "[Validador] Sucesso! Todos os trimestres passaram na auditoria matemática."
        )

    return {
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
            "[Loop] Limite de tentativas atingido. Seguindo à Calculadora com ressalvas registradas."
        )

    return "calculadora"


def calculadora_node(state: AgentState) -> dict[str, Any]:
    """
    Executa toda a cadeia financeira determinística (zero tokens LLM),
    registra resumo no terminal, persiste JSON quantitativo e prepara a entrada do Analista.
    """
    dados_extraidos = state.get("dados_extraidos") or {}
    ticker = state.get("ticker", "Desconhecido")
    setor = state.get("setor", "geral")
    caminhos_pdfs = state.get("caminhos_pdfs") or []
    erros_validacao = state.get("erros_validacao") or {}

    historico_ordenado = sorted(
        dados_extraidos.values(),
        key=lambda x: chave_ordenacao_trimestre(str(x.get("trimestre_referencia", ""))),
    )
    mais_recente = historico_ordenado[-1]

    lucros_trimestrais = [
        float(tri["lucro_liquido_trimestral"]) for tri in historico_ordenado
    ]
    num_acoes_pdf = float(mais_recente["numero_total_acoes"])
    pl_pdf = float(mais_recente["patrimonio_liquido"])
    roe_pdf = float(mais_recente["roe_reportado"])
    payout_pdf = float(mais_recente["payout_estimado"])
    trimestre_recente = str(mais_recente.get("trimestre_referencia", "N/A"))

    momento_operacional = _classificar_momento_operacional(lucros_trimestrais)
    avisos: list[str] = []

    # 1. Cotação atual (sem sharesOutstanding / VPA do Yahoo Finance)
    dados_mercado = buscar_dados_mercado_yfinance.invoke({"ticker": ticker})
    preco_atual = dados_mercado.get("preco_atual")
    moeda = str(dados_mercado.get("moeda") or "BRL")
    erro_dados_mercado = dados_mercado.get("erro")

    # 2. ROE médio histórico (4 anos)
    dados_roe_hist = buscar_medias_historicas_yfinance.invoke({"ticker": ticker})
    roe_historico = dados_roe_hist.get("roe_medio_historico")
    if roe_historico is None:
        avisos.append(
            f"ROE histórico indisponível no Yahoo Finance ({dados_roe_hist.get('erro', 'sem dados')}); "
            f"Cenário Conservador utilizou roe_reportado do {trimestre_recente}."
        )
    roe_usado_conservador = float(roe_historico) if roe_historico is not None else roe_pdf
    roe_usado_mediano = roe_pdf

    # 3. Payout médio histórico (com fallback JCP e teto sustentável 1.0)
    dados_payout_hist = buscar_payout_medio_historico_yfinance.invoke({"ticker": ticker})
    payout_obs = dados_payout_hist.get("payout_medio_observado")
    payout_sust = dados_payout_hist.get("payout_sustentavel")
    fonte_payout = str(dados_payout_hist.get("fonte_payout") or "indisponivel")
    if dados_payout_hist.get("aviso"):
        avisos.append(str(dados_payout_hist["aviso"]))

    payout_pdf_seguro = min(payout_pdf, 1.0) if payout_pdf > 0 else None
    if payout_sust is not None and float(payout_sust) > 0:
        payout_usado_conservador: float | None = min(float(payout_sust), 1.0)
    else:
        payout_usado_conservador = payout_pdf_seguro
        fonte_payout = "pdf_recente_fallback"
        avisos.append(
            f"Payout histórico sustentável indisponível/zerado no Yahoo Finance; "
            f"Cenário Conservador utilizou payout_estimado do {trimestre_recente}."
        )

    payout_usado_mediano = (
        payout_pdf_seguro
        if payout_pdf_seguro is not None
        else payout_usado_conservador
    )

    # 4. Projeção anual de lucros (Conservador e Mediano)
    projecao_raw = projetar_lucro_anual_cenarios.invoke(
        {
            "lucros_trimestrais_realizados": lucros_trimestrais,
            "momento_operacional": momento_operacional,
        }
    )
    projecao = ProjecaoLucrosAnuais.model_validate(projecao_raw)

    # 5. Múltiplos básicos (estritamente com PL e ações do PDF)
    multiplos_cons: MultiplosCenario | None = None
    multiplos_med: MultiplosCenario | None = None
    if preco_atual is not None and preco_atual > 0 and num_acoes_pdf > 0:
        multiplos_cons = MultiplosCenario.model_validate(
            calcular_multiplos_basicos.invoke(
                {
                    "preco_atual": float(preco_atual),
                    "lucro_anual": projecao.lucro_anual_conservador,
                    "patrimonio_liquido": pl_pdf,
                    "num_acoes": num_acoes_pdf,
                }
            )
        )
        multiplos_med = MultiplosCenario.model_validate(
            calcular_multiplos_basicos.invoke(
                {
                    "preco_atual": float(preco_atual),
                    "lucro_anual": projecao.lucro_anual_mediano,
                    "patrimonio_liquido": pl_pdf,
                    "num_acoes": num_acoes_pdf,
                }
            )
        )

    # 6. Valuation DDM de 2 Estágios (Conservador e Mediano)
    val_cons, erro_val_cons = _executar_ddm_seguro(
        lucro_base=projecao.lucro_anual_conservador,
        payout=payout_usado_conservador,
        roe=roe_usado_conservador,
        num_acoes=num_acoes_pdf,
        preco_atual=float(preco_atual) if preco_atual is not None else None,
    )
    val_med, erro_val_med = _executar_ddm_seguro(
        lucro_base=projecao.lucro_anual_mediano,
        payout=payout_usado_mediano,
        roe=roe_usado_mediano,
        num_acoes=num_acoes_pdf,
        preco_atual=float(preco_atual) if preco_atual is not None else None,
    )

    resultado = ResultadoQuantitativo(
        ticker=ticker,
        setor=setor,
        preco_atual=float(preco_atual) if preco_atual is not None else None,
        moeda=moeda,
        erro_dados_mercado=erro_dados_mercado,
        parametros=ParametrosAvaliacao(
            momento_operacional=momento_operacional,
            roe_historico=float(roe_historico) if roe_historico is not None else None,
            roe_pdf_recente=roe_pdf,
            roe_usado_conservador=roe_usado_conservador,
            roe_usado_mediano=roe_usado_mediano,
            payout_medio_observado=(
                float(payout_obs) if payout_obs is not None else None
            ),
            payout_sustentavel=(
                float(payout_sust) if payout_sust is not None else None
            ),
            payout_pdf_recente=payout_pdf,
            payout_usado_conservador=payout_usado_conservador,
            payout_usado_mediano=payout_usado_mediano,
            fonte_payout=fonte_payout,
            num_acoes_pdf=num_acoes_pdf,
            patrimonio_liquido_pdf=pl_pdf,
            trimestre_mais_recente=trimestre_recente,
            taxa_desconto=TAXA_DESCONTO_PADRAO,
            crescimento_perpetuo=CRESCIMENTO_PERPETUO_PADRAO,
        ),
        projecao_lucros=projecao,
        multiplos_conservador=multiplos_cons,
        multiplos_mediano=multiplos_med,
        valuation_conservador=val_cons,
        valuation_mediano=val_med,
        erro_valuation_conservador=erro_val_cons,
        erro_valuation_mediano=erro_val_med,
        avisos=avisos,
    )

    _logar_resumo_quantitativo(resultado)
    _salvar_json_quantitativo(resultado, caminhos_pdfs)

    resultado_dict = resultado.model_dump()
    contexto_qualitativo = [
        {
            "trimestre_referencia": tri.get("trimestre_referencia"),
            "arquivo_origem": tri.get("arquivo_origem"),
            "resumo_momento_operacional": tri.get("resumo_momento_operacional"),
            "visao_da_gestao": tri.get("visao_da_gestao"),
        }
        for tri in historico_ordenado
    ]

    bloco_ressalva = ""
    if erros_validacao:
        resumo_ressalvas = "\n".join(
            f"Arquivo {Path(caminho).name}:\n" + "\n".join(lista)
            for caminho, lista in erros_validacao.items()
        )
        bloco_ressalva = (
            "\n\nAVISO DE AUDITORIA: Após o limite máximo de tentativas de re-extração, "
            "as seguintes ressalvas contábeis permaneceram e devem ser citadas no relatório:\n"
            f"{resumo_ressalvas}"
        )

    mensagem = HumanMessage(
        content=(
            f"Ativo: {ticker}.\n"
            "RESULTADO QUANTITATIVO PRÉ-CALCULADO (Python determinístico — NÃO recalcule):\n"
            f"{json.dumps(resultado_dict, indent=2, ensure_ascii=False)}\n\n"
            "CONTEXTO QUALITATIVO DOS TRIMESTRES (em ordem cronológica):\n"
            f"{json.dumps(contexto_qualitativo, indent=2, ensure_ascii=False)}"
            f"{bloco_ressalva}\n\n"
            "Redija o relatório técnico final em Markdown reproduzindo fielmente todos os números "
            "do resultado quantitativo e integrando as perspectivas qualitativas da gestão."
        )
    )

    return {
        "messages": [mensagem],
        "resultado_quantitativo": resultado_dict,
        "analise_llm_indisponivel": False,
        "erro_analista": None,
    }


def analista_node(
    state: AgentState,
    llm: Any = None,
    fallback_llm: Any = None,
) -> dict[str, Any]:
    """
    Executa chamada única ao LLM para redação qualitativa do relatório.
    Em erro transitório (ex.: 503/429/rede), tenta GEMINI_FALLBACK_MODEL (se configurado)
    e, se persistir, gera relatório determinístico de fallback sem abortar o pipeline.
    """
    mensagens = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in mensagens):
        protocolo = obter_protocolo(state.get("setor", "geral"))
        prompt_sistema = SystemMessage(
            content=montar_prompt_sistema_analista(protocolo)
        )
        mensagens = [prompt_sistema, *mensagens]

    modelo_primario = llm if llm is not None else obter_llm()
    logger.info("[Analista] Redigindo relatório qualitativo final (chamada única)...")

    try:
        resposta = modelo_primario.invoke(mensagens)
        return {
            "messages": [resposta],
            "analise_llm_indisponivel": False,
            "erro_analista": None,
        }
    except Exception as exc_primario:
        if not _eh_erro_transitorio_llm(exc_primario):
            raise

        logger.warning(
            "[Analista] Falha transitória no modelo primário: %s", exc_primario
        )

        modelo_reserva = fallback_llm
        if modelo_reserva is None and llm is None:
            nome_fallback = obter_modelo_gemini_fallback()
            if nome_fallback:
                logger.info(
                    "[Analista] Tentando modelo Gemini fallback configurado: %s",
                    nome_fallback,
                )
                modelo_reserva = obter_llm(modelo=nome_fallback)

        if modelo_reserva is not None:
            try:
                resposta_fallback = modelo_reserva.invoke(mensagens)
                return {
                    "messages": [resposta_fallback],
                    "analise_llm_indisponivel": False,
                    "erro_analista": None,
                }
            except Exception as exc_fallback:
                if not _eh_erro_transitorio_llm(exc_fallback):
                    raise
                erro_final = f"Primário: {exc_primario} | Fallback: {exc_fallback}"
                logger.warning(
                    "[Analista] Modelo fallback também indisponível (%s). "
                    "Gerando relatório determinístico quantitativo.",
                    exc_fallback,
                )
        else:
            erro_final = str(exc_primario)
            logger.warning(
                "[Analista] Serviço de LLM indisponível sem modelo reserva. "
                "Gerando relatório determinístico quantitativo."
            )

        texto_fallback = gerar_relatorio_fallback_quantitativo(
            resultado_quantitativo=state.get("resultado_quantitativo") or {},
            dados_extraidos=state.get("dados_extraidos") or {},
            erro_analista=erro_final,
            erros_validacao=state.get("erros_validacao") or {},
        )
        return {
            "messages": [AIMessage(content=texto_fallback)],
            "analise_llm_indisponivel": True,
            "erro_analista": erro_final,
        }


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

    if state.get("analise_llm_indisponivel", False):
        logger.warning(
            "[Compliance] Violação detectada em modo fallback (LLM indisponível). "
            "Anexando alerta sem tentar reescrita."
        )
        texto_com_alerta = anexar_alerta_compliance(
            texto_relatorio, resultado.violacoes
        )
        return {
            "messages": [AIMessage(content=texto_com_alerta)],
            "violacoes_compliance": resultado.violacoes,
        }

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
    if state.get("analise_llm_indisponivel", False):
        return "fim"

    violacoes = state.get("violacoes_compliance") or []
    tentativas = state.get("tentativas_compliance", 0)

    if violacoes and tentativas <= MAX_TENTATIVAS_COMPLIANCE:
        return "analista"
    return "fim"
