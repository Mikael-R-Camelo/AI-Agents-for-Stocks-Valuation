import itertools
import json
import logging
import statistics
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
    CONSTANTE_GRAHAM_PADRAO,
    CRESCIMENTO_PERPETUO_PADRAO,
    MAX_TENTATIVAS_COMPLIANCE,
    MAX_TENTATIVAS_EXTRACAO,
    TAXA_DESCONTO_PADRAO,
    TOLERANCIA_CAIXA,
    TOLERANCIA_ROE,
    YIELD_MINIMO_BAZIN_CONSERVADOR,
    YIELD_MINIMO_BAZIN_PADRAO,
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
    ResumoTriangulacaoCenario,
    TriangulacaoValuation,
    ValuationBazin,
    ValuationBazinCenario,
    ValuationCenario,
    ValuationGraham,
)
from tools.market_data import (
    buscar_dados_mercado_yfinance,
    buscar_medias_historicas_yfinance,
    buscar_payout_medio_historico_yfinance,
    calcular_multiplos_basicos,
    projetar_lucro_anual_cenarios,
)
from tools.utils import chave_ordenacao_trimestre
from tools.valuation import (
    calcular_preco_teto_bazin,
    calcular_valor_intrinseco_graham,
    fluxo_de_caixa_futuro_buffet,
)

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


def _executar_bazin_seguro(
    *,
    dpa: float | None,
    preco_atual: float | None,
    yield_minimo: float,
) -> tuple[ValuationBazin | None, str | None]:
    """Executa Décio Bazin isolando erros de domínio (ex.: DPA <= 0 ou cotação indisponível)."""
    if preco_atual is None or preco_atual <= 0:
        return None, "Cotação atual indisponível no Yahoo Finance."
    if dpa is None or dpa <= 0:
        return None, "DPA projetado positivo indisponível para o método Décio Bazin."

    try:
        bruto = calcular_preco_teto_bazin.invoke(
            {
                "dpa": dpa,
                "preco_atual": preco_atual,
                "yield_minimo": yield_minimo,
            }
        )
    except ValueError as exc:
        return None, str(exc)

    return ValuationBazin.model_validate(bruto), None


def _executar_bazin_cenario(
    *,
    lpa_exato: float | None,
    payout: float | None,
    preco_atual: float | None,
) -> ValuationBazinCenario:
    """Calcula Décio Bazin nos dois patamares oficiais (6% clássico e 8% juros Brasil)."""
    dpa_exato: float | None = None
    if lpa_exato is not None and payout is not None and payout > 0:
        dpa_exato = lpa_exato * min(float(payout), 1.0)

    dy_6, erro_6 = _executar_bazin_seguro(
        dpa=dpa_exato,
        preco_atual=preco_atual,
        yield_minimo=YIELD_MINIMO_BAZIN_PADRAO,
    )
    dy_8, erro_8 = _executar_bazin_seguro(
        dpa=dpa_exato,
        preco_atual=preco_atual,
        yield_minimo=YIELD_MINIMO_BAZIN_CONSERVADOR,
    )
    return ValuationBazinCenario(
        dy_6=dy_6,
        dy_8=dy_8,
        erro_dy_6=erro_6,
        erro_dy_8=erro_8,
    )


def _executar_graham_seguro(
    *,
    lpa_exato: float | None,
    vpa_exato: float | None,
    preco_atual: float | None,
) -> tuple[ValuationGraham | None, str | None]:
    """Executa Benjamin Graham em precisão total isolando prejuízo ou PL negativo."""
    if preco_atual is None or preco_atual <= 0:
        return None, "Cotação atual indisponível no Yahoo Finance."
    if lpa_exato is None or vpa_exato is None:
        return None, "LPA ou VPA indisponíveis para a fórmula de Benjamin Graham."

    try:
        bruto = calcular_valor_intrinseco_graham.invoke(
            {
                "lpa": lpa_exato,
                "vpa": vpa_exato,
                "preco_atual": preco_atual,
                "constante_graham": CONSTANTE_GRAHAM_PADRAO,
            }
        )
    except ValueError as exc:
        return None, str(exc)

    return ValuationGraham.model_validate(bruto), None


def _montar_resumo_triangulacao_cenario(
    *,
    ddm: ValuationCenario | None,
    bazin: ValuationBazinCenario,
    graham: ValuationGraham | None,
) -> ResumoTriangulacaoCenario:
    """
    Consolida triangulação usando estritamente 1 representante por metodologia
    (DDM, Bazin 6% Clássico e Graham), sem peso duplo para Bazin 8%.
    """
    bazin_6 = bazin.dy_6
    bazin_8 = bazin.dy_8

    # Apenas 1 representante por metodologia entra nas estatísticas (mín/mediana/máx)
    representantes: list[tuple[str, float]] = []
    if ddm is not None:
        representantes.append(("ddm", ddm.preco_justo))
    if bazin_6 is not None:
        representantes.append(("bazin_6", bazin_6.preco_justo))
    if graham is not None:
        representantes.append(("graham", graham.preco_justo))

    modelos_triangulados = [nome for nome, _ in representantes]
    precos = [preco for _, preco in representantes]

    minimo = round(min(precos), 2) if precos else None
    mediana = round(float(statistics.median(precos)), 2) if precos else None
    maximo = round(max(precos), 2) if precos else None
    dispersao_pct = (
        round(((maximo / minimo) - 1.0) * 100, 2)
        if len(precos) >= 2 and minimo is not None and minimo > 0 and maximo is not None
        else None
    )

    return ResumoTriangulacaoCenario(
        ddm_preco_justo=ddm.preco_justo if ddm else None,
        ddm_margem_seguranca=ddm.margem_seguranca if ddm else None,
        ddm_upside_downside=ddm.upside_downside if ddm else None,
        bazin_6_preco_justo=bazin_6.preco_justo if bazin_6 else None,
        bazin_6_margem_seguranca=bazin_6.margem_seguranca if bazin_6 else None,
        bazin_6_upside_downside=bazin_6.upside_downside if bazin_6 else None,
        bazin_8_preco_justo=bazin_8.preco_justo if bazin_8 else None,
        bazin_8_margem_seguranca=bazin_8.margem_seguranca if bazin_8 else None,
        bazin_8_upside_downside=bazin_8.upside_downside if bazin_8 else None,
        graham_preco_justo=graham.preco_justo if graham else None,
        graham_margem_seguranca=graham.margem_seguranca if graham else None,
        graham_upside_downside=graham.upside_downside if graham else None,
        modelos_triangulados=modelos_triangulados,
        modelos_validos=len(precos),
        minimo=minimo,
        mediana=mediana,
        maximo=maximo,
        dispersao_pct=dispersao_pct,
        piso_bazin_8_juros_brasil=bazin_8.preco_justo if bazin_8 else None,
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


def _formatar_celula_modelo(
    preco: float | None,
    margem: float | None,
    upside: float | None,
    erro: str | None,
) -> str:
    if preco is None:
        return f"N/A ({erro})" if erro else "N/A"
    return (
        f"{_formatar_moeda(preco)} "
        f"(MS: {_formatar_pct(margem, sinal=True)} | Upside: {_formatar_pct(upside, sinal=True)})"
    )


def _logar_resumo_quantitativo(resultado: ResultadoQuantitativo) -> None:
    """Imprime resumo tabular determinístico com triangulação dos modelos antes do Analista."""
    vc = resultado.valuation_conservador
    vm = resultado.valuation_mediano
    bc = resultado.valuation_bazin_conservador
    bm = resultado.valuation_bazin_mediano
    gc = resultado.valuation_graham_conservador
    gm = resultado.valuation_graham_mediano
    mc = resultado.multiplos_conservador
    mm = resultado.multiplos_mediano
    tri = resultado.triangulacao
    p = resultado.parametros

    b6c = bc.dy_6 if bc else None
    b8c = bc.dy_8 if bc else None
    b6m = bm.dy_6 if bm else None
    b8m = bm.dy_8 if bm else None

    def _resumo_curto(
        preco: float | None, margem: float | None, upside: float | None
    ) -> str:
        if preco is None:
            return "N/A"
        return f"{_formatar_moeda(preco)} ({_formatar_pct(upside, sinal=True)} / MS {_formatar_pct(margem, sinal=True)})"

    eh_forward = resultado.projecao_lucros.trimestres_projetados > 0
    rotulo_lucro = "Lucro Anual Proj." if eh_forward else "Lucro Anual (4T)"
    rotulo_lpa_vpa = (
        "LPA (Fwd) / VPA (Atual)" if eh_forward else "LPA (4T) / VPA (Atual)"
    )
    rotulo_pl_pvp = (
        "P/L (Fwd) / P/VP (Atual)" if eh_forward else "P/L (4T) / P/VP (Atual)"
    )

    linhas = [
        f"[Calculadora] Triangulação Quantitativa ({resultado.ticker} | Setor: {resultado.setor})",
        (
            f"  Cotação Atual : {_formatar_moeda(resultado.preco_atual)} ({resultado.moeda}) | "
            f"Momento: {p.momento_operacional} | Ke={p.taxa_desconto:.1%} | g={p.crescimento_perpetuo:.1%}"
        ),
        (
            f"  Base PDF ({p.trimestre_mais_recente}): Ações={p.num_acoes_pdf:,.0f} | "
            f"PL={_formatar_moeda(p.patrimonio_liquido_pdf)} | Fonte Payout: {p.fonte_payout}"
        ),
        "  --------------------------------------------------------------------------------------------",
        f"  {'Métrica / Modelo':<26} | {'Conservador':>30} | {'Mediano':>30}",
        "  --------------------------------------------------------------------------------------------",
        f"  {rotulo_lucro:<26} | {_formatar_moeda(resultado.projecao_lucros.lucro_anual_conservador):>30} | {_formatar_moeda(resultado.projecao_lucros.lucro_anual_mediano):>30}",
        f"  {'DDM 2 Estágios (Ke 14,5%)':<26} | {_resumo_curto(vc.preco_justo if vc else None, vc.margem_seguranca if vc else None, vc.upside_downside if vc else None):>30} | {_resumo_curto(vm.preco_justo if vm else None, vm.margem_seguranca if vm else None, vm.upside_downside if vm else None):>30}",
        f"  {'Décio Bazin (DY 6% Cláss.)':<26} | {_resumo_curto(b6c.preco_justo if b6c else None, b6c.margem_seguranca if b6c else None, b6c.upside_downside if b6c else None):>30} | {_resumo_curto(b6m.preco_justo if b6m else None, b6m.margem_seguranca if b6m else None, b6m.upside_downside if b6m else None):>30}",
        f"  {'Décio Bazin (DY 8% Juros)':<26} | {_resumo_curto(b8c.preco_justo if b8c else None, b8c.margem_seguranca if b8c else None, b8c.upside_downside if b8c else None):>30} | {_resumo_curto(b8m.preco_justo if b8m else None, b8m.margem_seguranca if b8m else None, b8m.upside_downside if b8m else None):>30}",
        f"  {'Benjamin Graham (22,5)':<26} | {_resumo_curto(gc.preco_justo if gc else None, gc.margem_seguranca if gc else None, gc.upside_downside if gc else None):>30} | {_resumo_curto(gm.preco_justo if gm else None, gm.margem_seguranca if gm else None, gm.upside_downside if gm else None):>30}",
        "  --------------------------------------------------------------------------------------------",
        f"  {'Triangulação — Mediana':<26} | {_formatar_moeda(tri.conservador.mediana if tri else None):>30} | {_formatar_moeda(tri.mediano.mediana if tri else None):>30}",
        f"  {'Triangulação — Mín / Máx':<26} | {(_formatar_moeda(tri.conservador.minimo) + ' / ' + _formatar_moeda(tri.conservador.maximo)) if tri and tri.conservador.minimo is not None else 'N/A':>30} | {(_formatar_moeda(tri.mediano.minimo) + ' / ' + _formatar_moeda(tri.mediano.maximo)) if tri and tri.mediano.minimo is not None else 'N/A':>30}",
        f"  {'Dispersão (DDM/Baz6/Gra)':<26} | {_formatar_pct(tri.conservador.dispersao_pct if tri else None):>30} | {_formatar_pct(tri.mediano.dispersao_pct if tri else None):>30}",
        "  --------------------------------------------------------------------------------------------",
        f"  {rotulo_lpa_vpa:<26} | {(_formatar_moeda(mc.LPA) + ' / ' + _formatar_moeda(mc.VPA)) if mc else 'N/A':>30} | {(_formatar_moeda(mm.LPA) + ' / ' + _formatar_moeda(mm.VPA)) if mm else 'N/A':>30}",
        f"  {rotulo_pl_pvp:<26} | {(_formatar_numero(mc.P_L) + 'x / ' + _formatar_numero(mc.P_VP) + 'x') if mc else 'N/A':>30} | {(_formatar_numero(mm.P_L) + 'x / ' + _formatar_numero(mm.P_VP) + 'x') if mm else 'N/A':>30}",
        "  --------------------------------------------------------------------------------------------",
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
        for termo in (
            "servererror",
            "serviceunavailable",
            "timeout",
            "httperror",
            "ratelimit",
        )
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
    """Gera relatório Markdown completo com triangulação e em compliance quando o LLM está indisponível."""
    resultado = ResultadoQuantitativo.model_validate(resultado_quantitativo)
    vc = resultado.valuation_conservador
    vm = resultado.valuation_mediano
    bc = resultado.valuation_bazin_conservador
    bm = resultado.valuation_bazin_mediano
    gc = resultado.valuation_graham_conservador
    gm = resultado.valuation_graham_mediano
    mc = resultado.multiplos_conservador
    mm = resultado.multiplos_mediano
    tri = resultado.triangulacao
    p = resultado.parametros
    proj = resultado.projecao_lucros

    b6c = bc.dy_6 if bc else None
    b8c = bc.dy_8 if bc else None
    b6m = bm.dy_6 if bm else None
    b8m = bm.dy_8 if bm else None

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
        f"### Triangulação de Valuation ({resultado.ticker})",
        "",
        "| Metodologia | Cenário Conservador | Cenário Mediano | Referência Atual |",
        "| :--- | :---: | :---: | :---: |",
        (
            f"| **DDM de 2 Estágios ($K_e$ {_formatar_pct(p.taxa_desconto * 100)}, $g$ {_formatar_pct(p.crescimento_perpetuo * 100)})** "
            f"| {_formatar_celula_modelo(vc.preco_justo if vc else None, vc.margem_seguranca if vc else None, vc.upside_downside if vc else None, resultado.erro_valuation_conservador)} "
            f"| {_formatar_celula_modelo(vm.preco_justo if vm else None, vm.margem_seguranca if vm else None, vm.upside_downside if vm else None, resultado.erro_valuation_mediano)} "
            f"| {_formatar_moeda(resultado.preco_atual)} |"
        ),
        (
            f"| **Décio Bazin Clássico (DY {_formatar_pct(p.yield_minimo_bazin_padrao * 100)})** "
            f"| {_formatar_celula_modelo(b6c.preco_justo if b6c else None, b6c.margem_seguranca if b6c else None, b6c.upside_downside if b6c else None, bc.erro_dy_6 if bc else None)} "
            f"| {_formatar_celula_modelo(b6m.preco_justo if b6m else None, b6m.margem_seguranca if b6m else None, b6m.upside_downside if b6m else None, bm.erro_dy_6 if bm else None)} "
            f"| — |"
        ),
        (
            f"| **Décio Bazin Conservador / Juros BR (DY {_formatar_pct(p.yield_minimo_bazin_conservador * 100)})*** "
            f"| {_formatar_celula_modelo(b8c.preco_justo if b8c else None, b8c.margem_seguranca if b8c else None, b8c.upside_downside if b8c else None, bc.erro_dy_8 if bc else None)} "
            f"| {_formatar_celula_modelo(b8m.preco_justo if b8m else None, b8m.margem_seguranca if b8m else None, b8m.upside_downside if b8m else None, bm.erro_dy_8 if bm else None)} "
            f"| *Piso de renda* |"
        ),
        (
            f"| **Benjamin Graham (Constante {_formatar_numero(p.constante_graham, 1)})** "
            f"| {_formatar_celula_modelo(gc.preco_justo if gc else None, gc.margem_seguranca if gc else None, gc.upside_downside if gc else None, resultado.erro_valuation_graham_conservador)} "
            f"| {_formatar_celula_modelo(gm.preco_justo if gm else None, gm.margem_seguranca if gm else None, gm.upside_downside if gm else None, resultado.erro_valuation_graham_mediano)} "
            f"| — |"
        ),
    ]

    if tri is not None:
        linhas.extend(
            [
                f"| **Mediana da Triangulação (DDM, Bazin 6%, Graham)** | **{_formatar_moeda(tri.conservador.mediana)}** | **{_formatar_moeda(tri.mediano.mediana)}** | {_formatar_moeda(resultado.preco_atual)} |",
                f"| **Faixa (Mínimo – Máximo) / Dispersão** | {_formatar_moeda(tri.conservador.minimo)} – {_formatar_moeda(tri.conservador.maximo)} ({_formatar_pct(tri.conservador.dispersao_pct)}) | {_formatar_moeda(tri.mediano.minimo)} – {_formatar_moeda(tri.mediano.maximo)} ({_formatar_pct(tri.mediano.dispersao_pct)}) | — |",
            ]
        )

    eh_forward = proj.trimestres_projetados > 0
    rotulo_lucro = (
        "Lucro Líquido Projetado (R$)" if eh_forward else "Lucro Líquido Anual (R$)"
    )
    ref_lucro = (
        f"{_formatar_moeda(proj.soma_acumulada_ano)} *({proj.trimestres_realizados}T acum.)*"
        if eh_forward
        else f"{_formatar_moeda(proj.soma_acumulada_ano)} *(4T realizados)*"
    )
    rotulo_lpa = "LPA Projetado — Forward (R$)" if eh_forward else "LPA Anual (R$)"
    ref_lpa = (
        "— *(projeção anual)*"
        if eh_forward
        else f"{_formatar_moeda(mm.LPA if mm else None)} *(4T realizados)*"
    )
    rotulo_pl = "P/L Projetado — Forward" if eh_forward else "P/L Anual"
    ref_pl = (
        "— *(projeção anual)*"
        if eh_forward
        else f"{_formatar_numero(mm.P_L if mm else None)} *(4T realizados)*"
    )
    rotulo_roe = (
        "ROE Calculado — Forward (%)" if eh_forward else "ROE Calculado — Anual (%)"
    )

    linhas.extend(
        [
            "",
            (
                "> *Nota metodológica:* As estatísticas de triangulação utilizam estritamente **um representante por metodologia** "
                "(DDM de 2 Estágios, Bazin 6% Clássico e Benjamin Graham) para evitar peso duplo ao método Bazin. "
                "O **Bazin 8%** atua como piso conservador de renda em ambiente de Selic elevada e compensação aproximada "
                "ao fato de o DPA projetado ser bruto (antes de eventual IR de 15% sobre JCP). A constante 22,5 de Graham "
                "tende a ser mais otimista em setores intensivos em patrimônio sob juros brasileiros elevados."
            ),
            "",
            f"### Tabela Comparativa de Lucros e Múltiplos ({resultado.ticker})",
            "",
            "| Métrica | Cenário Conservador | Cenário Mediano | Referência Atual |",
            "| :--- | :---: | :---: | :---: |",
            f"| **{rotulo_lucro}** | {_formatar_moeda(proj.lucro_anual_conservador)} | {_formatar_moeda(proj.lucro_anual_mediano)} | {ref_lucro} |",
            f"| **{rotulo_lpa}** | {_formatar_moeda(mc.LPA if mc else None)} | {_formatar_moeda(mm.LPA if mm else None)} | {ref_lpa} |",
            f"| **VPA Atual (R$)** | {_formatar_moeda(mc.VPA if mc else None)} | {_formatar_moeda(mm.VPA if mm else None)} | {_formatar_moeda(mm.VPA if mm else None)} *({p.trimestre_mais_recente})* |",
            f"| **{rotulo_pl}** | {_formatar_numero(mc.P_L if mc else None)} | {_formatar_numero(mm.P_L if mm else None)} | {ref_pl} |",
            f"| **P/VP Atual** | {_formatar_numero(mc.P_VP if mc else None)} | {_formatar_numero(mm.P_VP if mm else None)} | {_formatar_numero(mm.P_VP if mm else None)} *({p.trimestre_mais_recente})* |",
            f"| **{rotulo_roe}** | {_formatar_pct(mc.ROE_calculado * 100 if mc and mc.ROE_calculado is not None else None)} | {_formatar_pct(mm.ROE_calculado * 100 if mm and mm.ROE_calculado is not None else None)} | {_formatar_pct(p.roe_pdf_recente * 100)} *({p.trimestre_mais_recente})* |",
            f"| **Crescimento Estágio 1 — DDM (%)** | {_formatar_pct(vc.crescimento_estagio_1 if vc else None)} | {_formatar_pct(vm.crescimento_estagio_1 if vm else None)} | — |",
            f"| **Payout Terminal Sustentável — DDM (%)** | {_formatar_pct(vc.payout_terminal if vc else None)} | {_formatar_pct(vm.payout_terminal if vm else None)} | — |",
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
            f"- **Taxa de Desconto ($K_e$) / Crescimento Perpétuo ($g$):** {_formatar_pct(p.taxa_desconto * 100)} / {_formatar_pct(p.crescimento_perpetuo * 100)}",
            f"- **Bazin DY Mínimo (Clássico / Juros BR):** {_formatar_pct(p.yield_minimo_bazin_padrao * 100)} / {_formatar_pct(p.yield_minimo_bazin_conservador * 100)}",
            f"- **Constante de Graham:** {_formatar_numero(p.constante_graham, 1)}",
            f"- **ROE Conservador / Mediano:** {_formatar_pct(p.roe_usado_conservador * 100)} / {_formatar_pct(p.roe_usado_mediano * 100)}",
            f"- **Payout Conservador / Mediano:** {_formatar_pct(p.payout_usado_conservador * 100 if p.payout_usado_conservador is not None else None)} / {_formatar_pct(p.payout_usado_mediano * 100 if p.payout_usado_mediano is not None else None)} *(fonte histórica: `{p.fonte_payout}`)*",
        ]
    )

    if resultado.erro_dados_mercado:
        linhas.append(f"- **Alerta de Mercado:** {resultado.erro_dados_mercado}")
    if resultado.erro_valuation_conservador:
        linhas.append(
            f"- **Alerta DDM Conservador:** {resultado.erro_valuation_conservador}"
        )
    if resultado.erro_valuation_mediano:
        linhas.append(f"- **Alerta DDM Mediano:** {resultado.erro_valuation_mediano}")
    for aviso in resultado.avisos:
        linhas.append(f"- **Aviso:** {aviso}")

    if erros_validacao:
        linhas.extend(["", "### Ressalvas da Auditoria Contábil", ""])
        for caminho, lista in erros_validacao.items():
            linhas.append(f"- **{Path(caminho).name}:** " + " ".join(lista))

    if historico_ordenado:
        linhas.extend(["", "---", "", "### Contexto Qualitativo Extraído dos PDFs", ""])
        for tri_pdf in historico_ordenado:
            ref = tri_pdf.get("trimestre_referencia", "N/A")
            resumo = tri_pdf.get("resumo_momento_operacional", "")
            visao = tri_pdf.get("visao_da_gestao", "")
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
    Executa toda a cadeia financeira determinística (zero tokens LLM) com triangulação
    DDM de 2 Estágios + Décio Bazin (6% e 8%) + Benjamin Graham, registra resumo no
    terminal, persiste JSON quantitativo e prepara a entrada do Analista.
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
    preco_float = float(preco_atual) if preco_atual is not None else None
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
    roe_usado_conservador = (
        float(roe_historico) if roe_historico is not None else roe_pdf
    )
    roe_usado_mediano = roe_pdf

    # 3. Payout médio histórico (com fallback JCP e teto sustentável 1.0)
    dados_payout_hist = buscar_payout_medio_historico_yfinance.invoke(
        {"ticker": ticker}
    )
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
        payout_pdf_seguro if payout_pdf_seguro is not None else payout_usado_conservador
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
    if preco_float is not None and preco_float > 0 and num_acoes_pdf > 0:
        multiplos_cons = MultiplosCenario.model_validate(
            calcular_multiplos_basicos.invoke(
                {
                    "preco_atual": preco_float,
                    "lucro_anual": projecao.lucro_anual_conservador,
                    "patrimonio_liquido": pl_pdf,
                    "num_acoes": num_acoes_pdf,
                }
            )
        )
        multiplos_med = MultiplosCenario.model_validate(
            calcular_multiplos_basicos.invoke(
                {
                    "preco_atual": preco_float,
                    "lucro_anual": projecao.lucro_anual_mediano,
                    "patrimonio_liquido": pl_pdf,
                    "num_acoes": num_acoes_pdf,
                }
            )
        )

    # LPA e VPA em precisão total float para evitar erro de arredondamento em cascata
    lpa_cons_exato = (
        projecao.lucro_anual_conservador / num_acoes_pdf if num_acoes_pdf > 0 else None
    )
    lpa_med_exato = (
        projecao.lucro_anual_mediano / num_acoes_pdf if num_acoes_pdf > 0 else None
    )
    vpa_exato = pl_pdf / num_acoes_pdf if num_acoes_pdf > 0 else None

    # 6. Valuation 1 — DDM de 2 Estágios (Conservador e Mediano)
    val_cons, erro_val_cons = _executar_ddm_seguro(
        lucro_base=projecao.lucro_anual_conservador,
        payout=payout_usado_conservador,
        roe=roe_usado_conservador,
        num_acoes=num_acoes_pdf,
        preco_atual=preco_float,
    )
    val_med, erro_val_med = _executar_ddm_seguro(
        lucro_base=projecao.lucro_anual_mediano,
        payout=payout_usado_mediano,
        roe=roe_usado_mediano,
        num_acoes=num_acoes_pdf,
        preco_atual=preco_float,
    )

    # 7. Valuation 2 — Décio Bazin em dois patamares (6% clássico e 8% juros BR)
    bazin_cons = _executar_bazin_cenario(
        lpa_exato=lpa_cons_exato,
        payout=payout_usado_conservador,
        preco_atual=preco_float,
    )
    bazin_med = _executar_bazin_cenario(
        lpa_exato=lpa_med_exato,
        payout=payout_usado_mediano,
        preco_atual=preco_float,
    )

    # 8. Valuation 3 — Benjamin Graham (22,5)
    graham_cons, erro_graham_cons = _executar_graham_seguro(
        lpa_exato=lpa_cons_exato,
        vpa_exato=vpa_exato,
        preco_atual=preco_float,
    )
    graham_med, erro_graham_med = _executar_graham_seguro(
        lpa_exato=lpa_med_exato,
        vpa_exato=vpa_exato,
        preco_atual=preco_float,
    )

    # 9. Triangulação consolidada (1 representante por metodologia: DDM, Bazin 6%, Graham)
    triangulacao = TriangulacaoValuation(
        conservador=_montar_resumo_triangulacao_cenario(
            ddm=val_cons,
            bazin=bazin_cons,
            graham=graham_cons,
        ),
        mediano=_montar_resumo_triangulacao_cenario(
            ddm=val_med,
            bazin=bazin_med,
            graham=graham_med,
        ),
    )

    resultado = ResultadoQuantitativo(
        ticker=ticker,
        setor=setor,
        preco_atual=preco_float,
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
            yield_minimo_bazin_padrao=YIELD_MINIMO_BAZIN_PADRAO,
            yield_minimo_bazin_conservador=YIELD_MINIMO_BAZIN_CONSERVADOR,
            constante_graham=CONSTANTE_GRAHAM_PADRAO,
        ),
        projecao_lucros=projecao,
        multiplos_conservador=multiplos_cons,
        multiplos_mediano=multiplos_med,
        valuation_conservador=val_cons,
        valuation_mediano=val_med,
        erro_valuation_conservador=erro_val_cons,
        erro_valuation_mediano=erro_val_med,
        valuation_bazin_conservador=bazin_cons,
        valuation_bazin_mediano=bazin_med,
        valuation_graham_conservador=graham_cons,
        valuation_graham_mediano=graham_med,
        erro_valuation_graham_conservador=erro_graham_cons,
        erro_valuation_graham_mediano=erro_graham_med,
        triangulacao=triangulacao,
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
            "da triangulação (DDM, Bazin 6%, Bazin 8% e Graham) e integrando as perspectivas qualitativas da gestão."
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
