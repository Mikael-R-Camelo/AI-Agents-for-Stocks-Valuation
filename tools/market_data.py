import logging
import math
import unicodedata
from typing import Any

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from tools.utils import obter_ticker_yfinance

logger = logging.getLogger(__name__)

_ALIASES_LUCRO_LIQUIDO = [
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income Continuous Operations",
    "Net Income Including Noncontrolling Interests",
]

_ALIASES_PATRIMONIO_LIQUIDO = [
    "Stockholders Equity",
    "Common Stock Equity",
    "Total Equity Gross Minority Interest",
]

_ALIASES_DIVIDENDOS_TOTAIS = [
    "Cash Dividends Paid",
    "Dividends Paid",
    "Payment Of Dividends",
]

_LIMITE_ANOS_HISTORICOS = 4
_LIMITE_DISTORCAO_PAYOUT = 1.50


def _extrair_float_valido(valor: Any) -> float | None:
    """Converte um valor para float finito ou retorna None se inválido/NaN."""
    if valor is None or isinstance(valor, bool) or pd.isna(valor):
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) else None


def _normalizar_texto_chave(texto: str) -> str:
    """Remove acentos, espaços extras e converte para minúsculas."""
    sem_acento = "".join(
        c
        for c in unicodedata.normalize("NFKD", texto.strip().lower())
        if not unicodedata.combining(c)
    )
    return sem_acento


def _aplicar_estresse_conservador(valor: float, percentual: float) -> float:
    """Aplica estresse conservador: reduz lucros positivos e agrava prejuízos negativos."""
    if valor >= 0:
        return valor * (1.0 - percentual)
    return valor * (1.0 + percentual)


def _buscar_linha_contabil(
    df: pd.DataFrame | None, aliases: list[str]
) -> pd.Series | None:
    """Localiza a primeira linha contábil disponível entre os aliases informados."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    for nome in aliases:
        if nome in df.index:
            linha = df.loc[nome]
            if isinstance(linha, pd.DataFrame):
                return linha.iloc[0]
            return linha
    return None


def _buscar_linha_dividendos(fc: pd.DataFrame | None) -> pd.Series | None:
    """Localiza linha agregada de dividendos pagos ou soma classes ON/PN na DFC."""
    linha_total = _buscar_linha_contabil(fc, _ALIASES_DIVIDENDOS_TOTAIS)
    if linha_total is not None:
        return linha_total

    linha_on = _buscar_linha_contabil(fc, ["Common Stock Dividend Paid"])
    linha_pn = _buscar_linha_contabil(fc, ["Preferred Stock Dividend Paid"])

    if linha_on is not None and linha_pn is not None:
        return linha_on.fillna(0.0) + linha_pn.fillna(0.0)
    return linha_on if linha_on is not None else linha_pn


def _ordenar_datas_desc(indices: list[Any]) -> list[Any]:
    """Ordena colunas de datas contábeis do mais recente para o mais antigo."""
    try:
        return sorted(indices, reverse=True)
    except TypeError:
        return list(indices)


@tool
def buscar_dados_mercado_yfinance(ticker: str) -> dict[str, Any]:
    """
    Busca EXCLUSIVAMENTE a cotação atual de mercado (preço e moeda) na B3 via Yahoo Finance.
    Não retorna número de ações nem VPA para evitar distorções de classes ON/PN — utilize
    sempre `numero_total_acoes` e `patrimonio_liquido` extraídos do PDF oficial.
    """
    try:
        acao = obter_ticker_yfinance(ticker)

        info = acao.info or {}
        if not isinstance(info, dict):
            info = {}

        preco_atual: float | None = None
        for campo in ("currentPrice", "regularMarketPrice", "previousClose"):
            candidato = _extrair_float_valido(info.get(campo))
            if candidato is not None and candidato > 0:
                preco_atual = candidato
                break

        if preco_atual is None:
            fast_info = getattr(acao, "fast_info", None)
            if fast_info is not None:
                for campo in ("lastPrice", "previousClose"):
                    try:
                        candidato = _extrair_float_valido(fast_info.get(campo))
                    except (AttributeError, KeyError, TypeError, ValueError):
                        candidato = _extrair_float_valido(
                            getattr(fast_info, campo, None)
                        )
                    if candidato is not None and candidato > 0:
                        preco_atual = candidato
                        break

        if preco_atual is None:
            return {
                "ticker": ticker.strip().upper(),
                "erro": "Não foi possível obter cotação válida no Yahoo Finance.",
            }

        return {
            "ticker": ticker.strip().upper(),
            "preco_atual": round(preco_atual, 2),
            "moeda": str(info.get("currency") or "BRL"),
        }

    except Exception as e:
        logger.exception("Falha ao buscar cotação no Yahoo Finance para %s", ticker)
        return {
            "ticker": ticker.strip().upper(),
            "erro": f"Falha ao buscar cotação no Yahoo Finance: {e!s}",
        }


@tool
def projetar_lucro_anual_cenarios(
    lucros_trimestrais_realizados: list[float],
    momento_operacional: str = "estavel",
) -> dict[str, Any]:
    """
    Calcula o lucro anual projetado em dois cenários (Conservador e Mediano)
    a partir de uma lista com os lucros de 1, 2, 3 ou 4 trimestres do ano.

    Args:
        lucros_trimestrais_realizados: Lista de lucros em R$ na ordem cronológica (ex: [1T, 2T, 3T]).
        momento_operacional: 'deteriorando', 'estavel' ou 'recuperacao' (baseado na leitura do release).
    """
    n = len(lucros_trimestrais_realizados)
    if n == 0 or n > 4:
        raise ValueError("A lista deve conter entre 1 e 4 trimestres.")

    lucros_validos: list[float] = []
    for valor in lucros_trimestrais_realizados:
        numero = _extrair_float_valido(valor)
        if numero is None:
            raise ValueError(
                "Todos os valores em lucros_trimestrais_realizados devem ser números finitos."
            )
        lucros_validos.append(numero)

    momento_norm = _normalizar_texto_chave(momento_operacional)
    if momento_norm not in {"recuperacao", "estavel", "deteriorando"}:
        raise ValueError(
            f"momento_operacional inválido: '{momento_operacional}'. "
            "Use 'recuperacao', 'estavel' ou 'deteriorando'."
        )

    soma_realizada = sum(lucros_validos)
    faltantes = 4 - n

    if faltantes == 0:
        proj_conservadora = _aplicar_estresse_conservador(soma_realizada, 0.05)
        proj_mediana = soma_realizada
    else:
        media_trimestral = soma_realizada / n
        pior_trimestre = min(lucros_validos)
        ultimo_trimestre = lucros_validos[-1]

        if momento_norm == "recuperacao":
            # Conservador = trimestres faltantes voltam à média do ano
            # Mediano = mantém o run-rate do último trimestre
            proj_conservadora = soma_realizada + (media_trimestral * faltantes)
            proj_mediana = soma_realizada + (ultimo_trimestre * faltantes)
        elif momento_norm == "deteriorando":
            # Conservador = pior trimestre agravado em 10%
            # Mediano = repete o pior trimestre
            pior_estressado = _aplicar_estresse_conservador(pior_trimestre, 0.10)
            proj_conservadora = soma_realizada + (pior_estressado * faltantes)
            proj_mediana = soma_realizada + (pior_trimestre * faltantes)
        else:
            # Estável:
            # Conservador = repete o pior trimestre do ano
            # Mediano = repete a média dos trimestres realizados
            proj_conservadora = soma_realizada + (pior_trimestre * faltantes)
            proj_mediana = soma_realizada + (media_trimestral * faltantes)

        # Invariante: cenário conservador nunca pode ser mais otimista que o mediano
        proj_conservadora = min(proj_conservadora, proj_mediana)

    return {
        "trimestres_realizados": n,
        "trimestres_projetados": faltantes,
        "momento_operacional": momento_norm,
        "soma_acumulada_ano": round(soma_realizada, 2),
        "lucro_anual_conservador": round(proj_conservadora, 2),
        "lucro_anual_mediano": round(proj_mediana, 2),
    }


@tool
def calcular_multiplos_basicos(
    preco_atual: float,
    lucro_anual: float,
    patrimonio_liquido: float,
    num_acoes: float,
) -> dict[str, float | None]:
    """
    Calcula LPA, VPA, P/L, P/VP e ROE de forma determinística.
    Utilize exclusivamente `patrimonio_liquido` e `num_acoes` extraídos do PDF oficial.
    Retorna None para múltiplos não aplicáveis (ex: P/L em caso de prejuízo).
    """
    for nome, valor in {
        "preco_atual": preco_atual,
        "lucro_anual": lucro_anual,
        "patrimonio_liquido": patrimonio_liquido,
        "num_acoes": num_acoes,
    }.items():
        if _extrair_float_valido(valor) is None:
            raise ValueError(f"Parâmetro '{nome}' deve ser um número finito válido.")

    if num_acoes <= 0:
        raise ValueError(
            "num_acoes deve ser estritamente positivo (> 0) e extraído do PDF oficial."
        )
    if preco_atual <= 0:
        raise ValueError("preco_atual deve ser estritamente positivo (> 0).")

    lpa = lucro_anual / num_acoes
    vpa = patrimonio_liquido / num_acoes
    pl = round(preco_atual / lpa, 2) if lpa > 0 else None
    pvp = round(preco_atual / vpa, 2) if vpa > 0 else None
    roe = (
        round(lucro_anual / patrimonio_liquido, 4) if patrimonio_liquido > 0 else None
    )

    return {
        "LPA": round(lpa, 2),
        "VPA": round(vpa, 2),
        "P_L": pl,
        "P_VP": pvp,
        "ROE_calculado": roe,
    }


@tool
def buscar_medias_historicas_yfinance(ticker: str) -> dict[str, Any]:
    """
    Busca o ROE médio (Retorno sobre Patrimônio Líquido)
    dos últimos 4 anos utilizando os dados contábeis anuais do Yahoo Finance.
    """
    try:
        acao = obter_ticker_yfinance(ticker)

        dre = acao.financials
        bp = acao.balance_sheet

        lucros = _buscar_linha_contabil(dre, _ALIASES_LUCRO_LIQUIDO)
        patrimonios = _buscar_linha_contabil(bp, _ALIASES_PATRIMONIO_LIQUIDO)

        if lucros is None or patrimonios is None:
            return {
                "ticker": ticker.strip().upper(),
                "erro": "Dados contábeis (DRE/Balanço) indisponíveis no Yahoo Finance para calcular o ROE.",
            }

        anos_comuns = [ano for ano in lucros.index if ano in patrimonios.index]
        anos_ordenados = _ordenar_datas_desc(anos_comuns)[:_LIMITE_ANOS_HISTORICOS]

        roe_anos: list[float] = []
        for ano in anos_ordenados:
            lucro = _extrair_float_valido(lucros[ano])
            patrimonio = _extrair_float_valido(patrimonios[ano])

            if lucro is not None and patrimonio is not None and patrimonio > 0:
                roe_anos.append(lucro / patrimonio)

        if not roe_anos:
            return {
                "ticker": ticker.strip().upper(),
                "erro": "Dados contábeis insuficientes no Yahoo Finance para calcular o ROE.",
            }

        roe_medio = sum(roe_anos) / len(roe_anos)

        return {
            "ticker": ticker.strip().upper(),
            "anos_analisados": len(roe_anos),
            "roe_medio_historico": round(roe_medio, 4),
            "roes_anuais_identificados": [round(r, 4) for r in roe_anos],
        }

    except Exception as e:
        logger.exception("Falha ao buscar histórico de ROE para %s", ticker)
        return {
            "ticker": ticker.strip().upper(),
            "erro": f"Falha ao buscar histórico de ROE: {e!s}",
        }


def _obter_payout_fallback_info(acao: yf.Ticker) -> float | None:
    """Obtém `payoutRatio` via ticker.info quando a DFC vier zerada/ausente (ex: JCP em bancos)."""
    try:
        info = acao.info or {}
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.debug("Fallback info['payoutRatio'] indisponível: %s", exc)
        return None

    if not isinstance(info, dict):
        return None

    payout_info = _extrair_float_valido(info.get("payoutRatio"))
    if payout_info is None or payout_info <= 0 or payout_info > _LIMITE_DISTORCAO_PAYOUT:
        return None
    return payout_info


@tool
def buscar_payout_medio_historico_yfinance(ticker: str) -> dict[str, Any]:
    """
    Busca o Payout médio (percentual do lucro distribuído aos acionistas)
    dos últimos 4 anos utilizando os dados de fluxo de caixa do Yahoo Finance.

    Quando a linha de dividendos na DFC vier ausente ou zerada em todos os anos
    (comum em bancos brasileiros que distribuem via JCP), aplica fallback para
    `info["payoutRatio"]`.

    Retorna tanto `payout_medio_observado` (média real dos anos válidos até 150%)
    quanto `payout_sustentavel` (limitado a 1.0), que deve ser utilizado no valuation DDM.
    """
    try:
        ticker_limpo = ticker.strip().upper()
        acao = obter_ticker_yfinance(ticker)

        dre = acao.financials
        fc = acao.cash_flow

        lucros = _buscar_linha_contabil(dre, _ALIASES_LUCRO_LIQUIDO)
        linha_dividendos = _buscar_linha_dividendos(fc)

        payout_anos_observados: list[float] = []
        anos_ignorados_distorcao = 0
        anos_com_prejuizo = 0

        if lucros is not None and linha_dividendos is not None:
            anos_comuns = [
                ano for ano in lucros.index if ano in linha_dividendos.index
            ]
            anos_ordenados = _ordenar_datas_desc(anos_comuns)[:_LIMITE_ANOS_HISTORICOS]

            for ano in anos_ordenados:
                lucro = _extrair_float_valido(lucros[ano])
                dividendos = _extrair_float_valido(linha_dividendos[ano])

                if lucro is None or dividendos is None:
                    continue

                if lucro <= 0:
                    anos_com_prejuizo += 1
                    continue

                # Dividendos na DFC aparecem como saída de caixa (negativo)
                payout_bruto = abs(dividendos) / lucro

                # Ignora distorções não-recorrentes onde o payout superou 150% do lucro
                if payout_bruto > _LIMITE_DISTORCAO_PAYOUT:
                    anos_ignorados_distorcao += 1
                    continue

                payout_anos_observados.append(payout_bruto)

        dfc_zerada_ou_ausente = linha_dividendos is None or (
            bool(payout_anos_observados)
            and all(p == 0.0 for p in payout_anos_observados)
        )

        if dfc_zerada_ou_ausente:
            payout_info = _obter_payout_fallback_info(acao)
            if payout_info is not None:
                payout_sustentavel = min(payout_info, 1.0)
                return {
                    "ticker": ticker_limpo,
                    "fonte_payout": "yfinance_info_payoutRatio_fallback",
                    "anos_analisados": len(payout_anos_observados),
                    "anos_ignorados_distorcao": anos_ignorados_distorcao,
                    "anos_com_prejuizo": anos_com_prejuizo,
                    "payout_medio_observado": round(payout_info, 4),
                    "payout_sustentavel": round(payout_sustentavel, 4),
                    "payout_limitado_para_valuation": payout_info > 1.0,
                    "payouts_anuais_observados": [
                        round(p, 4) for p in payout_anos_observados
                    ],
                    "aviso": (
                        "Linha de dividendos na DFC do Yahoo Finance retornou zerada/ausente "
                        "(comum em empresas B3 com JCP); utilizado fallback info['payoutRatio']."
                    ),
                }

        if lucros is None or linha_dividendos is None:
            return {
                "ticker": ticker_limpo,
                "erro": "Dados de DRE ou Fluxo de Caixa insuficientes no Yahoo Finance para calcular o Payout.",
            }

        if not payout_anos_observados:
            return {
                "ticker": ticker_limpo,
                "erro": (
                    "Dados válidos de fluxo de caixa insuficientes no Yahoo Finance para calcular o Payout "
                    f"(anos ignorados por distorção >150%: {anos_ignorados_distorcao}, "
                    f"anos com prejuízo: {anos_com_prejuizo})."
                ),
            }

        payout_medio_observado = sum(payout_anos_observados) / len(
            payout_anos_observados
        )
        payout_sustentavel = min(payout_medio_observado, 1.0)

        return {
            "ticker": ticker_limpo,
            "fonte_payout": "dfc_yfinance",
            "anos_analisados": len(payout_anos_observados),
            "anos_ignorados_distorcao": anos_ignorados_distorcao,
            "anos_com_prejuizo": anos_com_prejuizo,
            "payout_medio_observado": round(payout_medio_observado, 4),
            "payout_sustentavel": round(payout_sustentavel, 4),
            "payout_limitado_para_valuation": payout_medio_observado > 1.0,
            "payouts_anuais_observados": [
                round(p, 4) for p in payout_anos_observados
            ],
        }

    except Exception as e:
        logger.exception("Falha ao buscar histórico de Payout para %s", ticker)
        return {
            "ticker": ticker.strip().upper(),
            "erro": f"Falha ao buscar histórico de Payout: {e!s}",
        }
