import logging
import re
from functools import lru_cache

import yfinance as yf

from config import DEFAULT_GEMINI_MODEL, obter_modelo_gemini

logger = logging.getLogger(__name__)

MAPA_SETORES_YFINANCE: dict[str, str] = {
    "Financial Services": "banco",
    "Utilities": "energia",
    "Energy": "commodities",
    "Basic Materials": "commodities",
    "Real Estate": "imobiliario",
    "Consumer Defensive": "varejo",
    "Consumer Cyclical": "varejo",
    "Healthcare": "saude",
    "Industrials": "industria",
}

_REGEX_TRIMESTRE = re.compile(r"([1-4])\s*[TtQq]\s*(\d{2,4})")

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "MAPA_SETORES_YFINANCE",
    "chave_ordenacao_trimestre",
    "identificar_setor_pelo_ticker",
    "limpar_cache_yfinance",
    "normalizar_ticker_b3",
    "obter_modelo_gemini",
    "obter_ticker_yfinance",
]


def normalizar_ticker_b3(ticker: str) -> str:
    """Normaliza ticker da B3 em maiúsculas garantindo sufixo '.SA' sem duplicação."""
    limpo = ticker.strip().upper()
    if not limpo:
        raise ValueError("Ticker não pode ser vazio.")
    return limpo if limpo.endswith(".SA") else f"{limpo}.SA"


@lru_cache(maxsize=64)
def _obter_ticker_yfinance_por_simbolo(ticker_yf: str) -> yf.Ticker:
    return yf.Ticker(ticker_yf)


def obter_ticker_yfinance(ticker: str) -> yf.Ticker:
    """Retorna instância cacheada em memória de `yf.Ticker` para o ticker B3 normalizado."""
    return _obter_ticker_yfinance_por_simbolo(normalizar_ticker_b3(ticker))


def limpar_cache_yfinance() -> None:
    """Limpa o cache em memória de instâncias `yf.Ticker`."""
    _obter_ticker_yfinance_por_simbolo.cache_clear()


def chave_ordenacao_trimestre(trimestre_ref: str) -> tuple[int, int]:
    """Converte código de trimestre (ex: '4T25', '1T26', '3T2026') em tupla cronológica (ano, trimestre)."""
    if not isinstance(trimestre_ref, str):
        return (9999, 9)

    match = _REGEX_TRIMESTRE.search(trimestre_ref.strip())
    if not match:
        return (9999, 9)

    trimestre = int(match.group(1))
    ano_str = match.group(2)
    ano = 2000 + int(ano_str) if len(ano_str) == 2 else int(ano_str)
    return (ano, trimestre)


def identificar_setor_pelo_ticker(ticker: str) -> str:
    """Busca o setor no Yahoo Finance e traduz para o protocolo interno do agente."""
    try:
        info = obter_ticker_yfinance(ticker).info or {}
        if not isinstance(info, dict):
            info = {}

        setor_yf = str(info.get("sector") or "").strip()
        industria_yf = str(info.get("industry") or "").strip()

        setor_traduzido = MAPA_SETORES_YFINANCE.get(setor_yf, "geral")

        logger.info(
            "[Sistema] Setor identificado pelo Yahoo: %s (%s) -> Protocolo: '%s'",
            setor_yf or "N/A",
            industria_yf or "N/A",
            setor_traduzido,
        )
        return setor_traduzido

    except Exception:
        logger.exception(
            "[Aviso] Falha ao identificar setor para %s. Usando protocolo 'geral'.",
            ticker,
        )
        return "geral"
