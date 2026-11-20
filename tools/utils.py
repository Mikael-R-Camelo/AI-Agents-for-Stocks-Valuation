import os
import re

import yfinance as yf

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

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


def obter_modelo_gemini() -> str:
    """Retorna o modelo Gemini configurado via GEMINI_MODEL (padrão funcional mantido)."""
    modelo = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    return modelo or DEFAULT_GEMINI_MODEL


def normalizar_ticker_b3(ticker: str) -> str:
    """Normaliza ticker da B3 em maiúsculas garantindo sufixo '.SA' sem duplicação."""
    limpo = ticker.strip().upper()
    if not limpo:
        raise ValueError("Ticker não pode ser vazio.")
    return limpo if limpo.endswith(".SA") else f"{limpo}.SA"


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
        ticker_yf = normalizar_ticker_b3(ticker)
        info = yf.Ticker(ticker_yf).info or {}
        if not isinstance(info, dict):
            info = {}

        setor_yf = str(info.get("sector") or "").strip()
        industria_yf = str(info.get("industry") or "").strip()

        setor_traduzido = MAPA_SETORES_YFINANCE.get(setor_yf, "geral")

        print(
            f"[Sistema] Setor identificado pelo Yahoo: {setor_yf or 'N/A'} "
            f"({industria_yf or 'N/A'}) -> Protocolo: '{setor_traduzido}'"
        )
        return setor_traduzido

    except Exception as e:
        print(
            f"[Aviso] Falha ao identificar setor para {ticker}: {e!s}. Usando 'geral'."
        )
        return "geral"
