import yfinance as yf


def identificar_setor_pelo_ticker(ticker: str) -> str:
    """Busca o setor no Yahoo Finance e traduz para o protocolo do nosso agente."""
    ticker_yf = (
        f"{ticker.upper().strip()}.SA"
        if not ticker.upper().endswith(".SA")
        else ticker.upper()
    )

    try:
        info = yf.Ticker(ticker_yf).info
        setor_yf = info.get("sector", "")
        industria_yf = info.get("industry", "")

        # Mapeamento: De (Inglês Yahoo) -> Para (Seu Protocolo Interno)
        mapa_setores = {
            "Financial Services": "banco",
            "Utilities": "energia",  # Transmissão/Geração
            "Energy": "commodities",  # Petróleo/Gás
            "Basic Materials": "commodities",  # Mineração/Siderurgia
            "Real Estate": "imobiliario",
            "Consumer Defensive": "varejo",
            "Consumer Cyclical": "varejo",
            "Healthcare": "saude",
            "Industrials": "industria",
        }

        # Tenta achar o setor no nosso mapa. Se não achar, usa um protocolo padrão "geral"
        setor_traduzido = mapa_setores.get(setor_yf, "geral")

        print(
            f"[Sistema] Setor identificado pelo Yahoo: {setor_yf} ({industria_yf}) -> Protocolo: '{setor_traduzido}'"
        )
        return setor_traduzido

    except Exception as e:
        print(
            f"[Aviso] Falha ao identificar setor para {ticker}: {e!s}. Usando 'geral'."
        )
        return "geral"
