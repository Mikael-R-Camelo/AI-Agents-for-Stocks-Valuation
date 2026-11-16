import yfinance as yf
from langchain_core.tools import tool


@tool
def buscar_dados_mercado_yfinance(ticker: str) -> dict:
    """Busca a cotação atual e o número total de ações na B3 via Yahoo Finance."""
    # Na B3, o Yahoo Finance exige o sufixo .SA (ex: BBDC3.SA)
    ticker_yf = (
        f"{ticker.upper().strip()}.SA"
        if not ticker.upper().endswith(".SA")
        else ticker.upper()
    )

    acao = yf.Ticker(ticker_yf)
    info = acao.info

    preco_atual = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    # Tenta pegar o total de ações; se falhar em PN/ON, calcula via Market Cap / Preço
    num_acoes = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    market_cap = info.get("marketCap", 0.0)

    if not num_acoes and preco_atual > 0 and market_cap > 0:
        num_acoes = market_cap / preco_atual

    return {
        "ticker": ticker.upper(),
        "preco_atual": round(float(preco_atual), 2),
        "num_acoes_total": float(num_acoes) if num_acoes else 0.0,
        "valor_patrimonial_por_acao_vpa": info.get("bookValue", 0.0),
    }


@tool
def projetar_lucro_anual_cenarios(
    lucros_trimestrais_realizados: list[float],
    momento_operacional: str = "estavel",
) -> dict:
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

    soma_realizada = sum(lucros_trimestrais_realizados)
    faltantes = 4 - n

    if faltantes == 0:
        return {
            "trimestres_contabilizados": 4,
            "lucro_anual_conservador": soma_realizada
            * 0.95,  # Margem de segurança de 5%
            "lucro_anual_mediano": soma_realizada,
        }

    media_trimestral = soma_realizada / n
    pior_trimestre = min(lucros_trimestrais_realizados)
    ultimo_trimestre = lucros_trimestrais_realizados[-1]

    # Lógica baseada no momento operacional identificado pelo Agente
    if momento_operacional == "recuperacao":
        # Em turnaround (ex: Bradesco melhorando ROE a cada tri):
        # Conservador = assume que os próximos voltam para a média do ano
        # Mediano = assume que mantém o ritmo do último trimestre (Run-rate atual)
        proj_conservadora = soma_realizada + (media_trimestral * faltantes)
        proj_mediana = soma_realizada + (ultimo_trimestre * faltantes)
    elif momento_operacional == "deteriorando":
        # Conservador = pior trimestre com corte adicional de 10%
        # Mediano = repete o pior/último trimestre
        proj_conservadora = soma_realizada + ((pior_trimestre * 0.90) * faltantes)
        proj_mediana = soma_realizada + (pior_trimestre * faltantes)
    else:
        # Estável:
        # Conservador = repete o pior trimestre do ano
        # Mediano = repete a média dos trimestres realizados
        proj_conservadora = soma_realizada + (pior_trimestre * faltantes)
        proj_mediana = soma_realizada + (media_trimestral * faltantes)

    return {
        "trimestres_realizados": n,
        "trimestres_projetados": faltantes,
        "soma_acumulada_ano": soma_realizada,
        "lucro_anual_conservador": round(proj_conservadora, 2),
        "lucro_anual_mediano": round(proj_mediana, 2),
    }


@tool
def calcular_multiplos_basicos(
    preco_atual: float,
    lucro_anual: float,
    patrimonio_liquido: float,
    num_acoes: float,
) -> dict:
    """Calcula LPA, VPA, P/L, P/VP e ROE de forma determinística."""
    lpa = lucro_anual / num_acoes
    vpa = patrimonio_liquido / num_acoes
    pl = preco_atual / lpa if lpa > 0 else 0.0
    pvp = preco_atual / vpa if vpa > 0 else 0.0
    roe = lucro_anual / patrimonio_liquido if patrimonio_liquido > 0 else 0.0

    return {
        "LPA": round(lpa, 2),
        "VPA": round(vpa, 2),
        "P_L": round(pl, 2),
        "P_VP": round(pvp, 2),
        "ROE_calculado": round(roe, 4),
    }
