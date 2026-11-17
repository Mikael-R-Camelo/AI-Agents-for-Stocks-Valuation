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


import pandas as pd
from langchain_core.tools import tool


@tool
def buscar_medias_historicas_yfinance(ticker: str) -> dict:
    """
    Busca o ROE médio (Retorno sobre Patrimônio Líquido)
    dos últimos 4 anos utilizando os dados contábeis anuais do Yahoo Finance.
    """
    ticker_yf = (
        f"{ticker.upper().strip()}.SA"
        if not ticker.upper().endswith(".SA")
        else ticker.upper()
    )
    acao = yf.Ticker(ticker_yf)

    try:
        # Puxa a DRE (Lucro) e o Balanço Patrimonial (Patrimônio) anuais
        dre = acao.financials
        bp = acao.balance_sheet

        roe_anos = []

        # Verifica se as linhas contábeis padrão existem no ticker
        if "Net Income" in dre.index and "Stockholders Equity" in bp.index:
            lucros = dre.loc["Net Income"]
            patrimonios = bp.loc["Stockholders Equity"]

            # Itera sobre as colunas (que representam os anos)
            for ano in lucros.index:
                if ano in patrimonios.index:
                    lucro = lucros[ano]
                    patrimonio = patrimonios[ano]

                    # Filtra valores vazios (NaN) e garante que o patrimônio é positivo
                    if pd.notna(lucro) and pd.notna(patrimonio) and patrimonio > 0:
                        roe_ano = lucro / patrimonio
                        roe_anos.append(roe_ano)

        if not roe_anos:
            return {
                "erro": "Dados contábeis insuficientes no Yahoo Finance para calcular o ROE."
            }

        roe_medio = sum(roe_anos) / len(roe_anos)

        return {
            "ticker": ticker.upper(),
            "anos_analisados": len(roe_anos),
            "roe_medio_historico": round(roe_medio, 4),  # Ex: 0.1650 (16,5%)
            "roes_anuais_identificados": [
                round(r, 4) for r in roe_anos
            ],  # Mostra a variação pro agente
        }

    except Exception as e:
        return {"erro": f"Falha ao buscar histórico: {e!s}"}


@tool
def buscar_payout_medio_historico_yfinance(ticker: str) -> dict:
    """
    Busca o Payout médio (percentual do lucro distribuído aos acionistas)
    dos últimos 4 anos utilizando os dados de fluxo de caixa do Yahoo Finance.
    """
    ticker_yf = (
        f"{ticker.upper().strip()}.SA"
        if not ticker.upper().endswith(".SA")
        else ticker.upper()
    )
    acao = yf.Ticker(ticker_yf)

    try:
        # Puxa a DRE (para o Lucro) e o Fluxo de Caixa (para os Dividendos Pagos)
        dre = acao.financials
        fc = acao.cash_flow

        payout_anos = []

        if "Net Income" in dre.index:
            lucros = dre.loc["Net Income"]

            # Procura a linha de dividendos pagos (o YF usa nomenclaturas diferentes por empresa)
            linha_dividendos = None
            for nome in [
                "Cash Dividends Paid",
                "Dividends Paid",
                "Common Stock Dividend Paid",
            ]:
                if nome in fc.index:
                    linha_dividendos = fc.loc[nome]
                    break

            if linha_dividendos is not None:
                for ano in lucros.index:
                    if ano in linha_dividendos.index:
                        lucro = lucros[ano]
                        dividendos = linha_dividendos[ano]

                        # Calcula apenas se houve lucro (payout de prejuízo não é sustentável)
                        if pd.notna(lucro) and pd.notna(dividendos) and lucro > 0:
                            # Dividendos no FC aparecem como saída de caixa (negativo)
                            payout_ano = abs(dividendos) / lucro

                            # Trava de segurança: ignorar distorções onde a empresa pagou > 150% do lucro
                            # Limitamos o payout máximo sustentável em 100% (1.0) para a média
                            payout_ano = min(float(payout_ano), 1.0)
                            payout_anos.append(payout_ano)

        if not payout_anos:
            return {
                "erro": "Dados de fluxo de caixa insuficientes no Yahoo Finance para calcular o Payout."
            }

        payout_medio = sum(payout_anos) / len(payout_anos)

        return {
            "ticker": ticker.upper(),
            "anos_analisados": len(payout_anos),
            "payout_medio_historico": round(payout_medio, 4),  # Ex: 0.6000 (60%)
            "payouts_anuais_identificados": [round(p, 4) for p in payout_anos],
        }

    except Exception as e:
        return {"erro": f"Falha ao buscar histórico de Payout: {e!s}"}
