import math

from config import CRESCIMENTO_PERPETUO_PADRAO, TAXA_DESCONTO_PADRAO
from langchain_core.tools import tool


@tool
def fluxo_de_caixa_futuro_buffet(
    lucro_base: float,
    payout: float,
    roe: float,
    num_acoes: float,
    preco_atual: float,
    taxa_desconto: float = TAXA_DESCONTO_PADRAO,
    crescimento_perpetuo: float = CRESCIMENTO_PERPETUO_PADRAO,
    anos_projecao: int = 4,
) -> dict[str, float]:
    """
    Calcula o Valuation (Preço Justo) via DDM de 2 Estágios consistente:
    - Estágio 1 (Anos 1 a N): projeta o lucro anual crescendo à taxa sustentável
      g = (1 - payout) * roe e desconta os dividendos efetivos (lucro_projetado * payout).
    - Estágio 2 (Perpetuidade / Valor Terminal): assume convergência do retorno marginal
      ao custo de capital próprio (ROE_terminal = Ke), aplicando o payout sustentável de
      maturidade payout_terminal = 1 - (crescimento_perpetuo / taxa_desconto) sobre o lucro
      do ano N+1 no modelo de Gordon Growth.

    Atenção às unidades obrigatórias:
    - lucro_base: Lucro anual base em Reais (R$) absolutos (> 0).
    - payout: Decimal no intervalo (0, 1] (ex: 0.50 para 50%).
    - roe: Decimal (ex: 0.16 para 16%).
    - num_acoes: Número TOTAL de ações (ON + PN) extraído do PDF (> 0).
    - preco_atual: Cotação atual da ação em R$ (> 0).
    - taxa_desconto: Custo de capital próprio Ke em decimal (padrão: 0.145 para 14,5%).
    - crescimento_perpetuo: Crescimento perpétuo g em decimal (padrão: 0.05 para 5%).
    - anos_projecao: Horizonte explícito de projeção em anos (>= 1, padrão: 4).
    """
    parametros_numericos = {
        "lucro_base": lucro_base,
        "payout": payout,
        "roe": roe,
        "taxa_desconto": taxa_desconto,
        "crescimento_perpetuo": crescimento_perpetuo,
        "num_acoes": num_acoes,
        "preco_atual": preco_atual,
    }
    for nome, valor in parametros_numericos.items():
        if (
            isinstance(valor, bool)
            or not isinstance(valor, (int, float))
            or not math.isfinite(valor)
        ):
            raise ValueError(f"Parâmetro '{nome}' deve ser um número finito válido.")

    if anos_projecao < 1:
        raise ValueError("anos_projecao deve ser maior ou igual a 1.")
    if lucro_base <= 0:
        raise ValueError(
            "lucro_base deve ser estritamente positivo (> 0) para o modelo DDM."
        )
    if num_acoes <= 0:
        raise ValueError(
            "num_acoes deve ser estritamente positivo (> 0) e extraído do PDF oficial."
        )
    if preco_atual <= 0:
        raise ValueError("preco_atual deve ser estritamente positivo (> 0).")
    if not (0 < payout <= 1):
        raise ValueError(
            "payout deve ser um decimal no intervalo (0, 1] (ex: 0.50 para 50%). "
            "O modelo DDM requer distribuição positiva de dividendos."
        )
    if abs(roe) > 2.0:
        raise ValueError(
            "roe deve estar em formato decimal (ex: 0.16 para 16%), não em pontos percentuais."
        )
    if not (0 < taxa_desconto < 1):
        raise ValueError(
            "taxa_desconto deve ser um decimal no intervalo (0, 1) (ex: 0.145 para 14,5%)."
        )
    if not (0 <= crescimento_perpetuo < 1):
        raise ValueError(
            "crescimento_perpetuo deve ser um decimal no intervalo [0, 1) (ex: 0.05 para 5%)."
        )
    if taxa_desconto <= crescimento_perpetuo:
        raise ValueError(
            "taxa_desconto deve ser estritamente maior que crescimento_perpetuo "
            "para convergência do valor terminal."
        )

    # 1. Taxa de Crescimento Sustentável no Estágio 1: g = (1 - payout) * ROE
    crescimento_sustentavel = (1.0 - payout) * roe
    if crescimento_sustentavel <= -1.0:
        raise ValueError(
            "Combinação de ROE e payout resulta em crescimento sustentável <= -100%."
        )

    lucros_projetados: list[float] = []
    vpl_fluxos: list[float] = []
    lucro_anterior = float(lucro_base)

    # 2. Estágio 1: Projeção dos Lucros e Desconto dos Dividendos Efetivos (Anos 1 a N)
    for ano in range(1, anos_projecao + 1):
        lucro_projetado = lucro_anterior * (1.0 + crescimento_sustentavel)
        lucros_projetados.append(lucro_projetado)

        dividendo_projetado = lucro_projetado * payout
        vpl = dividendo_projetado / ((1.0 + taxa_desconto) ** ano)
        vpl_fluxos.append(vpl)

        lucro_anterior = lucro_projetado

    # 3. Estágio 2: Valor Terminal com convergência do retorno marginal ao Ke
    # Em equilíbrio competitivo (ROE_terminal = Ke), payout_terminal = 1 - (g_perpetuo / Ke)
    payout_terminal = 1.0 - (crescimento_perpetuo / taxa_desconto)
    lucro_terminal = lucros_projetados[-1] * (1.0 + crescimento_perpetuo)
    dividendo_terminal = lucro_terminal * payout_terminal

    valor_terminal = dividendo_terminal / (taxa_desconto - crescimento_perpetuo)
    vpl_terminal = valor_terminal / ((1.0 + taxa_desconto) ** anos_projecao)

    # 4. Resultados Finais
    market_cap_projetado = sum(vpl_fluxos) + vpl_terminal
    preco_justo = market_cap_projetado / num_acoes

    if preco_justo <= 0:
        raise ValueError(
            "Preço justo calculado não é positivo; revise os parâmetros de entrada."
        )

    margem_seguranca = (preco_justo - preco_atual) / preco_justo
    upside_tradicional = (preco_justo / preco_atual) - 1.0

    return {
        "Crescimento Sustentável (%)": round(crescimento_sustentavel * 100, 2),
        "Payout Terminal Sustentável (%)": round(payout_terminal * 100, 2),
        "Taxa de Desconto Ke (%)": round(taxa_desconto * 100, 2),
        "Crescimento Perpétuo (%)": round(crescimento_perpetuo * 100, 2),
        "Market Cap Justo (R$)": round(market_cap_projetado, 2),
        "Preço Teto / Justo (R$)": round(preco_justo, 2),
        "Margem de Segurança (%)": round(margem_seguranca * 100, 2),
        "Upside Real (%)": round(upside_tradicional * 100, 2),
    }
