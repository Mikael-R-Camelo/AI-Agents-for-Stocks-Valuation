from langchain_core.tools import tool


@tool
def fluxo_de_caixa_futuro_buffet(
    lucro_base: float,
    payout: float,
    roe: float,
    taxa_desconto: float,
    crescimento_perpetuo: float,
    num_acoes: float,
    preco_atual: float,
    crescimento_ano_1: float = None,  # type: ignore
    anos_projecao: int = 4,
):
    """
    Calcula o Valuation (Preço Justo) utilizando o método de fluxo de caixa futuro inspirado na abordagem de Warren Buffett.
    """

    # 1. Taxa de Crescimento Sustentável (g)
    # A parte do lucro que não é distribuída como dividendo (1 - payout) multiplicada pelo retorno do patrimônio (ROE)
    crescimento_sustentavel = (1 - payout) * roe

    fluxos_projetados = []
    vpl_fluxos = []
    lucro_anterior = lucro_base

    # 2. Projeção dos Fluxos (Anos 1 a N)
    for ano in range(1, anos_projecao + 1):
        # A planilha permite usar uma taxa de expectativa diferente (ex: 30%) no primeiro ano projetado
        if ano == 1 and crescimento_ano_1 is not None:
            taxa_crescimento = crescimento_ano_1
        else:
            taxa_crescimento = crescimento_sustentavel

        lucro_projetado = lucro_anterior * (1 + taxa_crescimento)
        fluxos_projetados.append(lucro_projetado)

        # Desconto a Valor Presente (VPL = FV / (1 + r)^n)
        vpl = lucro_projetado / ((1 + taxa_desconto) ** ano)
        vpl_fluxos.append(vpl)

        lucro_anterior = lucro_projetado

    # 3. Cálculo do Valor Terminal (Perpetuidade no final da projeção)
    ultimo_lucro = fluxos_projetados[-1]

    # Fórmula de perpetuidade traduzida exatamente da célula F31 da planilha
    denominador_perpetuidade = ((1 + taxa_desconto) / (1 + crescimento_perpetuo)) - 1
    valor_terminal = ultimo_lucro / denominador_perpetuidade

    # O VPL da perpetuidade é descontado usando o tempo do último ano de projeção
    vpl_terminal = valor_terminal / ((1 + taxa_desconto) ** anos_projecao)

    # 4. Resultados Finais
    market_cap_projetado = sum(vpl_fluxos) + vpl_terminal
    preco_justo = market_cap_projetado / num_acoes

    # A planilha usa (Preço Justo - Preço Atual) / Preço Justo como Margem de Segurança
    margem_seguranca = (preco_justo - preco_atual) / preco_justo

    # Bônus: Cálculo tradicional de Upside (potencial de ganho)
    upside_tradicional = (preco_justo / preco_atual) - 1

    return {
        "Crescimento Sustentável (Anos 2+)": round(crescimento_sustentavel * 100, 2),
        "Market Cap Justo (R$)": round(market_cap_projetado, 2),
        "Preço Teto / Justo (R$)": round(preco_justo, 2),
        "Margem de Segurança Planilha (%)": round(margem_seguranca * 100, 2),
        "Upside Real (%)": round(upside_tradicional * 100, 2),
    }


# # --- EXEMPLO DE USO PARA BBDC3---
# resultado = fluxo_de_caixa_futuro_buffet(
#     lucro_base=17_542_153_000,  # Lucro de 2024 da planilha
#     payout=0.50,  # Payout de 50%
#     roe=0.1165,  # ROE de 11,65%
#     taxa_desconto=0.15,  # Taxa de 15% (Selic + Risco)
#     crescimento_perpetuo=0.03,  # Crescimento perpétuo de 3%
#     num_acoes=10_592_012_000,  # Total de ações
#     preco_atual=16.72,  # Preço lido na planilha
#     crescimento_ano_1=0.30,  # A planilha jogou 30% de aumento no 1º ano projetado
#     anos_projecao=4,
# )

# for chave, valor in resultado.items():
#     print(f"{chave}: {valor}")
