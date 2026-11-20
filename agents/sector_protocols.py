from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocoloSetor:
    """Regras que orientam a análise de um tipo de ativo."""

    nome: str
    instrucoes: str


_PROTOCOLO_ENERGIA = ProtocoloSetor(
    nome="energia e utilidades públicas",
    instrucoes=(
        "Para elétricas e utilidades públicas, considere previsibilidade de fluxo de caixa, "
        "alavancagem (Dívida Líquida/EBITDA), prazos de concessão/contratos regulados, "
        "capacidade de distribuição de dividendos e riscos regulatórios."
    ),
)

PROTOCOLOS_SETOR: dict[str, ProtocoloSetor] = {
    "banco": ProtocoloSetor(
        nome="bancos",
        instrucoes=(
            "Para bancos, considere a qualidade do lucro recorrente, ROE, payout, "
            "margem financeira, PDD/inadimplência, eficiência operacional, índice de Basileia "
            "(quando mencionado no release) e crescimento sustentável. Não trate receita e "
            "fluxo de caixa de um banco como se fossem de uma empresa não financeira."
        ),
    ),
    "energia": _PROTOCOLO_ENERGIA,
    "eletrica": _PROTOCOLO_ENERGIA,
    "holding": ProtocoloSetor(
        nome="holdings",
        instrucoes=(
            "Para holdings, considere o valor das participações, o desconto ou prêmio "
            "de holding, a dívida líquida na controladora e a capacidade de repasse de "
            "dividendos recebidos das investidas."
        ),
    ),
    "commodities": ProtocoloSetor(
        nome="commodities (energia, mineração e materiais básicos)",
        instrucoes=(
            "Para empresas de commodities, avalie o estágio do ciclo de preços internacionais, "
            "exposição cambial, custo caixa de produção, disciplina de capex, alavancagem "
            "financeira e sustentabilidade dos dividendos em cenários menos favoráveis."
        ),
    ),
    "imobiliario": ProtocoloSetor(
        nome="setor imobiliário e construção civil",
        instrucoes=(
            "Para empresas imobiliárias, avalie ciclo de juros, lançamentos/VGV, velocidade "
            "de vendas (VSO), margem bruta a apropriar, geração de caixa operacional e "
            "estrutura de capital."
        ),
    ),
    "varejo": ProtocoloSetor(
        nome="varejo e consumo",
        instrucoes=(
            "Para empresas de varejo e consumo, considere crescimento de vendas mesmas lojas "
            "(SSS), evolução de margem bruta e EBITDA, giro de estoques/capital de giro, "
            "endividamento financeiro e sensibilidade à taxa de juros e renda disponível."
        ),
    ),
    "saude": ProtocoloSetor(
        nome="saúde",
        instrucoes=(
            "Para empresas de saúde, avalie sinistralidade, ticket médio, ocupação/volume, "
            "integração de aquisições, ciclo de recebimento e alavancagem financeira."
        ),
    ),
    "industria": ProtocoloSetor(
        nome="indústria e bens de capital",
        instrucoes=(
            "Para indústrias, avalie carteira de pedidos, ROIC, margens operacionais, "
            "repasses de custos de insumos, exposição cambial e eficiência de capital."
        ),
    ),
    "geral": ProtocoloSetor(
        nome="empresas de capital aberto (protocolo geral)",
        instrucoes=(
            "Considere qualidade e recorrência do lucro líquido, ROE sustentável, "
            "disciplina na alocação de capital, consistência do payout histórico, "
            "saúde financeira e riscos operacionais e macroeconômicos relatados pela gestão."
        ),
    ),
}


def obter_protocolo(setor: str) -> ProtocoloSetor:
    """Retorna o protocolo do setor ou falha explicitamente para setor inválido."""
    chave = (setor or "").strip().lower()
    try:
        return PROTOCOLOS_SETOR[chave]
    except KeyError as exc:
        setores_disponiveis = ", ".join(sorted(PROTOCOLOS_SETOR))
        raise ValueError(
            f"Setor '{setor}' não possui protocolo. "
            f"Use um destes: {setores_disponiveis}."
        ) from exc
