from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocoloSetor:
    """Regras que orientam a análise de um tipo de ativo."""

    nome: str
    instrucoes: str


# TODO: Por enquanto os prompts estão muito genéricos somente para fins de testes simples.
#       Necessário criar prompts mais específicos para cada setor, com instruções detalhadas
#       sobre como analisar os dados financeiros e de mercado.
PROTOCOLOS_SETOR: dict[str, ProtocoloSetor] = {
    "banco": ProtocoloSetor(
        nome="bancos",
        instrucoes=(
            "Para bancos, considere a qualidade do lucro, ROE, payout, "
            "índice de Basileia e crescimento sustentável. Não trate receita "
            "e fluxo de caixa de um banco como se fossem de uma empresa não "
            "financeira."
        ),
    ),
    "eletrica": ProtocoloSetor(
        nome="elétricas",
        instrucoes=(
            "Para elétricas, considere previsibilidade do fluxo de caixa, "
            "dívida, contratos, capacidade de distribuição de dividendos e "
            "riscos regulatórios."
        ),
    ),
    "holding": ProtocoloSetor(
        nome="holdings",
        instrucoes=(
            "Para holdings, considere o valor das participações, o desconto "
            "ou prêmio de holding, a dívida líquida e compare o valor da "
            "holding com a compra direta das participações quando houver "
            "dados suficientes."
        ),
    ),
}


def obter_protocolo(setor: str) -> ProtocoloSetor:
    """Retorna o protocolo do setor ou falha explicitamente para setor inválido."""
    chave = setor.strip().lower()
    try:
        return PROTOCOLOS_SETOR[chave]
    except KeyError as exc:
        setores_disponiveis = ", ".join(sorted(PROTOCOLOS_SETOR))
        raise ValueError(
            f"Setor '{setor}' não possui protocolo. "
            f"Use um destes: {setores_disponiveis}."
        ) from exc
