LIMITE_CARACTERES_DOCUMENTO = 150_000


def montar_prompt_extrator(
    texto_documento: str,
    feedback_correcao: str | None = None,
    limite_caracteres: int = LIMITE_CARACTERES_DOCUMENTO,
) -> str:
    """Monta o prompt de extração estruturada de um PDF trimestral, com suporte a feedback corretivo."""
    bloco_correcao = ""
    if feedback_correcao:
        bloco_correcao = (
            "\n\nATENÇÃO — CORREÇÃO OBRIGATÓRIA DA TENTATIVA ANTERIOR:\n"
            "A auditoria matemática identificou inconsistências na sua extração anterior deste mesmo PDF:\n"
            f"{feedback_correcao}\n"
            "Releia atentamente as tabelas, verifique a escala (mil/milhões/bilhões), "
            "garanta formato decimal para percentuais (ex: 0.161 para 16,1%) e não confunda "
            "valores trimestrais com acumulados do ano (6M/9M/12M).\n"
        )

    return (
        "Você é um extrator de dados financeiros contábeis de alta precisão.\n"
        "Leia a Demonstração Financeira / Release de Resultados abaixo e extraia os dados EXCLUSIVOS deste trimestre isolado.\n"
        "CUIDADO: Tabelas contábeis costumam mostrar colunas de 'Acumulado do Ano' (ex: 9M26) ao lado do trimestre isolado (ex: 3T26). "
        "Extraia ESTRITAMENTE o lucro do trimestre isolado de 3 meses e converta a escala (Mil/Milhões/Bilhões) para unidades de Reais.\n"
        "Para campos de DFC (caixa_inicial, caixa_final, variacao_liquida_caixa, efeito_cambial_caixa), "
        "preencha com null caso o documento não contenha a Demonstração de Fluxo de Caixa."
        f"{bloco_correcao}\n\n"
        f"DOCUMENTO:\n{texto_documento[:limite_caracteres]}"
    )
