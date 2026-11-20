from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agents.prompts import montar_prompt_extrator
from tools.utils import obter_modelo_gemini


class DadosTrimestraisPDF(BaseModel):
    trimestre_referencia: str = Field(
        description="Trimestre e ano do relatório no formato '1T26', '2T26', '3T26' ou '4T26'."
    )
    lucro_liquido_trimestral: float = Field(
        description="Lucro Líquido Recorrente ISOLADO deste trimestre em Reais (R$). "
        "ATENÇÃO: Não pegue o lucro acumulado do ano (9M/6M). Converta milhões/bilhões para o número absoluto completo (ex: 7109000000.0)."
    )
    patrimonio_liquido: float = Field(
        description="Patrimônio Líquido total do período em Reais (R$) absolutos."
    )
    roe_reportado: float = Field(
        description="ROE (Retorno sobre Patrimônio Líquido) anualizado reportado no trimestre em decimal (ex: 0.158 para 15,8%)."
    )
    payout_estimado: float = Field(
        description="Percentual de distribuição de dividendos/JCP sobre o lucro em decimal (ex: 0.50). Se não encontrar explícito, preencha com 0.50."
    )
    resumo_momento_operacional: str = Field(
        description="Resumo objetivo de até 4 frases sobre a tendência operacional no trimestre (ex: evolução da margem financeira, PDD/inadimplência e eficiência). O que aconteceu de fato nos números (crescimento da margem, queda na inadimplência, etc)."
    )
    visao_da_gestao: str = Field(
        description="Resumo de até 3 frases sobre as falas da diretoria, o 'guidance' (projeções) e o tom (otimista, cauteloso ou pessimista) para os próximos trimestres."
    )
    numero_total_acoes: float = Field(
        description="Número TOTAL de ações da companhia (somando ON e PN, se houver). Converta bilhões para o número absoluto (ex: 11178825623.0)."
    )
    caixa_inicial: float | None = Field(
        default=None,
        description=(
            "Saldo inicial de 'Caixa e equivalentes de caixa' na DFC em Reais (R$) absolutos. "
            "Preencha com null se o documento não apresentar DFC (comum em press releases de bancos)."
        ),
    )
    caixa_final: float | None = Field(
        default=None,
        description=(
            "Saldo final de 'Caixa e equivalentes de caixa' na DFC em Reais (R$) absolutos. "
            "Preencha com null se o documento não apresentar DFC."
        ),
    )
    variacao_liquida_caixa: float | None = Field(
        default=None,
        description=(
            "Aumento ou redução líquida de caixa e equivalentes durante o período na DFC em Reais (R$) absolutos. "
            "Pode ser negativo. Preencha com null se o documento não apresentar DFC."
        ),
    )
    efeito_cambial_caixa: float | None = Field(
        default=None,
        description=(
            "Efeito da variação cambial sobre caixa e equivalentes de caixa na DFC em Reais (R$) absolutos, "
            "quando divulgado em linha separada. Preencha com null se não aplicável ou ausente."
        ),
    )


def extrair_dados_do_pdf(
    caminho_pdf: str,
    feedback_correcao: str | None = None,
    llm: ChatGoogleGenerativeAI | None = None,
) -> dict[str, Any]:
    """Extrai dados trimestrais estruturados de um PDF, aceitando feedback corretivo em retries."""
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()
    texto_completo = "\n".join(doc.page_content for doc in documentos)

    modelo_llm = llm or ChatGoogleGenerativeAI(
        model=obter_modelo_gemini(),
        temperature=0,
        max_retries=3,
    )
    llm_estruturado = modelo_llm.with_structured_output(DadosTrimestraisPDF)

    prompt = montar_prompt_extrator(
        texto_documento=texto_completo,
        feedback_correcao=feedback_correcao,
    )

    resultado = llm_estruturado.invoke(prompt)
    if resultado is None:
        raise ValueError(
            f"O modelo não retornou dados estruturados válidos para o PDF '{caminho_pdf}'."
        )

    return resultado.model_dump()
