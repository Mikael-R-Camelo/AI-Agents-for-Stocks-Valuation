from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


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


def extrair_dados_do_pdf(caminho_pdf: str) -> dict:
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()
    texto_completo = "\n".join([doc.page_content for doc in documentos])

    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, max_retries=3)
    llm_estruturado = llm.with_structured_output(DadosTrimestraisPDF)

    prompt = (
        "Você é um extrator de dados financeiros contábeis de alta precisão.\n"
        "Leia a Demonstração Financeira / Release de Resultados abaixo e extraia os dados EXCLUSIVOS deste trimestre isolado.\n"
        "CUIDADO: Tabelas contábeis costumam mostrar colunas de 'Acumulado do Ano' (ex: 9M26) ao lado do trimestre isolado (ex: 3T26). "
        "Extraia ESTRITAMENTE o lucro do trimestre isolado de 3 meses e converta a escala (Mil/Milhões/Bilhões) para unidades de Reais.\n\n"
        f"DOCUMENTO:\n{texto_completo[:150000]}"
    )

    resultado = llm_estruturado.invoke(prompt)
    return resultado.model_dump()
