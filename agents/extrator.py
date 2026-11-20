import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, ValidationError

from agents.prompts import montar_prompt_extrator
from config import (
    LIMITE_CARACTERES_DOCUMENTO,
    VERSAO_SCHEMA_EXTRATOR,
    obter_dir_cache_extrator,
    obter_llm,
    obter_modelo_gemini,
)

logger = logging.getLogger(__name__)

_PALAVRAS_CHAVE_PESO: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        5,
        (
            "lucro liquido",
            "lucro recorrente",
            "resultado recorrente",
            "patrimonio liquido",
            "retorno sobre o patrimonio",
            "roae",
            "roe",
            "capital social",
            "quantidade de acoes",
            "total de acoes",
            "acoes emitidas",
            "acoes em circulacao",
        ),
    ),
    (
        4,
        (
            "demonstracao do resultado",
            "balanco patrimonial",
            "fluxo de caixa",
            "caixa e equivalentes",
            "variacao cambial",
            "dividendos",
            "juros sobre o capital",
            "jcp",
            "payout",
        ),
    ),
    (
        2,
        (
            "mensagem da administracao",
            "perspectivas",
            "guidance",
            "destaques",
            "margem financeira",
            "inadimplencia",
            "eficiencia",
            "basileia",
            "ebitda",
            "divida liquida",
        ),
    ),
)

_REGEX_NUMEROS_TABELA = re.compile(r"\d+[.,]\d+")


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


class EntradaCacheExtracao(BaseModel):
    hash_pdf: str
    versao_schema: str
    modelo: str
    dados: DadosTrimestraisPDF


def _normalizar_para_busca(texto: str) -> str:
    sem_acento = "".join(
        c
        for c in unicodedata.normalize("NFKD", texto.lower())
        if not unicodedata.combining(c)
    )
    return sem_acento


def _pontuar_pagina_financeira(indice: int, texto_pagina: str) -> int:
    texto_norm = _normalizar_para_busca(texto_pagina)
    pontos = 3 if indice < 2 else 0

    for peso, termos in _PALAVRAS_CHAVE_PESO:
        for termo in termos:
            if termo in texto_norm:
                pontos += peso

    if len(_REGEX_NUMEROS_TABELA.findall(texto_pagina)) >= 5:
        pontos += 1

    return pontos


def selecionar_texto_relevante_pdf(
    paginas: list[str],
    limite_caracteres: int = LIMITE_CARACTERES_DOCUMENTO,
) -> str:
    """
    Retorna o texto integral se couber no limite; caso contrário, rankeia páginas
    por relevância contábil (DRE, PL, ROE, ações, DFC, guidance) preservando a ordem original.
    """
    texto_integral = "\n".join(paginas)
    if len(texto_integral) <= limite_caracteres:
        return texto_integral

    paginas_pontuadas = [
        (_pontuar_pagina_financeira(idx, texto), idx, texto)
        for idx, texto in enumerate(paginas)
    ]
    # Maior pontuação primeiro; em empate, páginas mais iniciais têm precedência
    paginas_pontuadas.sort(key=lambda item: (-item[0], item[1]))

    selecionadas: list[tuple[int, str]] = []
    tamanho_acumulado = 0

    for _pontos, idx, texto in paginas_pontuadas:
        bloco = f"[Página {idx + 1}]\n{texto}"
        custo = len(bloco) + (2 if selecionadas else 0)
        if tamanho_acumulado + custo <= limite_caracteres:
            selecionadas.append((idx, bloco))
            tamanho_acumulado += custo

    if not selecionadas and paginas_pontuadas:
        _, idx_topo, texto_topo = paginas_pontuadas[0]
        bloco_topo = f"[Página {idx_topo + 1}]\n{texto_topo}"
        selecionadas.append((idx_topo, bloco_topo[:limite_caracteres]))

    # Reordena na sequência original do documento para coerência de leitura pelo LLM
    selecionadas.sort(key=lambda item: item[0])

    logger.info(
        "[Extrator] PDF extenso (%d chars > %d): selecionadas %d/%d páginas mais relevantes.",
        len(texto_integral),
        limite_caracteres,
        len(selecionadas),
        len(paginas),
    )
    return "\n\n".join(bloco for _, bloco in selecionadas)


def calcular_hash_arquivo(caminho_pdf: str | Path) -> str:
    """Calcula hash SHA-256 do conteúdo binário de um arquivo PDF."""
    digest = hashlib.sha256()
    with Path(caminho_pdf).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(65_536), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _caminho_arquivo_cache(
    caminho_pdf: str | Path,
    modelo: str | None = None,
    dir_cache: Path | None = None,
) -> tuple[Path, str, str]:
    hash_pdf = calcular_hash_arquivo(caminho_pdf)
    modelo_efetivo = modelo or obter_modelo_gemini()
    chave_composta = hashlib.sha256(
        f"{hash_pdf}:{VERSAO_SCHEMA_EXTRATOR}:{modelo_efetivo}".encode()
    ).hexdigest()
    diretorio = dir_cache or obter_dir_cache_extrator()
    return diretorio / f"{chave_composta}.json", hash_pdf, modelo_efetivo


def carregar_cache_extracao(
    caminho_pdf: str | Path,
    modelo: str | None = None,
    dir_cache: Path | None = None,
) -> dict[str, Any] | None:
    """Carrega extração validada previamente salva em cache SHA-256, ou retorna None."""
    caminho_pdf_path = Path(caminho_pdf)
    if not caminho_pdf_path.is_file():
        return None

    try:
        caminho_cache, hash_pdf, modelo_efetivo = _caminho_arquivo_cache(
            caminho_pdf_path, modelo=modelo, dir_cache=dir_cache
        )
        if not caminho_cache.is_file():
            return None

        bruto = json.loads(caminho_cache.read_text(encoding="utf-8"))
        entrada = EntradaCacheExtracao.model_validate(bruto)

        if (
            entrada.hash_pdf != hash_pdf
            or entrada.versao_schema != VERSAO_SCHEMA_EXTRATOR
            or entrada.modelo != modelo_efetivo
        ):
            return None

        return entrada.dados.model_dump()
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "[Cache] Cache inválido ou ilegível para '%s' (%s); ignorando.",
            caminho_pdf_path.name,
            exc,
        )
        return None


def salvar_cache_extracao(
    caminho_pdf: str | Path,
    dados: dict[str, Any],
    modelo: str | None = None,
    dir_cache: Path | None = None,
) -> Path | None:
    """Persiste no cache local JSON uma extração previamente APROVADA pelo validador."""
    caminho_pdf_path = Path(caminho_pdf)
    if not caminho_pdf_path.is_file():
        return None

    try:
        dados_validados = DadosTrimestraisPDF.model_validate(dados)
        caminho_cache, hash_pdf, modelo_efetivo = _caminho_arquivo_cache(
            caminho_pdf_path, modelo=modelo, dir_cache=dir_cache
        )
        caminho_cache.parent.mkdir(parents=True, exist_ok=True)

        entrada = EntradaCacheExtracao(
            hash_pdf=hash_pdf,
            versao_schema=VERSAO_SCHEMA_EXTRATOR,
            modelo=modelo_efetivo,
            dados=dados_validados,
        )

        caminho_tmp = caminho_cache.with_suffix(".tmp")
        caminho_tmp.write_text(
            entrada.model_dump_json(indent=2),
            encoding="utf-8",
        )
        caminho_tmp.replace(caminho_cache)
        logger.info("[Cache] Extração validada salva em cache para %s.", caminho_pdf_path.name)
        return caminho_cache
    except (OSError, ValidationError) as exc:
        logger.warning(
            "[Cache] Não foi possível salvar cache para '%s': %s",
            caminho_pdf_path.name,
            exc,
        )
        return None


def extrair_dados_do_pdf(
    caminho_pdf: str,
    feedback_correcao: str | None = None,
    llm: ChatGoogleGenerativeAI | None = None,
) -> dict[str, Any]:
    """Extrai dados trimestrais estruturados de um PDF, aceitando feedback corretivo em retries."""
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()
    paginas = [doc.page_content for doc in documentos]
    texto_relevante = selecionar_texto_relevante_pdf(paginas)

    modelo_llm = llm or obter_llm()
    llm_estruturado = modelo_llm.with_structured_output(DadosTrimestraisPDF)

    prompt = montar_prompt_extrator(
        texto_documento=texto_relevante,
        feedback_correcao=feedback_correcao,
    )

    resultado = llm_estruturado.invoke(prompt)
    if resultado is None:
        raise ValueError(
            f"O modelo não retornou dados estruturados válidos para o PDF '{caminho_pdf}'."
        )

    return resultado.model_dump()
