import re
from typing import Any

from pydantic import BaseModel, Field

from config import MAX_TENTATIVAS_COMPLIANCE

__all__ = [
    "MAX_TENTATIVAS_COMPLIANCE",
    "ResultadoCompliance",
    "anexar_alerta_compliance",
    "auditar_compliance_relatorio",
    "extrair_texto_mensagem",
    "montar_feedback_compliance",
]

_PADROES_RECOMENDACAO_EXPLICITA: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "rotulo_recomendacao",
        re.compile(
            r"\b(?:recomenda[çc][aã]o|classifica[çc][aã]o|veredito|parecer|call|"
            r"a[çc][aã]o\s+recomendada)\s*(?:final|atual|do\s+analista)?\s*"
            r"[:\-–—]\s*(?:compra\s+forte|venda\s+forte|compra|comprar|venda|"
            r"vender|manter|manuten[çc][aã]o|neutr[ao]|buy|strong\s+buy|sell|"
            r"strong\s+sell|hold|outperform|underperform)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rating_explicito",
        re.compile(
            r"\brating\s*[:\-–—]\s*(?:strong\s+buy|buy|strong\s+sell|sell|hold|"
            r"outperform|underperform|overweight|underweight|compra|venda|manter)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "verbo_recomendacao_direta",
        re.compile(
            r"\b(?:recomendamos|sugerimos|indicamos|aconselhamos)\s+"
            r"(?:a\s+|o\s+|fortemente\s+|cautelosamente\s+)?"
            r"(?:compra|comprar|venda|vender|manter|manuten[çc][aã]o|"
            r"a\s+aquisi[çc][aã]o\s+d[aeo]s?\s+a[çc][õo]es|"
            r"desfazer[- ]se\s+d[aeo]s?\s+a[çc][õo]es)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "voz_passiva_recomendacao",
        re.compile(
            r"\b(?:recomenda[- ]se|sugere[- ]se|indica[- ]se)\s+"
            r"(?:a\s+)?(?:compra|comprar|venda|vender|manter|manuten[çc][aã]o)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sinal_ou_recomendacao_de_operacao",
        re.compile(
            r"\b(?:recomenda[çc][aã]o|indica[çc][aã]o|sinal|gatilho|chamada)\s+"
            r"(?:clara\s+|forte\s+|expl[íi]cita\s+)?de\s+"
            r"(?:compra|venda|manuten[çc][aã]o)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "diretriz_operacional_direta",
        re.compile(
            r"\b(?:vale\s+a\s+pena|hora\s+de|momento\s+(?:ideal\s+|certo\s+)?(?:para|de))\s+"
            r"(?:comprar|vender)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "imperativo_investidor",
        re.compile(
            r"\b(?:deve(?:m)?|deveria(?:m)?)\s+(?:comprar|vender|manter)\s+"
            r"(?:o\s+papel|a\s+a[çc][aã]o|as\s+a[çc][õo]es|o\s+ativo|"
            r"este\s+ativo|esta\s+a[çc][aã]o)\b",
            re.IGNORECASE,
        ),
    ),
)

# Quebra de cláusula (pontuação forte ou conjunção adversativa) para isolar escopo de negação
_REGEX_QUEBRA_CLAUSULA = re.compile(
    r"[.;!\n]|,?\s*\b(?:mas|por[eé]m|contudo|entretanto|todavia|no\s+entanto)\b",
    re.IGNORECASE,
)

# Contexto negativo / disclaimer jurídico na mesma cláusula anterior ao trecho detectado
_REGEX_NEGACAO_DISCLAIMER = re.compile(
    r"\b(?:n[aã]o\s+(?:constitui|representa|configura|se\s+trata\s+de|caracteriza|"
    r"implica|[eé]|deve\s+ser\s+interpretad[oa]\s+como|h[aá]|emite|emitimos|"
    r"faz|fazemos|cont[eé]m|traz|recomendamos|sugerimos|indicamos|aconselhamos)|"
    r"sem\s+(?:car[aá]ter\s+de\s+|qualquer\s+)?|proibid[oa]\s+de|vedad[oa]|"
    r"aus[eê]ncia\s+de|isento\s+de|n[aã]o\s+s[aã]o|not\s+a|does\s+not\s+constitute)\b|"
    r"\bn[aã]o\s*$",
    re.IGNORECASE,
)

_REGEX_MARKDOWN_INLINE = re.compile(r"[*_`]+")


class ResultadoCompliance(BaseModel):
    aprovado: bool = Field(
        description="Indica se o texto está livre de recomendações explícitas de investimento."
    )
    violacoes: list[str] = Field(
        default_factory=list,
        description="Lista de trechos e regras de compliance violadas.",
    )


def extrair_texto_mensagem(conteudo_bruto: Any) -> str:
    """Converte o conteúdo de uma mensagem (string ou lista de blocos Gemini) em texto puro."""
    if isinstance(conteudo_bruto, list):
        return "\n".join(
            bloco.get("text", "")
            for bloco in conteudo_bruto
            if isinstance(bloco, dict)
        )
    return str(conteudo_bruto)


def _esta_em_contexto_negativo(texto_limpo: str, inicio_match: int) -> bool:
    """Verifica se o trecho detectado é negado na mesma cláusula (ex: disclaimer legal)."""
    janela_inicio = max(0, inicio_match - 120)
    prefixo = texto_limpo[janela_inicio:inicio_match]

    partes = _REGEX_QUEBRA_CLAUSULA.split(prefixo)
    clausula_atual = partes[-1] if partes else prefixo
    return bool(_REGEX_NEGACAO_DISCLAIMER.search(clausula_atual))


def auditar_compliance_relatorio(texto: str) -> ResultadoCompliance:
    """
    Audita determinísticamente um relatório em busca de recomendações explícitas de
    compra, venda ou manutenção, ignorando uso financeiro legítimo e disclaimers negativos.
    """
    if not texto or not texto.strip():
        return ResultadoCompliance(aprovado=True, violacoes=[])

    texto_normalizado = _REGEX_MARKDOWN_INLINE.sub("", texto)
    violacoes: list[str] = []

    for nome_regra, padrao in _PADROES_RECOMENDACAO_EXPLICITA:
        for match in padrao.finditer(texto_normalizado):
            if _esta_em_contexto_negativo(texto_normalizado, match.start()):
                continue
            trecho = match.group(0).strip()
            registro = f"{nome_regra}: '{trecho}'"
            if registro not in violacoes:
                violacoes.append(registro)

    return ResultadoCompliance(
        aprovado=not violacoes,
        violacoes=violacoes,
    )


def montar_feedback_compliance(violacoes: list[str]) -> str:
    """Gera instrução corretiva para reescrita limpa do relatório pelo Analista."""
    lista = "\n".join(f"- {v}" for v in violacoes)
    return (
        "ERRO DE COMPLIANCE DETECTADO NO RELATÓRIO GERADO.\n"
        "Foram identificadas recomendações explícitas proibidas nos seguintes trechos:\n"
        f"{lista}\n\n"
        "Reescreva o relatório completo mantendo EXATAMENTE todos os números, tabelas, "
        "múltiplos e premissas retornados pelas ferramentas (sem chamar ferramentas novamente), "
        "removendo qualquer recomendação explícita de compra, venda ou manutenção. "
        "Apresente apenas a análise objetiva de Preço Justo, Margem de Segurança e Upside."
    )


def anexar_alerta_compliance(texto: str, violacoes: list[str]) -> str:
    """Adiciona cabeçalho explícito de alerta caso o limite de reescritas seja atingido."""
    lista = "; ".join(violacoes)
    banner = (
        "> **ALERTA DE COMPLIANCE AUTOMATIZADO:** Após o limite de reescritas automáticas, "
        "a auditoria determinística ainda detectou possíveis recomendações explícitas neste texto "
        f"({lista}). Revise manualmente antes de distribuir ou utilizar este relatório."
    )
    return f"{banner}\n\n{texto}"
