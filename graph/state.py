import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from config import (
    CONSTANTE_GRAHAM_PADRAO,
    YIELD_MINIMO_BAZIN_CONSERVADOR,
    YIELD_MINIMO_BAZIN_PADRAO,
)

MomentoOperacional = Literal["recuperacao", "estavel", "deteriorando"]


class ParametrosAvaliacao(BaseModel):
    momento_operacional: MomentoOperacional
    roe_historico: float | None = None
    roe_pdf_recente: float
    roe_usado_conservador: float
    roe_usado_mediano: float
    payout_medio_observado: float | None = None
    payout_sustentavel: float | None = None
    payout_pdf_recente: float
    payout_usado_conservador: float | None = None
    payout_usado_mediano: float | None = None
    fonte_payout: str
    num_acoes_pdf: float
    patrimonio_liquido_pdf: float
    trimestre_mais_recente: str
    taxa_desconto: float
    crescimento_perpetuo: float
    yield_minimo_bazin_padrao: float = YIELD_MINIMO_BAZIN_PADRAO
    yield_minimo_bazin_conservador: float = YIELD_MINIMO_BAZIN_CONSERVADOR
    constante_graham: float = CONSTANTE_GRAHAM_PADRAO


class ProjecaoLucrosAnuais(BaseModel):
    trimestres_realizados: int
    trimestres_projetados: int
    momento_operacional: MomentoOperacional
    soma_acumulada_ano: float
    lucro_anual_conservador: float
    lucro_anual_mediano: float


class MultiplosCenario(BaseModel):
    LPA: float
    VPA: float
    P_L: float | None = None
    P_VP: float | None = None
    ROE_calculado: float | None = None


class ValuationCenario(BaseModel):
    preco_justo: float
    market_cap_justo: float
    margem_seguranca: float
    upside_downside: float
    crescimento_estagio_1: float
    payout_terminal: float
    taxa_desconto_pct: float
    crescimento_perpetuo_pct: float


class ValuationBazin(BaseModel):
    preco_justo: float
    margem_seguranca: float
    upside_downside: float
    dpa_utilizado: float
    yield_minimo: float
    yield_minimo_pct: float


class ValuationBazinCenario(BaseModel):
    dy_6: ValuationBazin | None = None
    dy_8: ValuationBazin | None = None
    erro_dy_6: str | None = None
    erro_dy_8: str | None = None


class ValuationGraham(BaseModel):
    preco_justo: float
    margem_seguranca: float
    upside_downside: float
    lpa_utilizado: float
    vpa_utilizado: float
    constante_graham: float


class ResumoTriangulacaoCenario(BaseModel):
    ddm_preco_justo: float | None = None
    ddm_margem_seguranca: float | None = None
    ddm_upside_downside: float | None = None
    bazin_6_preco_justo: float | None = None
    bazin_6_margem_seguranca: float | None = None
    bazin_6_upside_downside: float | None = None
    bazin_8_preco_justo: float | None = None
    bazin_8_margem_seguranca: float | None = None
    bazin_8_upside_downside: float | None = None
    graham_preco_justo: float | None = None
    graham_margem_seguranca: float | None = None
    graham_upside_downside: float | None = None
    modelos_triangulados: list[str] = Field(default_factory=list)
    modelos_validos: int = 0
    minimo: float | None = None
    mediana: float | None = None
    maximo: float | None = None
    dispersao_pct: float | None = None
    piso_bazin_8_juros_brasil: float | None = None


class TriangulacaoValuation(BaseModel):
    nota_metodologica: str = (
        "As estatísticas de triangulação (mínimo, mediana, máximo e dispersão) consideram "
        "estritamente um representante por metodologia (DDM de 2 Estágios, Bazin 6% Clássico "
        "e Benjamin Graham) para evitar peso duplo ao método Bazin; Bazin 8% atua como piso "
        "conservador de referência para o cenário de juros brasileiros e IR sobre JCP."
    )
    conservador: ResumoTriangulacaoCenario
    mediano: ResumoTriangulacaoCenario


class ResultadoQuantitativo(BaseModel):
    ticker: str
    setor: str
    preco_atual: float | None = None
    moeda: str = "BRL"
    erro_dados_mercado: str | None = None
    parametros: ParametrosAvaliacao
    projecao_lucros: ProjecaoLucrosAnuais
    multiplos_conservador: MultiplosCenario | None = None
    multiplos_mediano: MultiplosCenario | None = None
    valuation_conservador: ValuationCenario | None = None
    valuation_mediano: ValuationCenario | None = None
    erro_valuation_conservador: str | None = None
    erro_valuation_mediano: str | None = None
    valuation_bazin_conservador: ValuationBazinCenario | None = None
    valuation_bazin_mediano: ValuationBazinCenario | None = None
    valuation_graham_conservador: ValuationGraham | None = None
    valuation_graham_mediano: ValuationGraham | None = None
    erro_valuation_graham_conservador: str | None = None
    erro_valuation_graham_mediano: str | None = None
    triangulacao: TriangulacaoValuation | None = None
    avisos: list[str] = Field(default_factory=list)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    setor: NotRequired[str]
    ticker: NotRequired[str]
    caminhos_pdfs: NotRequired[list[str]]
    tentativas_extracao: int
    dados_extraidos: NotRequired[dict[str, dict[str, Any]]]
    erros_validacao: NotRequired[dict[str, list[str]]]
    validacao_concluida: NotRequired[bool]
    resultado_quantitativo: NotRequired[dict[str, Any]]
    analise_llm_indisponivel: NotRequired[bool]
    erro_analista: NotRequired[str | None]
    tentativas_compliance: NotRequired[int]
    violacoes_compliance: NotRequired[list[str]]
    usar_cache: NotRequired[bool]
    origem_cache: NotRequired[dict[str, bool]]
