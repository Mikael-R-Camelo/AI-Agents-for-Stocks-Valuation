import operator
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    setor: NotRequired[str]
    ticker: NotRequired[str]
    caminhos_pdfs: NotRequired[list[str]]
    tentativas_extracao: int
    dados_extraidos: NotRequired[dict[str, dict[str, Any]]]
    erros_validacao: NotRequired[dict[str, list[str]]]
    validacao_concluida: NotRequired[bool]
    tentativas_compliance: NotRequired[int]
    violacoes_compliance: NotRequired[list[str]]
    usar_cache: NotRequired[bool]
    origem_cache: NotRequired[dict[str, bool]]
