from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import obter_llm
from graph.nodes import (
    analista_node,
    compliance_node,
    extrator_node,
    roteador_compliance,
    roteador_validador,
    validador_node,
)
from graph.state import AgentState
from tools.market_data import (
    buscar_dados_mercado_yfinance,
    buscar_medias_historicas_yfinance,
    buscar_payout_medio_historico_yfinance,
    calcular_multiplos_basicos,
    projetar_lucro_anual_cenarios,
)
from tools.valuation import fluxo_de_caixa_futuro_buffet

FERRAMENTAS = [
    buscar_dados_mercado_yfinance,
    buscar_medias_historicas_yfinance,
    buscar_payout_medio_historico_yfinance,
    projetar_lucro_anual_cenarios,
    calcular_multiplos_basicos,
    fluxo_de_caixa_futuro_buffet,
]


def build_app(llm: Any = None, model_with_tools: Any = None):
    """
    Constrói e compila o grafo LangGraph com injeção opcional de LLM/model_with_tools.
    Permite testes determinísticos sem exigir GOOGLE_API_KEY no tempo de importação.
    """
    cache_local: dict[str, Any] = {
        "llm": llm,
        "model_with_tools": model_with_tools,
    }

    def _resolver_llm() -> Any:
        if cache_local["llm"] is None:
            cache_local["llm"] = obter_llm()
        return cache_local["llm"]

    def _resolver_model_with_tools() -> Any:
        if cache_local["model_with_tools"] is None:
            modelo = _resolver_llm()
            cache_local["model_with_tools"] = (
                modelo.bind_tools(FERRAMENTAS)
                if hasattr(modelo, "bind_tools")
                else modelo
            )
        return cache_local["model_with_tools"]

    def _extrator(state: AgentState) -> dict[str, Any]:
        return extrator_node(state, llm=cache_local["llm"])

    def _analista(state: AgentState) -> dict[str, Any]:
        return analista_node(state, model_with_tools=_resolver_model_with_tools())

    workflow = StateGraph(AgentState)

    workflow.add_node("extrator", _extrator)
    workflow.add_node("validador", validador_node)
    workflow.add_node("analista", _analista)
    workflow.add_node("ferramentas", ToolNode(FERRAMENTAS))
    workflow.add_node("compliance", compliance_node)

    workflow.add_edge(START, "extrator")
    workflow.add_edge("extrator", "validador")

    workflow.add_conditional_edges(
        "validador",
        roteador_validador,
        {"extrator": "extrator", "analista": "analista", "fim": END},
    )

    workflow.add_conditional_edges(
        "analista", tools_condition, {"tools": "ferramentas", "__end__": "compliance"}
    )
    workflow.add_edge("ferramentas", "analista")

    workflow.add_conditional_edges(
        "compliance",
        roteador_compliance,
        {"analista": "analista", "fim": END},
    )

    return workflow.compile()
