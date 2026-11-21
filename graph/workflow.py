from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    analista_node,
    calculadora_node,
    compliance_node,
    extrator_node,
    roteador_compliance,
    roteador_validador,
    validador_node,
)
from graph.state import AgentState


def build_app(llm: Any = None, fallback_llm: Any = None):
    """
    Constrói e compila o grafo LangGraph com injeção opcional de LLM primário e fallback.
    Topologia desacoplada: extrator -> validador -> calculadora (determinística) -> analista -> compliance.
    """

    def _extrator(state: AgentState) -> dict[str, Any]:
        return extrator_node(state, llm=llm)

    def _analista(state: AgentState) -> dict[str, Any]:
        return analista_node(state, llm=llm, fallback_llm=fallback_llm)

    workflow = StateGraph(AgentState)

    workflow.add_node("extrator", _extrator)
    workflow.add_node("validador", validador_node)
    workflow.add_node("calculadora", calculadora_node)
    workflow.add_node("analista", _analista)
    workflow.add_node("compliance", compliance_node)

    workflow.add_edge(START, "extrator")
    workflow.add_edge("extrator", "validador")

    workflow.add_conditional_edges(
        "validador",
        roteador_validador,
        {"extrator": "extrator", "calculadora": "calculadora", "fim": END},
    )

    workflow.add_edge("calculadora", "analista")
    workflow.add_edge("analista", "compliance")

    workflow.add_conditional_edges(
        "compliance",
        roteador_compliance,
        {"analista": "analista", "fim": END},
    )

    return workflow.compile()
