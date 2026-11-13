import os

from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o ambiente do sistema
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY não foi encontrada. "
        "Defina essa variável no arquivo .env antes de executar o programa."
    )

from typing import Annotated, NotRequired

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from sector_protocols import obter_protocolo
from tools.valuation import fluxo_de_caixa_futuro_buffet
from typing_extensions import TypedDict


@tool
def ferramenta_valuation_buffet(
    lucro_base: float,
    payout: float,
    roe: float,
    taxa_desconto: float,
    crescimento_perpetuo: float,
    num_acoes: float,
    preco_atual: float,
    crescimento_ano_1: float = 0.0,
    anos_projecao: int = 4,
) -> dict:
    """Calcula o Valuation (Preço Justo) utilizando o método de fluxo de caixa futuro."""
    return fluxo_de_caixa_futuro_buffet(
        lucro_base,
        payout,
        roe,
        taxa_desconto,
        crescimento_perpetuo,
        num_acoes,
        preco_atual,
        crescimento_ano_1,
        anos_projecao,
    )


ferramentas = [ferramenta_valuation_buffet]


# CONFIGURAÇÃO DO ESTADO E DO LLM
class State(TypedDict):
    messages: Annotated[list, add_messages]
    setor: NotRequired[str]


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    # timeout=60,
    max_retries=10,
)
llm_com_ferramentas = llm.bind_tools(ferramentas)


# 3. O NÓ DO ANALISTA
def analista_node(state: State):
    mensagens = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in mensagens):
        protocolo = obter_protocolo(state.get("setor", "banco"))
        prompt_sistema = SystemMessage(
            content=(
                f"Você é um analista sênior especializado em {protocolo.nome}. "
                "Você receberá dados extraídos de balanços em formato JSON. "
                f"{protocolo.instrucoes} "
                "Sua tarefa é usar a ferramenta de valuation para calcular o "
                "preço teto e, em seguida, escrever um breve relatório "
                "recomendando se a ação está cara ou barata. "
                "Se algum dado necessário estiver ausente, informe a ausência "
                "em vez de inventar um valor."
            )
        )
        mensagens = [prompt_sistema] + mensagens

    print("[LLM] Enviando dados para o modelo...", flush=True)
    resposta = llm_com_ferramentas.invoke(mensagens)
    return {"messages": [resposta]}


# 4. CONSTRUÇÃO DO GRAFO
grafo = StateGraph(State)
grafo.add_node("Analista", analista_node)
grafo.add_node("Calculadora", ToolNode(ferramentas))

grafo.add_edge(START, "Analista")
grafo.add_conditional_edges(
    "Analista", tools_condition, {"tools": "Calculadora", "__end__": END}
)
grafo.add_edge("Calculadora", "Analista")

app = grafo.compile()

# 5. TESTE PRÁTICO COM MOCK JSON
if __name__ == "__main__":
    mock_json_bbdc3 = """
    {
      "ticker": "BBDC3",
      "setor": "banco",
      "lucro_base_2024": 17542153000,
      "payout_medio": 0.50,
      "roe_atual": 0.1165,
      "taxa_desconto_wacc": 0.15,
      "crescimento_perpetuo_esperado": 0.03,
      "total_acoes": 10592012000,
      "preco_tela_atual": 16.72,
      "estimativa_crescimento_ano_1": 0.30
    }
    """

    mensagem_usuario = HumanMessage(
        content=f"Aqui estão os dados financeiros que extraí do Bradesco: {mock_json_bbdc3}. Por favor, avalie."
    )

    print("\n--- INICIANDO ANÁLISE ---")
    for evento in app.stream({"messages": [mensagem_usuario], "setor": "banco"}):
        for nome_no, estado in evento.items():
            if nome_no == "Analista":
                msg = estado["messages"][-1]
                if msg.tool_calls:
                    print(
                        f"[Analista] Decidiu usar a calculadora com os parâmetros: {msg.tool_calls[0]['args']}\n"
                    )
                else:
                    print(f"[Relatório Final do Agente Analista]\n{msg.content}")
            elif nome_no == "Calculadora":
                resultado = estado["messages"][-1]
                print(
                    f"[Python / Matemática] O resultado exato da função foi: {resultado.content}\n"
                )
