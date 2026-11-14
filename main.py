import os

from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o ambiente do sistema
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY não foi encontrada. "
        "Defina essa variável no arquivo .env antes de executar o programa."
    )

import operator
from typing import Annotated, NotRequired

from agents.sector_protocols import obter_protocolo
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from tools.valuation import fluxo_de_caixa_futuro_buffet
from typing_extensions import TypedDict

# ==========================================
# Definindo as ferramentas e o modelo do agente
# ==========================================

ferramentas = [fluxo_de_caixa_futuro_buffet]

# Usamos a classe nativa do Google em vez do init_chat_model para garantir as integrações
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, max_retries=3)
model_with_tools = llm.bind_tools(ferramentas)


# ==========================================
# Definindo o estado do agente
# ==========================================
class AgentState(TypedDict):
    # O operator.add garante que as mensagens sejam empilhadas e nunca sobrescritas
    messages: Annotated[list[AnyMessage], operator.add]
    setor: NotRequired[str]


# ==========================================
# Definindo o nó do agente analista
# ==========================================
def analista_node(state: AgentState):
    mensagens = state["messages"]

    # Adiciona o System Prompt dinâmico apenas se for o início da conversa
    if not any(isinstance(m, SystemMessage) for m in mensagens):
        protocolo = obter_protocolo(state.get("setor", "banco"))
        prompt_sistema = SystemMessage(
            content=(
                f"Você é um analista sênior especializado em {protocolo.nome}. "
                f"{protocolo.instrucoes} "
                "Use a ferramenta de valuation para calcular o preço teto e, em seguida, "
                "escreva um relatório recomendando se a ação está cara ou barata."
            )
        )
        mensagens = [prompt_sistema] + mensagens

    print("[Analista] Processando dados...")
    resposta = model_with_tools.invoke(mensagens)

    # O retorno deve ser sempre um dicionário com a chave que queremos atualizar no estado
    return {"messages": [resposta]}


# ==========================================
# Criando o fluxo de trabalho do agente
# ==========================================
workflow = StateGraph(AgentState)

# Adiciona os nós
workflow.add_node("analista", analista_node)
workflow.add_node("ferramentas", ToolNode(ferramentas))  # Faz o papel do 'tool_node'

# Conecta o fluxo
workflow.add_edge(START, "analista")

# Se o LLM pedir ferramenta, vai para "ferramentas". Se não pedir, vai para END.
workflow.add_conditional_edges(
    "analista",
    tools_condition,  # Faz o papel do 'should_continue'
    {"tools": "ferramentas", "__end__": END},
)

# Após a ferramenta rodar, devolve para o analista ler o resultado
workflow.add_edge("ferramentas", "analista")

# Compila o agente
app = workflow.compile()

# TODO: Elaborar o agente de extração de dados para ler os balanços e resultados
#       trimestrais e gerar o JSON de entrada pra o agente analista.
if __name__ == "__main__":
    mock_json_bbdc3 = (
        "{"
        '"lucro_base_2024": 17542153000, '
        '"payout_medio": 0.50, '
        '"roe_atual": 0.1165, '
        '"taxa_desconto_wacc": 0.15, '
        '"crescimento_perpetuo_esperado": 0.03, '
        '"total_acoes": 10592012000, '
        '"preco_atual": 16.72}'
    )

    mensagem_inicial = HumanMessage(content=f"Dados extraídos: {mock_json_bbdc3}")

    # Invoca o agente passando a mensagem e o setor
    resultado_final = app.invoke({"messages": [mensagem_inicial], "setor": "banco"})

    # Imprime apenas o relatório final (a última mensagem do LLM)
    print("\n--- RELATÓRIO FINAL ---")
    resultado_final["messages"][-1].pretty_print()
