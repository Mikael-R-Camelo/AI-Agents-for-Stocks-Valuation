from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agents.compliance import (
    MAX_TENTATIVAS_COMPLIANCE,
    ResultadoCompliance,
    auditar_compliance_relatorio,
)
from agents.extrator import DadosTrimestraisPDF
from agents.prompts import montar_prompt_extrator, montar_prompt_sistema_analista
from agents.sector_protocols import PROTOCOLOS_SETOR, obter_protocolo
from config import (
    CRESCIMENTO_PERPETUO_PADRAO,
    DEFAULT_GEMINI_MODEL,
    TAXA_DESCONTO_PADRAO,
    obter_modelo_gemini,
)
from graph.workflow import build_app
from tools.utils import (
    MAPA_SETORES_YFINANCE,
    chave_ordenacao_trimestre,
    normalizar_ticker_b3,
)


def test_todos_os_setores_mapeados_possuem_protocolo() -> None:
    setores_esperados = set(MAPA_SETORES_YFINANCE.values()) | {
        "geral",
        "eletrica",
        "holding",
    }
    for setor in setores_esperados:
        protocolo = obter_protocolo(setor)
        assert protocolo.nome and protocolo.instrucoes
        assert setor in PROTOCOLOS_SETOR

    with pytest.raises(ValueError):
        obter_protocolo("setor_inexistente")


def test_normalizacao_ticker_e_ordenacao_trimestres() -> None:
    assert normalizar_ticker_b3(" bbdc3 ") == "BBDC3.SA"
    assert normalizar_ticker_b3("bbdc3.sa ") == "BBDC3.SA"
    assert normalizar_ticker_b3("ITUB4.SA") == "ITUB4.SA"
    with pytest.raises(ValueError):
        normalizar_ticker_b3("   ")

    trimestres = ["2T26", "4T25", "1T26", "3T2025", "invalido"]
    assert sorted(trimestres, key=chave_ordenacao_trimestre) == [
        "3T2025",
        "4T25",
        "1T26",
        "2T26",
        "invalido",
    ]


def test_modelo_gemini_configuravel_com_default_preservado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert obter_modelo_gemini() == DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"

    monkeypatch.setenv("GEMINI_MODEL", "gemini-override-test")
    assert obter_modelo_gemini() == "gemini-override-test"


def test_interpolacao_dinamica_prompts() -> None:
    prompt_analista = montar_prompt_sistema_analista(obter_protocolo("banco"))
    assert f"taxa_desconto={TAXA_DESCONTO_PADRAO}" in prompt_analista
    assert f"crescimento_perpetuo={CRESCIMENTO_PERPETUO_PADRAO}" in prompt_analista
    assert "payout_sustentavel" in prompt_analista
    assert "numero_total_acoes" in prompt_analista
    assert "JCP" in prompt_analista

    prompt_extrator = montar_prompt_extrator("conteudo", feedback_correcao="- erro ROE")
    assert "conteudo" in prompt_extrator
    assert "- erro ROE" in prompt_extrator


@pytest.mark.parametrize(
    "texto_proibido",
    [
        "Recomendação: Compra",
        "**Recomendação:** **Venda**",
        "Recomendação final: Manter",
        "Recomendamos comprar BBDC3 neste patamar.",
        "Recomendamos vender o papel imediatamente.",
        "Sugerimos manter a posição no ativo.",
        "Rating: Buy",
        "Rating: Sell",
        "Rating: Hold",
        "O desconto atual representa um sinal de compra claro.",
        "A deterioração gera sinal de venda.",
        "Não há garantia de retorno, porém recomendamos comprar a ação.",
        "Os investidores devem comprar as ações do banco.",
    ],
)
def test_compliance_bloqueia_recomendacoes_explicitas(texto_proibido: str) -> None:
    res = auditar_compliance_relatorio(texto_proibido)
    assert isinstance(res, ResultadoCompliance)
    assert not res.aprovado
    assert res.violacoes


@pytest.mark.parametrize(
    "texto_valido",
    [
        "A venda de ativos não estratégicos e a compra de carteira de crédito ajudaram a manter a rentabilidade.",
        "Este relatório tem caráter informativo e não constitui recomendação de compra ou venda.",
        "Aviso legal: não se trata de recomendação de compra, venda ou manutenção de valores mobiliários.",
        "Disclaimer: não recomendamos comprar ou vender ações com base isolada neste documento.",
        "Sem caráter de recomendação de compra ou venda.",
        "O banco optou por manter o apetite de risco moderado e expandir a venda de seguros.",
    ],
)
def test_compliance_nao_gera_falsos_positivos(texto_valido: str) -> None:
    res = auditar_compliance_relatorio(texto_valido)
    assert res.aprovado
    assert res.violacoes == []


def test_loop_compliance_reescreve_e_respeita_limite() -> None:
    tri_valido = DadosTrimestraisPDF(
        trimestre_referencia="3T26",
        lucro_liquido_trimestral=7_110_000_000.0,
        patrimonio_liquido=182_282_051_282.0,
        roe_reportado=0.161,
        payout_estimado=0.50,
        resumo_momento_operacional="Operacao estavel.",
        visao_da_gestao="Tom cauteloso.",
        numero_total_acoes=11_178_825_623.0,
    ).model_dump()

    # Caso 1: corrige na 1ª reescrita
    llm_ok = MagicMock()
    llm_ok.bind_tools.return_value = llm_ok
    llm_ok.invoke.side_effect = [
        AIMessage(content="Preço Justo R$ 18,56. Recomendação: Compra."),
        AIMessage(
            content="Preço Justo R$ 18,56. Este estudo não constitui recomendação de compra ou venda."
        ),
    ]
    app_ok = build_app(llm=llm_ok)
    with patch("graph.nodes.extrair_dados_do_pdf", return_value=tri_valido):
        out_ok = app_ok.invoke(
            {
                "messages": [],
                "setor": "banco",
                "ticker": "BBDC3",
                "caminhos_pdfs": ["3T26.pdf"],
                "tentativas_extracao": 0,
                "usar_cache": False,
            }
        )
    assert out_ok["tentativas_compliance"] == 1
    assert out_ok["violacoes_compliance"] == []
    assert "ALERTA DE COMPLIANCE" not in out_ok["messages"][-1].content

    # Caso 2: persiste violando -> para após MAX_TENTATIVAS_COMPLIANCE e anexa alerta
    llm_teimoso = MagicMock()
    llm_teimoso.bind_tools.return_value = llm_teimoso
    llm_teimoso.invoke.return_value = AIMessage(
        content="Rating: Buy. Recomendamos comprar o papel."
    )
    app_cap = build_app(llm=llm_teimoso)
    with patch("graph.nodes.extrair_dados_do_pdf", return_value=tri_valido):
        out_cap = app_cap.invoke(
            {
                "messages": [],
                "setor": "banco",
                "ticker": "BBDC3",
                "caminhos_pdfs": ["3T26.pdf"],
                "tentativas_extracao": 0,
                "usar_cache": False,
            }
        )
    assert out_cap["tentativas_compliance"] == MAX_TENTATIVAS_COMPLIANCE + 1
    assert llm_teimoso.invoke.call_count == MAX_TENTATIVAS_COMPLIANCE + 1
    assert "ALERTA DE COMPLIANCE AUTOMATIZADO" in out_cap["messages"][-1].content
