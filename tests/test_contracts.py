from unittest.mock import MagicMock, patch

import pandas as pd
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
    CONSTANTE_GRAHAM_PADRAO,
    CRESCIMENTO_PERPETUO_PADRAO,
    DEFAULT_GEMINI_MODEL,
    TAXA_DESCONTO_PADRAO,
    YIELD_MINIMO_BAZIN_CONSERVADOR,
    YIELD_MINIMO_BAZIN_PADRAO,
    obter_modelo_gemini,
    obter_modelo_gemini_fallback,
)
from graph.workflow import build_app
from tools.utils import (
    MAPA_SETORES_YFINANCE,
    chave_ordenacao_trimestre,
    normalizar_ticker_b3,
)


def _mock_yfinance_basico() -> MagicMock:
    cols = pd.to_datetime(["2025-12-31"])
    mock_ticker = MagicMock()
    mock_ticker.info = {"currentPrice": 13.44, "currency": "BRL", "payoutRatio": 0.4839}
    mock_ticker.financials = pd.DataFrame([[113.5]], index=["Net Income"], columns=cols)
    mock_ticker.balance_sheet = pd.DataFrame(
        [[1000.0]], index=["Stockholders Equity"], columns=cols
    )
    mock_ticker.cash_flow = pd.DataFrame(
        [[0.0]], index=["Cash Dividends Paid"], columns=cols
    )
    return mock_ticker


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


def test_modelo_gemini_e_fallback_configuraveis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    assert obter_modelo_gemini() == DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"
    assert obter_modelo_gemini_fallback() is None

    # Se fallback for igual ao primário, deve retornar None
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_MODEL)
    assert obter_modelo_gemini_fallback() is None

    # Se fallback for distinto do primário, retorna o modelo reserva
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
    assert obter_modelo_gemini_fallback() == "gemini-2.5-flash"

    monkeypatch.setenv("GEMINI_MODEL", "gemini-override-test")
    assert obter_modelo_gemini() == "gemini-override-test"


def test_interpolacao_dinamica_prompts() -> None:
    prompt_analista = montar_prompt_sistema_analista(obter_protocolo("banco"))
    assert f"taxa_desconto={TAXA_DESCONTO_PADRAO}" in prompt_analista
    assert f"crescimento_perpetuo={CRESCIMENTO_PERPETUO_PADRAO}" in prompt_analista
    assert f"yield_minimo={YIELD_MINIMO_BAZIN_PADRAO}" in prompt_analista
    assert f"yield_minimo={YIELD_MINIMO_BAZIN_CONSERVADOR}" in prompt_analista
    assert f"constante_graham={CONSTANTE_GRAHAM_PADRAO}" in prompt_analista
    assert "NÃO deve chamar ferramentas" in prompt_analista
    assert "resultado_quantitativo" in prompt_analista
    assert "Triangulação de Valuation" in prompt_analista
    assert "sem peso duplo" in prompt_analista
    assert "IR de 15% sobre JCP" in prompt_analista
    assert "LPA Projetado (Forward)" in prompt_analista
    assert "P/L Projetado (Forward)" in prompt_analista
    assert "VPA Atual" in prompt_analista
    assert "P/VP Atual" in prompt_analista
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
    llm_ok.invoke.side_effect = [
        AIMessage(content="Preço Justo R$ 18,56. Recomendação: Compra."),
        AIMessage(
            content="Preço Justo R$ 18,56. Este estudo não constitui recomendação de compra ou venda."
        ),
    ]
    app_ok = build_app(llm=llm_ok)
    with (
        patch("graph.nodes.extrair_dados_do_pdf", return_value=tri_valido),
        patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_basico()),
    ):
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
    llm_teimoso.invoke.return_value = AIMessage(
        content="Rating: Buy. Recomendamos comprar o papel."
    )
    app_cap = build_app(llm=llm_teimoso)
    with (
        patch("graph.nodes.extrair_dados_do_pdf", return_value=tri_valido),
        patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_basico()),
    ):
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
