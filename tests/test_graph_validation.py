import json
import math
import statistics
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from agents.extrator import (
    DadosTrimestraisPDF,
    carregar_cache_extracao,
    salvar_cache_extracao,
    selecionar_texto_relevante_pdf,
)
from config import MAX_TENTATIVAS_EXTRACAO
from graph.nodes import (
    _classificar_momento_operacional,
    calculadora_node,
    roteador_compliance,
    validar_trimestre,
)
from graph.state import ResultadoQuantitativo
from graph.workflow import build_app


def _fazer_trimestre(
    ref: str = "3T26",
    lucro: float = 7_109_000_000.0,
    pl: float = 182_282_051_282.0,
    roe: float = 0.161,
    payout: float = 0.50,
    num_acoes: float = 11_178_825_623.0,
    caixa_i: float | None = None,
    caixa_f: float | None = None,
    var_c: float | None = None,
    cambial: float | None = None,
) -> dict:
    return DadosTrimestraisPDF(
        trimestre_referencia=ref,
        lucro_liquido_trimestral=lucro,
        patrimonio_liquido=pl,
        roe_reportado=roe,
        payout_estimado=payout,
        resumo_momento_operacional="Operacao estavel no trimestre.",
        visao_da_gestao="Diretoria manteve tom cauteloso.",
        numero_total_acoes=num_acoes,
        caixa_inicial=caixa_i,
        caixa_final=caixa_f,
        variacao_liquida_caixa=var_c,
        efeito_cambial_caixa=cambial,
    ).model_dump()


def _mock_yfinance_bbdc3() -> MagicMock:
    """Mock determinístico de yf.Ticker simulando BBDC3 com JCP (DFC zerada + info.payoutRatio)."""
    cols = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "currentPrice": 13.44,
        "currency": "BRL",
        "payoutRatio": 0.4839,
        "sharesOutstanding": 5_000_000_000,  # Distorcido ON/PN — deve ser IGNORADO
        "bookValue": 99.99,  # Deve ser IGNORADO
    }
    mock_ticker.financials = pd.DataFrame(
        [[132.7, 102.4, 85.7, 133.0]], index=["Net Income"], columns=cols
    )
    mock_ticker.balance_sheet = pd.DataFrame(
        [[1000.0, 1000.0, 1000.0, 1000.0]],
        index=["Stockholders Equity"],
        columns=cols,
    )
    mock_ticker.cash_flow = pd.DataFrame(
        [[0.0, 0.0, 0.0, 0.0]], index=["Cash Dividends Paid"], columns=cols
    )
    return mock_ticker


class TestValidarTrimestre:
    def test_aprova_roe_trimestral_anualizado_e_dfc_opcional_banco(self) -> None:
        dados = _fazer_trimestre()
        assert validar_trimestre(dados) == []

    def test_aprova_roe_anualizado_linear_e_composto(self) -> None:
        # ROE trimestral = 6.5% -> linear 26.0%, composto 28.65%
        assert validar_trimestre(_fazer_trimestre(lucro=65.0, pl=1000.0, roe=0.26)) == []
        assert (
            validar_trimestre(_fazer_trimestre(lucro=65.0, pl=1000.0, roe=0.2865)) == []
        )

    def test_detecta_lucro_acumulado_9m_e_escala_percentual(self) -> None:
        erros_9m = validar_trimestre(_fazer_trimestre(lucro=21_000_000_000.0, roe=0.161))
        assert any("ROE" in e for e in erros_9m)

        erros_escala = validar_trimestre(_fazer_trimestre(roe=16.1))
        assert any("ESCALA" in e for e in erros_escala)

    def test_trata_pl_zero_e_acoes_invalidas_sem_zerodivision(self) -> None:
        erros = validar_trimestre(_fazer_trimestre(pl=0.0, num_acoes=0.0))
        assert any("PATRIMÔNIO LÍQUIDO" in e for e in erros)
        assert any("AÇÕES" in e for e in erros)

    def test_pl_negativo_nao_dispara_erro_de_divisao(self) -> None:
        erros = validar_trimestre(_fazer_trimestre(pl=-100.0))
        assert erros == []

    def test_dfc_com_efeito_cambial_e_dfc_parcial_ou_divergente(self) -> None:
        assert (
            validar_trimestre(
                _fazer_trimestre(
                    caixa_i=1000.0, var_c=200.0, cambial=50.0, caixa_f=1250.0
                )
            )
            == []
        )

        erros_parcial = validar_trimestre(
            _fazer_trimestre(caixa_i=1000.0, caixa_f=None, var_c=200.0)
        )
        assert any("parcialmente" in e for e in erros_parcial)

        erros_div = validar_trimestre(
            _fazer_trimestre(caixa_i=1000.0, var_c=100.0, caixa_f=1500.0)
        )
        assert any("CAIXA" in e for e in erros_div)


class TestClassificarMomentoOperacional:
    @pytest.mark.parametrize(
        ("lucros", "esperado"),
        [
            ([100.0], "estavel"),
            ([100.0, 101.0], "estavel"),
            ([100.0, 100.5, 100.2], "estavel"),
            ([100.0, 105.0], "recuperacao"),
            ([6_811.0, 7_050.0, 7_109.0], "recuperacao"),
            ([100.0, 95.0], "deteriorando"),
            ([120.0, 110.0, 100.0], "deteriorando"),
            ([-120.0, -100.0, -80.0], "recuperacao"),
            ([-80.0, -100.0, -120.0], "deteriorando"),
            ([0.0, 50.0], "recuperacao"),
            ([0.0, -50.0], "deteriorando"),
            ([0.0, 0.0], "estavel"),
        ],
    )
    def test_classifica_trajetorias_lucros_e_prejuizos(
        self, lucros: list[float], esperado: str
    ) -> None:
        assert _classificar_momento_operacional(lucros) == esperado


class TestCalculadoraDeterministica:
    def test_calculadora_bbdc3_usa_acoes_pdf_fallback_jcp_e_grava_json(
        self, tmp_path: Path
    ) -> None:
        pasta_ativo = tmp_path / "BBDC3"
        pasta_ativo.mkdir()
        p1 = pasta_ativo / "1T26.pdf"
        p2 = pasta_ativo / "2T26.pdf"
        p3 = pasta_ativo / "3T26.pdf"
        for p in (p1, p2, p3):
            p.write_bytes(b"%PDF-1.4")

        state = {
            "messages": [],
            "ticker": "BBDC3",
            "setor": "banco",
            "caminhos_pdfs": [str(p1), str(p2), str(p3)],
            "tentativas_extracao": 1,
            "dados_extraidos": {
                str(p1): _fazer_trimestre(
                    "1T26", lucro=6_811_000_000.0, pl=175_000_000_000.0, roe=0.158
                ),
                str(p2): _fazer_trimestre(
                    "2T26", lucro=7_050_000_000.0, pl=179_000_000_000.0, roe=0.162
                ),
                str(p3): _fazer_trimestre(
                    "3T26", lucro=7_109_000_000.0, pl=182_282_051_282.0, roe=0.161
                ),
            },
            "erros_validacao": {},
            "validacao_concluida": True,
        }

        with patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()):
            out = calculadora_node(state)

        quant = ResultadoQuantitativo.model_validate(out["resultado_quantitativo"])
        assert quant.preco_atual == 13.44
        assert quant.parametros.num_acoes_pdf == 11_178_825_623.0
        assert quant.parametros.fonte_payout == "yfinance_info_payoutRatio_fallback"
        assert quant.parametros.payout_sustentavel == 0.4839
        assert quant.parametros.momento_operacional == "recuperacao"
        assert quant.multiplos_conservador is not None
        assert quant.multiplos_conservador.VPA == 16.31
        assert quant.valuation_conservador is not None
        assert quant.valuation_mediano is not None
        assert quant.valuation_conservador.preco_justo > 16.5
        assert quant.valuation_mediano.preco_justo > quant.valuation_conservador.preco_justo

        # Valida novos parâmetros e modelos Bazin (6% e 8%) + Graham (22.5)
        assert quant.parametros.yield_minimo_bazin_padrao == 0.06
        assert quant.parametros.yield_minimo_bazin_conservador == 0.08
        assert quant.parametros.constante_graham == 22.5

        assert quant.valuation_bazin_conservador is not None
        assert quant.valuation_bazin_conservador.dy_6 is not None
        assert quant.valuation_bazin_conservador.dy_8 is not None
        assert (
            quant.valuation_bazin_conservador.dy_8.preco_justo
            < quant.valuation_bazin_conservador.dy_6.preco_justo
        )

        assert quant.valuation_bazin_mediano is not None
        assert quant.valuation_bazin_mediano.dy_6 is not None
        assert quant.valuation_bazin_mediano.dy_8 is not None
        assert (
            quant.valuation_bazin_mediano.dy_8.preco_justo
            < quant.valuation_bazin_mediano.dy_6.preco_justo
        )

        assert quant.valuation_graham_conservador is not None
        assert quant.valuation_graham_mediano is not None
        lpa_med_exato = (
            quant.projecao_lucros.lucro_anual_mediano / quant.parametros.num_acoes_pdf
        )
        vpa_exato = (
            quant.parametros.patrimonio_liquido_pdf / quant.parametros.num_acoes_pdf
        )
        graham_med_esperado = round(math.sqrt(22.5 * lpa_med_exato * vpa_exato), 2)
        assert quant.valuation_graham_mediano.preco_justo == graham_med_esperado == 30.36

        # Valida Triangulação SEM peso duplo para Bazin 8% (apenas DDM, Bazin 6% e Graham)
        assert quant.triangulacao is not None
        tri_c = quant.triangulacao.conservador
        tri_m = quant.triangulacao.mediano

        esperados_c = [
            quant.valuation_conservador.preco_justo,
            quant.valuation_bazin_conservador.dy_6.preco_justo,
            quant.valuation_graham_conservador.preco_justo,
        ]
        assert tri_c.modelos_triangulados == ["ddm", "bazin_6", "graham"]
        assert tri_c.modelos_validos == 3
        assert tri_c.minimo == min(esperados_c)
        assert tri_c.maximo == max(esperados_c)
        assert tri_c.mediana == round(statistics.median(esperados_c), 2)
        assert tri_c.piso_bazin_8_juros_brasil == quant.valuation_bazin_conservador.dy_8.preco_justo
        assert tri_c.bazin_8_preco_justo == quant.valuation_bazin_conservador.dy_8.preco_justo
        assert tri_c.dispersao_pct is not None and tri_c.dispersao_pct > 0

        esperados_m = [
            quant.valuation_mediano.preco_justo,
            quant.valuation_bazin_mediano.dy_6.preco_justo,
            quant.valuation_graham_mediano.preco_justo,
        ]
        assert tri_m.modelos_triangulados == ["ddm", "bazin_6", "graham"]
        assert tri_m.modelos_validos == 3
        assert tri_m.minimo == min(esperados_m)
        assert tri_m.maximo == max(esperados_m)
        assert tri_m.mediana == round(statistics.median(esperados_m), 2)
        assert tri_m.piso_bazin_8_juros_brasil == quant.valuation_bazin_mediano.dy_8.preco_justo

        # Confirma persistência do JSON quantitativo na pasta do ativo
        caminho_json = pasta_ativo / "Valuation_BBDC3_quant.json"
        assert caminho_json.is_file()
        salvo = json.loads(caminho_json.read_text(encoding="utf-8"))
        assert salvo["ticker"] == "BBDC3"
        assert salvo["valuation_mediano"]["preco_justo"] == quant.valuation_mediano.preco_justo
        assert salvo["valuation_bazin_mediano"]["dy_6"]["preco_justo"] == quant.valuation_bazin_mediano.dy_6.preco_justo
        assert salvo["valuation_bazin_mediano"]["dy_8"]["preco_justo"] == quant.valuation_bazin_mediano.dy_8.preco_justo
        assert salvo["valuation_graham_mediano"]["preco_justo"] == quant.valuation_graham_mediano.preco_justo
        assert salvo["triangulacao"]["mediano"]["mediana"] == tri_m.mediana

    def test_calculadora_degrada_limpo_em_empresa_com_prejuizo(self) -> None:
        state = {
            "messages": [],
            "ticker": "PREJ3",
            "setor": "varejo",
            "caminhos_pdfs": ["1T26.pdf"],
            "tentativas_extracao": 1,
            "dados_extraidos": {
                "1T26.pdf": _fazer_trimestre(
                    "1T26", lucro=-500_000_000.0, pl=2_000_000_000.0, roe=-0.25
                )
            },
            "erros_validacao": {},
            "validacao_concluida": True,
        }

        with patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()):
            out = calculadora_node(state)

        quant = ResultadoQuantitativo.model_validate(out["resultado_quantitativo"])
        assert quant.multiplos_conservador is not None
        assert quant.multiplos_conservador.P_L is None
        assert quant.valuation_conservador is None
        assert quant.erro_valuation_conservador is not None
        assert quant.valuation_bazin_conservador is not None
        assert quant.valuation_bazin_conservador.dy_6 is None
        assert quant.valuation_bazin_conservador.dy_8 is None
        assert quant.valuation_bazin_conservador.erro_dy_6 is not None
        assert quant.valuation_bazin_conservador.erro_dy_8 is not None
        assert quant.valuation_graham_conservador is None
        assert quant.erro_valuation_graham_conservador is not None
        assert quant.triangulacao is not None
        assert quant.triangulacao.conservador.modelos_validos == 0
        assert quant.triangulacao.conservador.mediana is None

    def test_calculadora_prejuizo_e_pl_negativo_simultaneos_sem_crash(self) -> None:
        state = {
            "messages": [],
            "ticker": "CRISE3",
            "setor": "varejo",
            "caminhos_pdfs": ["1T26.pdf"],
            "tentativas_extracao": 1,
            "dados_extraidos": {
                "1T26.pdf": _fazer_trimestre(
                    "1T26", lucro=-500_000_000.0, pl=-2_000_000_000.0, roe=-0.25
                )
            },
            "erros_validacao": {},
            "validacao_concluida": True,
        }

        with patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()):
            out = calculadora_node(state)

        quant = ResultadoQuantitativo.model_validate(out["resultado_quantitativo"])
        assert quant.valuation_conservador is None
        assert quant.valuation_bazin_conservador is not None
        assert quant.valuation_bazin_conservador.dy_6 is None
        assert quant.valuation_bazin_conservador.dy_8 is None
        # Graham jamais pode calcular sqrt((-LPA) * (-VPA))
        assert quant.valuation_graham_conservador is None
        assert quant.valuation_graham_mediano is None
        assert quant.erro_valuation_graham_conservador is not None
        assert quant.triangulacao is not None
        assert quant.triangulacao.conservador.modelos_validos == 0
        assert quant.triangulacao.conservador.mediana is None

    def test_calculadora_pl_negativo_com_lucro_e_payout_positivos_degrada_apenas_graham(
        self,
    ) -> None:
        state = {
            "messages": [],
            "ticker": "PLNEG3",
            "setor": "varejo",
            "caminhos_pdfs": ["1T26.pdf"],
            "tentativas_extracao": 1,
            "dados_extraidos": {
                "1T26.pdf": _fazer_trimestre(
                    "1T26",
                    lucro=500_000_000.0,
                    pl=-2_000_000_000.0,
                    roe=0.15,
                    payout=0.50,
                    num_acoes=1_000_000_000.0,
                )
            },
            "erros_validacao": {},
            "validacao_concluida": True,
        }

        with patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()):
            out = calculadora_node(state)

        quant = ResultadoQuantitativo.model_validate(out["resultado_quantitativo"])
        assert quant.valuation_conservador is not None
        assert quant.valuation_bazin_conservador is not None
        assert quant.valuation_bazin_conservador.dy_6 is not None
        assert quant.valuation_bazin_conservador.dy_8 is not None
        assert quant.valuation_graham_conservador is None
        assert quant.erro_valuation_graham_conservador is not None
        assert quant.triangulacao is not None
        tri_c = quant.triangulacao.conservador
        assert tri_c.modelos_triangulados == ["ddm", "bazin_6"]
        assert tri_c.modelos_validos == 2
        assert tri_c.graham_preco_justo is None
        assert tri_c.piso_bazin_8_juros_brasil == quant.valuation_bazin_conservador.dy_8.preco_justo


class TestLoopReflexaoGrafo:
    def test_retry_cirurgico_apenas_no_pdf_reprovado_e_ordenacao_virada_de_ano(
        self,
    ) -> None:
        chamadas: list[tuple[str, str | None]] = []

        def fake_extract(caminho_pdf: str, feedback_correcao=None, llm=None):
            chamadas.append((caminho_pdf, feedback_correcao))
            if caminho_pdf == "p1.pdf":
                return _fazer_trimestre("4T25", 6_810_000_000.0, 175_000_000_000.0, 0.165)
            if caminho_pdf == "p2.pdf":
                if feedback_correcao is None:
                    return _fazer_trimestre(
                        "1T26", 20_000_000_000.0, 180_000_000_000.0, 0.160
                    )
                assert "ROE" in feedback_correcao
                return _fazer_trimestre("1T26", 7_050_000_000.0, 180_000_000_000.0, 0.166)
            return _fazer_trimestre("2T26", 7_110_000_000.0, 182_282_051_282.0, 0.161)

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(
            content="Preço Justo R$ 18,56. Margem de segurança de 27,60%."
        )

        app = build_app(llm=fake_llm)
        with (
            patch("graph.nodes.extrair_dados_do_pdf", side_effect=fake_extract),
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            out = app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": ["p2.pdf", "p3.pdf", "p1.pdf"],
                    "tentativas_extracao": 0,
                    "usar_cache": False,
                }
            )

        assert out["tentativas_extracao"] == 2
        assert out["validacao_concluida"] is True
        assert out["erros_validacao"] == {}
        assert out["analise_llm_indisponivel"] is False
        # Chamada única no Analista (sem bind_tools nem loop ReAct!)
        assert fake_llm.invoke.call_count == 1
        assert not fake_llm.bind_tools.called
        # 3 PDFs na passagem 1 + APENAS p2.pdf na passagem 2 = 4 chamadas
        assert len(chamadas) == 4
        assert chamadas[3][0] == "p2.pdf" and chamadas[3][1] is not None
        assert (
            out["resultado_quantitativo"]["parametros"]["trimestre_mais_recente"]
            == "2T26"
        )
        texto_consolidado = out["messages"][0].content
        bloco_qualitativo = texto_consolidado.split(
            "CONTEXTO QUALITATIVO DOS TRIMESTRES"
        )[1]
        assert (
            bloco_qualitativo.index("4T25")
            < bloco_qualitativo.index("1T26")
            < bloco_qualitativo.index("2T26")
        )

    def test_para_exatamente_no_limite_maximo_de_tentativas(self) -> None:
        chamadas: list[str] = []

        def fake_extract(caminho_pdf: str, feedback_correcao=None, llm=None):
            chamadas.append(caminho_pdf)
            return _fazer_trimestre(
                "3T26",
                lucro=20_000_000_000.0,
                pl=180_000_000_000.0,
                roe=0.160,
                caixa_i=100.0,
                var_c=10.0,
                caixa_f=500.0,
            )

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(
            content="Relatório com ressalvas registradas."
        )

        app = build_app(llm=fake_llm)
        with (
            patch("graph.nodes.extrair_dados_do_pdf", side_effect=fake_extract),
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            out = app.invoke(
                {
                    "messages": [],
                    "setor": "energia",
                    "ticker": "EGIE3",
                    "caminhos_pdfs": ["ruim.pdf"],
                    "tentativas_extracao": 0,
                    "usar_cache": False,
                }
            )

        assert out["tentativas_extracao"] == MAX_TENTATIVAS_EXTRACAO
        assert len(chamadas) == MAX_TENTATIVAS_EXTRACAO
        assert "AVISO DE AUDITORIA" in out["messages"][0].content

    def test_pasta_sem_pdfs_encerra_diretamente_sem_chamar_analista(self) -> None:
        fake_llm = MagicMock()
        app = build_app(llm=fake_llm)
        out = app.invoke(
            {
                "messages": [],
                "setor": "geral",
                "ticker": "VAZIO3",
                "caminhos_pdfs": [],
                "tentativas_extracao": 0,
                "usar_cache": False,
            }
        )

        assert out["messages"] == []
        assert not fake_llm.invoke.called


class TestFallbackAnalista503:
    def test_erro_503_sem_modelo_fallback_gera_relatorio_quantitativo_sem_loop_compliance(
        self,
    ) -> None:
        llm_503 = MagicMock()
        llm_503.invoke.side_effect = RuntimeError(
            "503 UNAVAILABLE: This model is currently experiencing high demand."
        )

        app = build_app(llm=llm_503)
        with (
            patch("graph.nodes.extrair_dados_do_pdf", return_value=_fazer_trimestre()),
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            out = app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": ["3T26.pdf"],
                    "tentativas_extracao": 0,
                    "usar_cache": False,
                }
            )

        assert out["analise_llm_indisponivel"] is True
        assert "503" in str(out["erro_analista"])
        assert out["violacoes_compliance"] == []
        # Apenas 1 tentativa no LLM primário; nunca entra em loop pelo compliance!
        assert llm_503.invoke.call_count == 1
        relatorio = out["messages"][-1].content
        assert "MODO DE FALLBACK QUANTITATIVO" in relatorio
        assert "Triangulação de Valuation (BBDC3)" in relatorio
        assert "DDM de 2 Estágios" in relatorio
        assert "Décio Bazin Clássico" in relatorio
        assert "Décio Bazin Conservador / Juros BR" in relatorio
        assert "Benjamin Graham" in relatorio
        assert "Mediana da Triangulação (DDM, Bazin 6%, Graham)" in relatorio
        assert "um representante por metodologia" in relatorio
        assert "IR de 15% sobre JCP" in relatorio
        assert "Tabela Comparativa de Lucros e Múltiplos (BBDC3)" in relatorio
        assert "LPA Projetado — Forward (R$)" in relatorio
        assert "VPA Atual (R$)" in relatorio
        assert "P/L Projetado — Forward" in relatorio
        assert "P/VP Atual" in relatorio
        assert "Contexto Qualitativo Extraído dos PDFs" in relatorio
        assert "não constitui recomendação de compra, venda ou manutenção" in relatorio

    def test_fallback_503_com_4_trimestres_rotula_multiplos_como_anuais_realizados(
        self,
    ) -> None:
        llm_503 = MagicMock()
        llm_503.invoke.side_effect = RuntimeError("503 UNAVAILABLE")

        trimestres = {
            "1T26.pdf": _fazer_trimestre("1T26", lucro=6_800_000_000.0),
            "2T26.pdf": _fazer_trimestre("2T26", lucro=7_000_000_000.0),
            "3T26.pdf": _fazer_trimestre("3T26", lucro=7_100_000_000.0),
            "4T26.pdf": _fazer_trimestre("4T26", lucro=7_200_000_000.0),
        }

        def fake_extract(caminho_pdf: str, feedback_correcao=None, llm=None):
            return trimestres[caminho_pdf]

        app = build_app(llm=llm_503)
        with (
            patch("graph.nodes.extrair_dados_do_pdf", side_effect=fake_extract),
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            out = app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": list(trimestres.keys()),
                    "tentativas_extracao": 0,
                    "usar_cache": False,
                }
            )

        relatorio = out["messages"][-1].content
        assert "Lucro Líquido Anual (R$)" in relatorio
        assert "LPA Anual (R$)" in relatorio
        assert "P/L Anual" in relatorio
        assert "4T realizados" in relatorio
        assert "LPA Projetado — Forward" not in relatorio

    def test_fallback_503_exibe_na_com_motivo_quando_modelo_indisponivel(self) -> None:
        llm_503 = MagicMock()
        llm_503.invoke.side_effect = RuntimeError("503 UNAVAILABLE")

        app = build_app(llm=llm_503)
        dados_pl_neg = _fazer_trimestre(
            "3T26",
            lucro=500_000_000.0,
            pl=-2_000_000_000.0,
            roe=0.15,
            payout=0.50,
            num_acoes=1_000_000_000.0,
        )
        with (
            patch("graph.nodes.extrair_dados_do_pdf", return_value=dados_pl_neg),
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            out = app.invoke(
                {
                    "messages": [],
                    "setor": "varejo",
                    "ticker": "PLNEG3",
                    "caminhos_pdfs": ["3T26.pdf"],
                    "tentativas_extracao": 0,
                    "usar_cache": False,
                }
            )

        relatorio = out["messages"][-1].content
        assert "Triangulação de Valuation (PLNEG3)" in relatorio
        assert "N/A (" in relatorio
        assert "vpa deve ser estritamente positivo" in relatorio

    def test_erro_503_usa_modelo_fallback_quando_configurado(self) -> None:
        llm_primario = MagicMock()
        llm_primario.invoke.side_effect = RuntimeError("503 Service Unavailable")
        llm_reserva = MagicMock()
        llm_reserva.invoke.return_value = AIMessage(
            content="Relatório gerado pelo modelo reserva. Não constitui recomendação de compra ou venda."
        )

        app = build_app(llm=llm_primario, fallback_llm=llm_reserva)
        with (
            patch("graph.nodes.extrair_dados_do_pdf", return_value=_fazer_trimestre()),
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            out = app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": ["3T26.pdf"],
                    "tentativas_extracao": 0,
                    "usar_cache": False,
                }
            )

        assert llm_primario.invoke.call_count == 1
        assert llm_reserva.invoke.call_count == 1
        assert out["analise_llm_indisponivel"] is False
        assert out["erro_analista"] is None
        assert "modelo reserva" in out["messages"][-1].content

    def test_erro_503_duplo_primario_e_fallback_degrada_para_quantitativo(self) -> None:
        llm_primario = MagicMock()
        llm_primario.invoke.side_effect = RuntimeError("503 UNAVAILABLE")
        llm_reserva = MagicMock()
        llm_reserva.invoke.side_effect = TimeoutError("Connection timed out")

        app = build_app(llm=llm_primario, fallback_llm=llm_reserva)
        with (
            patch("graph.nodes.extrair_dados_do_pdf", return_value=_fazer_trimestre()),
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            out = app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": ["3T26.pdf"],
                    "tentativas_extracao": 0,
                    "usar_cache": False,
                }
            )

        assert out["analise_llm_indisponivel"] is True
        assert "Primário:" in str(out["erro_analista"])
        assert "Fallback:" in str(out["erro_analista"])
        assert "MODO DE FALLBACK QUANTITATIVO" in out["messages"][-1].content

    def test_roteador_compliance_nunca_retorna_para_analista_em_modo_fallback(
        self,
    ) -> None:
        estado = {
            "messages": [],
            "tentativas_extracao": 1,
            "analise_llm_indisponivel": True,
            "violacoes_compliance": ["rotulo_recomendacao: 'Recomendação: Compra'"],
            "tentativas_compliance": 0,
        }
        assert roteador_compliance(estado) == "fim"


class TestFiltragemECachePDF:
    def test_filtragem_inteligente_preserva_tabelas_finais_e_ordem_original(
        self,
    ) -> None:
        paginas = [
            "Destaques do Trimestre e Mensagem da Administracao " + ("a" * 80),
            "Texto institucional generico sem dados contabeis " + ("x" * 200),
            "Marketing e eventos institucionais " + ("y" * 200),
            (
                "Demonstracao do Resultado: Lucro Liquido Recorrente 7.110,00 ROE 16,1% "
                "Patrimonio Liquido 182.282,00 Capital Social Quantidade de Acoes 11.178,82"
            ),
        ]
        # Com limite menor que o total, deve escolher paginas 1 e 4, mantendo ordem 1 -> 4
        selecionado = selecionar_texto_relevante_pdf(paginas, limite_caracteres=380)
        assert "[Página 1]" in selecionado
        assert "[Página 4]" in selecionado
        assert "[Página 2]" not in selecionado
        assert selecionado.index("[Página 1]") < selecionado.index("[Página 4]")

    def test_cache_sha256_grava_apenas_validados_e_reutiliza_na_segunda_execucao(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dir_cache = tmp_path / "cache"
        monkeypatch.setenv("CACHE_EXTRATOR_DIR", str(dir_cache))
        monkeypatch.setenv("USAR_CACHE_PDF", "true")

        pdf_ok = tmp_path / "3T26.pdf"
        pdf_ok.write_bytes(b"%PDF-1.4 conteudo simulado valido 3T26")

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="Relatório limpo aprovado.")

        app = build_app(llm=fake_llm)
        with (
            patch("graph.nodes.extrair_dados_do_pdf", return_value=_fazer_trimestre()) as mock_ext,
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            # 1ª execução: extrai via LLM, valida e salva no cache
            app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": [str(pdf_ok)],
                    "tentativas_extracao": 0,
                    "usar_cache": True,
                }
            )
            assert mock_ext.call_count == 1
            assert len(list(dir_cache.glob("*.json"))) == 1

            # 2ª execução: cache HIT (zero novas chamadas ao extrator!)
            app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": [str(pdf_ok)],
                    "tentativas_extracao": 0,
                    "usar_cache": True,
                }
            )
            assert mock_ext.call_count == 1

            # 3ª execução com usar_cache=False (--no-cache): ignora cache e re-extrai
            app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": [str(pdf_ok)],
                    "tentativas_extracao": 0,
                    "usar_cache": False,
                }
            )
            assert mock_ext.call_count == 2

    def test_cache_nunca_grava_extracao_reprovada_pelo_validador(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dir_cache = tmp_path / "cache"
        monkeypatch.setenv("CACHE_EXTRATOR_DIR", str(dir_cache))

        pdf_bad = tmp_path / "bad.pdf"
        pdf_bad.write_bytes(b"%PDF-1.4 conteudo simulado invalido")

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="Relatório com ressalvas.")

        app = build_app(llm=fake_llm)
        dados_ruins = _fazer_trimestre(lucro=25_000_000_000.0, roe=0.16)
        with (
            patch("graph.nodes.extrair_dados_do_pdf", return_value=dados_ruins),
            patch("tools.utils.yf.Ticker", return_value=_mock_yfinance_bbdc3()),
        ):
            app.invoke(
                {
                    "messages": [],
                    "setor": "banco",
                    "ticker": "BBDC3",
                    "caminhos_pdfs": [str(pdf_bad)],
                    "tentativas_extracao": 0,
                    "usar_cache": True,
                }
            )

        assert not dir_cache.exists() or len(list(dir_cache.glob("*.json"))) == 0

    def test_cache_corrompido_degrada_para_cache_miss(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 teste")
        cache_dir = tmp_path / "cache"

        caminho_salvo = salvar_cache_extracao(pdf, _fazer_trimestre(), dir_cache=cache_dir)
        assert caminho_salvo is not None
        caminho_salvo.write_text("{json quebrado", encoding="utf-8")

        assert carregar_cache_extracao(pdf, dir_cache=cache_dir) is None
