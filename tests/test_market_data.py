from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.market_data import (
    buscar_dados_mercado_yfinance,
    buscar_medias_historicas_yfinance,
    buscar_payout_medio_historico_yfinance,
    calcular_multiplos_basicos,
    projetar_lucro_anual_cenarios,
)
from tools.utils import limpar_cache_yfinance


class TestProjetarLucroAnualCenarios:
    @pytest.mark.parametrize(
        ("lucros", "momento", "esperado_cons", "esperado_med"),
        [
            ([100.0], "estavel", 400.0, 400.0),
            ([100.0, 120.0], "estavel", 420.0, 440.0),
            ([100.0, 110.0, 120.0], "Recuperação", 440.0, 450.0),
            ([100.0, 110.0, 120.0], "ESTÁVEL", 430.0, 440.0),
            ([120.0, 110.0, 100.0], "deteriorando", 420.0, 430.0),
            ([100.0, 100.0, 100.0, 100.0], "estavel", 380.0, 400.0),
        ],
    )
    def test_cenarios_1_a_4_trimestres_e_normalizacao(
        self,
        lucros: list[float],
        momento: str,
        esperado_cons: float,
        esperado_med: float,
    ) -> None:
        res = projetar_lucro_anual_cenarios.invoke(
            {
                "lucros_trimestrais_realizados": lucros,
                "momento_operacional": momento,
            }
        )
        assert res["lucro_anual_conservador"] == esperado_cons
        assert res["lucro_anual_mediano"] == esperado_med
        assert res["lucro_anual_conservador"] <= res["lucro_anual_mediano"]

    def test_contrato_chaves_identico_entre_4_e_menos_trimestres(self) -> None:
        res_4 = projetar_lucro_anual_cenarios.invoke(
            {"lucros_trimestrais_realizados": [100.0, 100.0, 100.0, 100.0]}
        )
        res_3 = projetar_lucro_anual_cenarios.invoke(
            {"lucros_trimestrais_realizados": [100.0, 100.0, 100.0]}
        )
        assert set(res_4.keys()) == set(res_3.keys())

    def test_estresse_conservador_correto_em_prejuizos(self) -> None:
        res_4 = projetar_lucro_anual_cenarios.invoke(
            {"lucros_trimestrais_realizados": [-100.0, -100.0, -100.0, -100.0]}
        )
        assert res_4["lucro_anual_mediano"] == -400.0
        assert res_4["lucro_anual_conservador"] == -420.0

        res_det = projetar_lucro_anual_cenarios.invoke(
            {
                "lucros_trimestrais_realizados": [-100.0, -120.0],
                "momento_operacional": "deteriorando",
            }
        )
        assert res_det["lucro_anual_mediano"] == -460.0
        assert res_det["lucro_anual_conservador"] == -484.0
        assert res_det["lucro_anual_conservador"] < res_det["lucro_anual_mediano"]

    def test_invariante_conservador_menor_igual_mediano_em_recuperacao_com_queda(
        self,
    ) -> None:
        res = projetar_lucro_anual_cenarios.invoke(
            {
                "lucros_trimestrais_realizados": [120.0, 110.0, 90.0],
                "momento_operacional": "recuperacao",
            }
        )
        assert res["lucro_anual_conservador"] <= res["lucro_anual_mediano"]

    def test_validacoes_entradas_invalidas(self) -> None:
        with pytest.raises(ValueError):
            projetar_lucro_anual_cenarios.invoke({"lucros_trimestrais_realizados": []})
        with pytest.raises(ValueError):
            projetar_lucro_anual_cenarios.invoke(
                {"lucros_trimestrais_realizados": [1, 2, 3, 4, 5]}
            )
        with pytest.raises(ValueError):
            projetar_lucro_anual_cenarios.invoke(
                {
                    "lucros_trimestrais_realizados": [100.0],
                    "momento_operacional": "otimista",
                }
            )


class TestCalcularMultiplosBasicos:
    def test_calculo_nominal(self) -> None:
        res = calcular_multiplos_basicos.invoke(
            {
                "preco_atual": 13.44,
                "lucro_anual": 27_960_000_000.0,
                "patrimonio_liquido": 182_282_051_282.0,
                "num_acoes": 11_178_825_623.0,
            }
        )
        assert res == {
            "LPA": 2.5,
            "VPA": 16.31,
            "P_L": 5.37,
            "P_VP": 0.82,
            "ROE_calculado": 0.1534,
        }

    def test_retorna_none_para_multiplos_nao_aplicaveis(self) -> None:
        res = calcular_multiplos_basicos.invoke(
            {
                "preco_atual": 10.0,
                "lucro_anual": -50.0,
                "patrimonio_liquido": -100.0,
                "num_acoes": 10.0,
            }
        )
        assert res["P_L"] is None
        assert res["P_VP"] is None
        assert res["ROE_calculado"] is None

    @pytest.mark.parametrize("campo,valor", [("num_acoes", 0.0), ("preco_atual", -1.0)])
    def test_rejeita_acoes_ou_preco_invalidos(self, campo: str, valor: float) -> None:
        params = {
            "preco_atual": 10.0,
            "lucro_anual": 100.0,
            "patrimonio_liquido": 500.0,
            "num_acoes": 50.0,
        }
        params[campo] = valor
        with pytest.raises(ValueError):
            calcular_multiplos_basicos.invoke(params)


class TestYFinanceTools:
    def test_dados_mercado_exclui_acoes_e_usa_fast_info(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "currentPrice": None,
            "regularMarketPrice": float("nan"),
            "marketCap": None,
            "sharesOutstanding": 5_000_000_000,
            "bookValue": 12.34,
            "currency": "BRL",
        }
        mock_ticker.fast_info = {"lastPrice": 13.44}

        with patch("tools.utils.yf.Ticker", return_value=mock_ticker):
            res = buscar_dados_mercado_yfinance.invoke({"ticker": "BBDC3"})

        assert res == {"ticker": "BBDC3", "preco_atual": 13.44, "moeda": "BRL"}
        assert "num_acoes_total" not in res
        assert "valor_patrimonial_por_acao_vpa" not in res

    def test_dados_mercado_trata_falha_rede(self) -> None:
        with patch("tools.utils.yf.Ticker", side_effect=RuntimeError("HTTP 429")):
            res = buscar_dados_mercado_yfinance.invoke({"ticker": "BBDC3"})
        assert "erro" in res and "HTTP 429" in res["erro"]

    def test_roe_historico_aliases_ifrs_e_limite_4_anos(self) -> None:
        cols = pd.to_datetime(
            ["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"]
        )
        mock_ticker = MagicMock()
        mock_ticker.financials = pd.DataFrame(
            [[100, 100, 100, 100, 100]],
            index=["Net Income Common Stockholders"],
            columns=cols,
        )
        mock_ticker.balance_sheet = pd.DataFrame(
            [[500, 500, 500, 500, 1000]],
            index=["Common Stock Equity"],
            columns=cols,
        )

        with patch("tools.utils.yf.Ticker", return_value=mock_ticker):
            res = buscar_medias_historicas_yfinance.invoke({"ticker": "ITUB4"})

        assert res["anos_analisados"] == 4
        assert res["roe_medio_historico"] == 0.2

    def test_roe_historico_trata_dataframes_vazios_ou_none(self) -> None:
        mock_ticker = MagicMock(financials=None, balance_sheet=pd.DataFrame())
        with patch("tools.utils.yf.Ticker", return_value=mock_ticker):
            res = buscar_medias_historicas_yfinance.invoke({"ticker": "VAZI3"})
        assert "erro" in res

    def test_payout_soma_on_pn_descarta_distorcao_e_limita_sustentavel(self) -> None:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31"])
        mock_ticker = MagicMock()
        mock_ticker.financials = pd.DataFrame(
            [[100, 100, 100]], index=["Net Income"], columns=cols
        )
        mock_ticker.cash_flow = pd.DataFrame(
            [[-70, -80, -180], [-50, -50, 0]],
            index=["Common Stock Dividend Paid", "Preferred Stock Dividend Paid"],
            columns=cols,
        )

        with patch("tools.utils.yf.Ticker", return_value=mock_ticker):
            res = buscar_payout_medio_historico_yfinance.invoke({"ticker": "EXTR4"})

        assert res["fonte_payout"] == "dfc_yfinance"
        assert res["anos_analisados"] == 2  # 2023 (180%) descartado (>150%)
        assert res["anos_ignorados_distorcao"] == 1
        assert res["payout_medio_observado"] == 1.25
        assert res["payout_sustentavel"] == 1.0
        assert res["payout_limitado_para_valuation"] is True

    def test_payout_fallback_jcp_quando_dfc_zerada(self) -> None:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31"])
        mock_ticker = MagicMock()
        mock_ticker.financials = pd.DataFrame(
            [[100, 100, 100]], index=["Net Income"], columns=cols
        )
        mock_ticker.cash_flow = pd.DataFrame(
            [[0.0, 0.0, 0.0]], index=["Cash Dividends Paid"], columns=cols
        )
        mock_ticker.info = {"payoutRatio": 0.4839}

        with patch("tools.utils.yf.Ticker", return_value=mock_ticker):
            res = buscar_payout_medio_historico_yfinance.invoke({"ticker": "BBDC3"})

        assert res["fonte_payout"] == "yfinance_info_payoutRatio_fallback"
        assert res["payout_medio_observado"] == 0.4839
        assert res["payout_sustentavel"] == 0.4839
        assert res["payout_limitado_para_valuation"] is False

    def test_payout_fallback_jcp_limita_extraordinario_e_preserva_zero_dividendo(
        self,
    ) -> None:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31"])
        dre = pd.DataFrame([[100, 100]], index=["Net Income"], columns=cols)
        fc_zero = pd.DataFrame(
            [[0.0, 0.0]], index=["Cash Dividends Paid"], columns=cols
        )

        t_extra = MagicMock(
            financials=dre, cash_flow=fc_zero, info={"payoutRatio": 1.30}
        )
        with patch("tools.utils.yf.Ticker", return_value=t_extra):
            res_extra = buscar_payout_medio_historico_yfinance.invoke(
                {"ticker": "EXJCP3"}
            )
        assert res_extra["payout_medio_observado"] == 1.3
        assert res_extra["payout_sustentavel"] == 1.0
        assert res_extra["payout_limitado_para_valuation"] is True

        t_zero = MagicMock(financials=dre, cash_flow=fc_zero, info={"payoutRatio": 0.0})
        with patch("tools.utils.yf.Ticker", return_value=t_zero):
            res_zero = buscar_payout_medio_historico_yfinance.invoke(
                {"ticker": "GROW3"}
            )
        assert res_zero["fonte_payout"] == "dfc_yfinance"
        assert res_zero["payout_sustentavel"] == 0.0

    def test_cache_em_memoria_reutiliza_mesma_instancia_ticker(self) -> None:
        cols = pd.to_datetime(["2025-12-31"])
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 15.0, "currency": "BRL"}
        mock_ticker.financials = pd.DataFrame(
            [[100]], index=["Net Income"], columns=cols
        )
        mock_ticker.balance_sheet = pd.DataFrame(
            [[500]], index=["Stockholders Equity"], columns=cols
        )
        mock_ticker.cash_flow = pd.DataFrame(
            [[-50]], index=["Cash Dividends Paid"], columns=cols
        )

        with patch("tools.utils.yf.Ticker", return_value=mock_ticker) as mock_ctor:
            buscar_dados_mercado_yfinance.invoke({"ticker": "BBDC3"})
            buscar_medias_historicas_yfinance.invoke({"ticker": "bbdc3.sa "})
            buscar_payout_medio_historico_yfinance.invoke({"ticker": "BBDC3.SA"})
            assert mock_ctor.call_count == 1

            limpar_cache_yfinance()
            buscar_dados_mercado_yfinance.invoke({"ticker": "BBDC3"})
            assert mock_ctor.call_count == 2
