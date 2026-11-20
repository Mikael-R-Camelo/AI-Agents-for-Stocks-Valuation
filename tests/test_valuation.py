import math

import pytest

from config import CRESCIMENTO_PERPETUO_PADRAO, TAXA_DESCONTO_PADRAO
from tools.valuation import fluxo_de_caixa_futuro_buffet


def test_constantes_oficiais_valuation() -> None:
    assert TAXA_DESCONTO_PADRAO == 0.145
    assert CRESCIMENTO_PERPETUO_PADRAO == 0.04


def test_schema_obrigatorio_da_tool_valuation() -> None:
    schema = fluxo_de_caixa_futuro_buffet.args_schema.model_json_schema()
    required = set(schema.get("required", []))
    assert {"lucro_base", "payout", "roe", "num_acoes", "preco_atual"}.issubset(
        required
    )


def test_ddm_dois_estagios_equivalencia_analitica_bbdc3() -> None:
    """Valida DDM de 2 estágios (Ke=14,5%, g=4,0%) contra forma fechada TV_N = E_{N+1}/Ke."""
    lucro_mediano = 27_960_000_000.0
    payout = 0.50
    roe = 0.161
    num_acoes = 11_178_825_623.0
    preco_atual = 13.44
    ke = TAXA_DESCONTO_PADRAO
    g_perp = CRESCIMENTO_PERPETUO_PADRAO

    res = fluxo_de_caixa_futuro_buffet.invoke(
        {
            "lucro_base": lucro_mediano,
            "payout": payout,
            "roe": roe,
            "num_acoes": num_acoes,
            "preco_atual": preco_atual,
        }
    )

    g_sust = (1.0 - payout) * roe
    pv_divs = sum(
        (lucro_mediano * ((1.0 + g_sust) ** t) * payout) / ((1.0 + ke) ** t)
        for t in range(1, 5)
    )
    e_n = lucro_mediano * ((1.0 + g_sust) ** 4)
    tv_fechado = (e_n * (1.0 + g_perp)) / ke
    pv_tv = tv_fechado / ((1.0 + ke) ** 4)
    preco_esperado = round((pv_divs + pv_tv) / num_acoes, 2)
    payout_terminal_esperado = round((1.0 - g_perp / ke) * 100, 2)

    assert res["Taxa de Desconto Ke (%)"] == 14.5
    assert res["Crescimento Perpétuo (%)"] == 4.0
    assert res["Crescimento Sustentável (%)"] == 8.05
    assert res["Payout Terminal Sustentável (%)"] == payout_terminal_esperado == 72.41
    assert res["Preço Teto / Justo (R$)"] == preco_esperado == 18.56
    assert res["Margem de Segurança (%)"] == pytest.approx(27.6, abs=0.05)
    assert res["Upside Real (%)"] == pytest.approx(38.11, abs=0.05)


def test_ddm_cenario_conservador_bbdc3_com_fallback_jcp() -> None:
    """Valida Cenário Conservador de BBDC3 com ROE histórico 11,35% e payout 48,39%."""
    res = fluxo_de_caixa_futuro_buffet.invoke(
        {
            "lucro_base": 27_781_000_000.0,
            "payout": 0.4839,
            "roe": 0.1135,
            "num_acoes": 11_178_825_623.0,
            "preco_atual": 13.44,
        }
    )
    assert res["Preço Teto / Justo (R$)"] == 16.99
    assert res["Margem de Segurança (%)"] == 20.9
    assert res["Upside Real (%)"] == 26.42


@pytest.mark.parametrize(
    "overrides",
    [
        {"taxa_desconto": 0.04, "crescimento_perpetuo": 0.04},
        {"taxa_desconto": 0.03, "crescimento_perpetuo": 0.05},
        {"crescimento_perpetuo": -0.01},
        {"crescimento_perpetuo": 1.0},
        {"taxa_desconto": 0.0},
        {"taxa_desconto": 1.0},
        {"payout": 0.0},
        {"payout": -0.2},
        {"payout": 1.01},
        {"roe": 16.1},
        {"roe": -2.5},
        {"lucro_base": 0.0},
        {"lucro_base": -500.0},
        {"num_acoes": 0.0},
        {"num_acoes": -100.0},
        {"preco_atual": 0.0},
        {"preco_atual": -10.0},
        {"anos_projecao": 0},
        {"lucro_base": math.nan},
        {"preco_atual": math.inf},
    ],
)
def test_ddm_guardas_de_dominio(overrides: dict) -> None:
    base = {
        "lucro_base": 1_000_000_000.0,
        "payout": 0.50,
        "roe": 0.15,
        "num_acoes": 100_000_000.0,
        "preco_atual": 50.0,
    }
    base.update(overrides)
    with pytest.raises(ValueError):
        fluxo_de_caixa_futuro_buffet.invoke(base)


def test_ddm_rejeita_booleano_como_numero() -> None:
    with pytest.raises(ValueError):
        fluxo_de_caixa_futuro_buffet.func(
            lucro_base=True,  # type: ignore[arg-type]
            payout=0.5,
            roe=0.15,
            num_acoes=100.0,
            preco_atual=10.0,
        )
