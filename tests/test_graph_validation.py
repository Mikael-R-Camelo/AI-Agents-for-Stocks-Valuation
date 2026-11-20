from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agents.extrator import (
    DadosTrimestraisPDF,
    carregar_cache_extracao,
    salvar_cache_extracao,
    selecionar_texto_relevante_pdf,
)
from config import MAX_TENTATIVAS_EXTRACAO
from graph.nodes import validar_trimestre
from graph.workflow import build_app


def _fazer_trimestre(
    ref: str = "3T26",
    lucro: float = 7_110_000_000.0,
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
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = AIMessage(
            content="Preço Justo R$ 18,56. Margem de segurança de 27,60%."
        )

        app = build_app(llm=fake_llm)
        with patch("graph.nodes.extrair_dados_do_pdf", side_effect=fake_extract):
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
        # 3 PDFs na passagem 1 + APENAS p2.pdf na passagem 2 = 4 chamadas
        assert len(chamadas) == 4
        assert chamadas[3][0] == "p2.pdf" and chamadas[3][1] is not None
        texto_consolidado = out["messages"][0].content
        assert (
            texto_consolidado.index("4T25")
            < texto_consolidado.index("1T26")
            < texto_consolidado.index("2T26")
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
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = AIMessage(
            content="Relatório com ressalvas registradas."
        )

        app = build_app(llm=fake_llm)
        with patch("graph.nodes.extrair_dados_do_pdf", side_effect=fake_extract):
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
        fake_llm.bind_tools.return_value = fake_llm

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
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = AIMessage(content="Relatório limpo aprovado.")

        app = build_app(llm=fake_llm)
        with patch(
            "graph.nodes.extrair_dados_do_pdf", return_value=_fazer_trimestre()
        ) as mock_ext:
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
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.invoke.return_value = AIMessage(content="Relatório com ressalvas.")

        app = build_app(llm=fake_llm)
        dados_ruins = _fazer_trimestre(lucro=25_000_000_000.0, roe=0.16)
        with patch("graph.nodes.extrair_dados_do_pdf", return_value=dados_ruins):
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
