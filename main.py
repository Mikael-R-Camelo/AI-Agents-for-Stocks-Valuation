import argparse
import logging
import sys
from pathlib import Path

from agents.compliance import extrair_texto_mensagem
from config import (
    BASE_DIR,
    cache_pdf_habilitado,
    configurar_logging,
    garantir_api_key,
)
from graph.workflow import build_app
from tools.utils import identificar_setor_pelo_ticker, limpar_cache_yfinance

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline multiagente de análise fundamentalista e valuation da B3."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignora e desativa o cache local SHA-256 de extrações de PDFs nesta execução.",
    )
    return parser.parse_args(argv)


def processar_relatorios(
    pasta_reports: Path | None = None,
    usar_cache: bool | None = None,
) -> int:
    """Executa o pipeline em lote para todas as subpastas de ativos em `reports/`."""
    garantir_api_key()
    limpar_cache_yfinance()

    diretorio_reports = pasta_reports or (BASE_DIR / "reports")
    if not diretorio_reports.exists():
        diretorio_reports.mkdir(parents=True)
        logger.info(
            "Pasta '%s' criada! Adicione subpastas (ex: reports/BBDC3/) com os PDFs.",
            diretorio_reports,
        )
        return 0

    cache_ativo = cache_pdf_habilitado() if usar_cache is None else usar_cache
    app = build_app()

    logger.info(
        "Iniciando varredura de relatórios trimestrais em '%s' (cache PDF: %s)...",
        diretorio_reports,
        "ativo" if cache_ativo else "desativado",
    )

    houve_falha = False

    for pasta_ticker in sorted(diretorio_reports.iterdir()):
        if not pasta_ticker.is_dir():
            continue

        ticker_atual = pasta_ticker.name
        arquivos_pdf = sorted(str(p) for p in pasta_ticker.glob("*.pdf"))
        if not arquivos_pdf:
            continue

        setor_dinamico = identificar_setor_pelo_ticker(ticker_atual)
        logger.info("%s", "=" * 60)
        logger.info(
            "Ativo: %s | Setor: %s | %d PDF(s) encontrado(s)",
            ticker_atual,
            setor_dinamico,
            len(arquivos_pdf),
        )
        logger.info("%s", "=" * 60)

        try:
            resultado_final = app.invoke(
                {
                    "messages": [],
                    "setor": setor_dinamico,
                    "ticker": ticker_atual,
                    "caminhos_pdfs": arquivos_pdf,
                    "tentativas_extracao": 0,
                    "dados_extraidos": {},
                    "erros_validacao": {},
                    "validacao_concluida": False,
                    "resultado_quantitativo": {},
                    "analise_llm_indisponivel": False,
                    "erro_analista": None,
                    "tentativas_compliance": 0,
                    "violacoes_compliance": [],
                    "usar_cache": cache_ativo,
                    "origem_cache": {},
                }
            )

            mensagens_finais = resultado_final.get("messages") or []
            if not mensagens_finais:
                logger.warning(
                    "Nenhuma resposta gerada pelo pipeline para %s.", ticker_atual
                )
                continue

            texto_relatorio = extrair_texto_mensagem(mensagens_finais[-1].content)
            caminho_md = pasta_ticker / f"Valuation_{ticker_atual}.md"
            caminho_md.write_text(texto_relatorio, encoding="utf-8")

            if resultado_final.get("analise_llm_indisponivel"):
                logger.warning(
                    "[FALLBACK] Relatório quantitativo salvo em %s (LLM indisponível: %s)",
                    caminho_md,
                    resultado_final.get("erro_analista"),
                )
            else:
                logger.info("[SUCESSO] Relatório salvo em: %s", caminho_md)

        except Exception:
            houve_falha = True
            logger.exception("[ERRO] Falha ao processar %s", ticker_atual)

    return 1 if houve_falha else 0


def main(argv: list[str] | None = None) -> int:
    configurar_logging()
    args = parse_args(argv)
    usar_cache = False if args.no_cache else None
    return processar_relatorios(usar_cache=usar_cache)


if __name__ == "__main__":
    sys.exit(main())
