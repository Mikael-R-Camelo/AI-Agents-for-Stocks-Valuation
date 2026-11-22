import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

BASE_DIR = Path(__file__).resolve().parent

# Modelo LLM
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# Parâmetros oficiais de Valuation (DDM de 2 Estágios)
TAXA_DESCONTO_PADRAO = 0.145
CRESCIMENTO_PERPETUO_PADRAO = 0.04

# Parâmetros oficiais de Triangulação (Décio Bazin e Benjamin Graham)
YIELD_MINIMO_BAZIN_PADRAO = 0.06
YIELD_MINIMO_BAZIN_CONSERVADOR = 0.08
CONSTANTE_GRAHAM_PADRAO = 22.5

# Limites dos loops de reflexão (Extração e Compliance)
MAX_TENTATIVAS_EXTRACAO = 3
MAX_TENTATIVAS_COMPLIANCE = 2
TOLERANCIA_ROE = 0.025
TOLERANCIA_CAIXA = 0.02

# Processamento e cache de PDFs
LIMITE_CARACTERES_DOCUMENTO = 150_000
CACHE_EXTRATOR_DIR = BASE_DIR / ".cache" / "extrator"
VERSAO_SCHEMA_EXTRATOR = "v2"

_VALORES_FALSOS = {"0", "false", "no", "nao", "não", "off"}


def carregar_ambiente() -> None:
    """Carrega variáveis do arquivo .env sem efeitos colaterais destrutivos."""
    load_dotenv()


def garantir_api_key() -> str:
    """Garante que GOOGLE_API_KEY está configurada antes de acionar a API do Gemini."""
    carregar_ambiente()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY não foi encontrada. "
            "Defina essa variável no arquivo .env antes de executar o programa."
        )
    return api_key


def obter_modelo_gemini() -> str:
    """Retorna o modelo Gemini configurado via GEMINI_MODEL (padrão funcional mantido)."""
    modelo = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    return modelo or DEFAULT_GEMINI_MODEL


def obter_modelo_gemini_fallback() -> str | None:
    """
    Retorna o modelo Gemini reserva configurado via GEMINI_FALLBACK_MODEL.
    Retorna None se não configurado ou se for idêntico ao modelo primário.
    """
    fallback = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
    if not fallback or fallback == obter_modelo_gemini():
        return None
    return fallback


def cache_pdf_habilitado() -> bool:
    """Indica se o cache local SHA-256 de extrações de PDF está habilitado."""
    valor = os.getenv("USAR_CACHE_PDF", "true").strip().lower()
    return valor not in _VALORES_FALSOS


def obter_dir_cache_extrator() -> Path:
    """Retorna o diretório de cache de extrações (configurável via CACHE_EXTRATOR_DIR)."""
    caminho_env = os.getenv("CACHE_EXTRATOR_DIR", "").strip()
    return Path(caminho_env) if caminho_env else CACHE_EXTRATOR_DIR


def configurar_logging(nivel: int | str = logging.INFO) -> None:
    """Configura logging estruturado limpo para execução em terminal."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(nivel)
        return

    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@lru_cache(maxsize=4)
def obter_llm(modelo: str | None = None) -> ChatGoogleGenerativeAI:
    """Factory cacheada do cliente Gemini (requer GOOGLE_API_KEY apenas quando chamada)."""
    garantir_api_key()
    return ChatGoogleGenerativeAI(
        model=modelo or obter_modelo_gemini(),
        temperature=0,
        max_retries=3,
    )
