import pytest

from config import obter_llm
from tools.utils import limpar_cache_yfinance


@pytest.fixture(autouse=True)
def isolar_caches_e_ambiente(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garante isolamento determinístico de caches em memória e desativa cache de PDF em disco por padrão."""
    monkeypatch.setenv("USAR_CACHE_PDF", "false")
    limpar_cache_yfinance()
    obter_llm.cache_clear()
    yield
    limpar_cache_yfinance()
    obter_llm.cache_clear()
