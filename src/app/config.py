"""Configuracion central del sistema.

Este modulo define un unico objeto Settings que lee las variables del .env,
las convierte al tipo correcto y las valida. Es la unica fuente de verdad de
configuracion: el resto del codigo recibe este objeto en vez de leer os.getenv
por su cuenta. La Factory se apoya en estos valores para decidir que
implementaciones concretas construir.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Le indica a pydantic que lea el archivo .env y que ignore variables de
    # entorno extra que no esten declaradas aqui.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Gemini) ---
    gemini_api_key: str = Field(..., description="Clave de la API de Gemini")
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2

    # --- Embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-base"

    # --- Scraping ---
    scrape_base_url: str = "https://www.bbva.com.co/"
    scraper_type: str = "requests"
    max_depth: int = 2
    max_pages: int = 40
    scrape_delay: float = 1.0

    # --- Chunking ---
    chunk_size: int = 900
    chunk_overlap: int = 140

    # --- Recuperacion ---
    retrieval_top_k: int = 5
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Historial ---
    history_window: int = 6

    # --- Rutas de datos ---
    raw_data_dir: Path = Path("data/raw")
    clean_data_dir: Path = Path("data/clean")
    chroma_dir: Path = Path("data/chroma")
    sqlite_path: Path = Path("data/app.db")
    chroma_collection: str = "bbva_docs"

    @field_validator("scraper_type")
    @classmethod
    def _validar_scraper(cls, v: str) -> str:
        # Falla temprano si alguien pone un valor que la Factory no sabe construir.
        permitidos = {"requests", "playwright"}
        if v not in permitidos:
            raise ValueError(
                f"SCRAPER_TYPE debe ser uno de {permitidos}, se recibio '{v}'"
            )
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _validar_solape(cls, v: int, info) -> int:
        # El solape mayor o igual al tamano de chunk no tiene sentido y romperia
        # el troceado, asi que lo bloqueamos aqui.
        size = info.data.get("chunk_size", 900)
        if v >= size:
            raise ValueError(
                f"CHUNK_OVERLAP ({v}) debe ser menor que CHUNK_SIZE ({size})"
            )
        return v

    def asegurar_directorios(self) -> None:
        # Crea las carpetas de datos si no existen, para que los modulos que
        # escriben en ellas no fallen en el primer arranque.
        for d in (self.raw_data_dir, self.clean_data_dir, self.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    # lru_cache garantiza que Settings se construya una sola vez y se reutilice
    # esa misma instancia en todo el proceso (un unico panel de control).
    return Settings()