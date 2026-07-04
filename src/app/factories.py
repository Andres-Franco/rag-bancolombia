"""Fabricas de componentes (patron Factory).

Centraliza la construccion de los componentes intercambiables del sistema. Cada
funcion fabrica recibe la configuracion y devuelve la implementacion concreta ya
lista para usar. Es el unico lugar donde se decide "que" implementacion se
instancia; el resto del codigo solo pide el componente y lo usa.

Factory y Strategy trabajan juntos: las estrategias (p. ej. los scrapers) viven
en sus propios modulos; aqui se decide cual de ellas construir segun la config.
"""

from app.config import Settings
from app.ingest.scraper import PlaywrightScraper, RequestsScraper, Scraper


def crear_scraper(settings: Settings) -> Scraper:
    """Construye la estrategia de scraping indicada por la configuracion.

    Lee settings.scraper_type y devuelve la instancia concreta. El tipo de
    retorno anotado es Scraper (el contrato), no la clase concreta: quien llama
    recibe "un scraper" y no necesita saber cual, que es la esencia del Strategy.
    """
    tipo = settings.scraper_type

    if tipo == "requests":
        return RequestsScraper()

    if tipo == "playwright":
        return PlaywrightScraper()

    # Red de seguridad. En la practica config.py ya valida que scraper_type sea
    # uno de los permitidos, asi que este punto no deberia alcanzarse nunca. Lo
    # dejamos por robustez: si alguien anade una estrategia al validador pero
    # olvida registrarla aqui, el error es explicito en vez de silencioso.
    raise ValueError(
        f"No hay una estrategia de scraping registrada para '{tipo}'. "
        f"Revisa crear_scraper en factories.py."
    )


def crear_embedder(settings: Settings):
    # Importamos aqui dentro para no cargar sentence-transformers (y torch, que
    # es pesado) salvo cuando de verdad se necesita construir el embedder.
    from app.rag.embeddings import Embedder

    return Embedder(modelo=settings.embedding_model)


def crear_vectorstore(settings: Settings):
    # Importamos aqui dentro para no cargar chromadb salvo cuando se necesita.
    from app.rag.vectorstore import VectorStore

    return VectorStore(ruta=settings.chroma_dir, coleccion=settings.chroma_collection)


def crear_llm(settings: Settings):
    # Importamos aqui dentro para no cargar el cliente de Gemini si no se usa.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.gemini_api_key,
        max_retries=2,  # reintentos automaticos ante fallos transitorios
    )