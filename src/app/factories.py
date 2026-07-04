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