"""Crawler: recorre el sitio a partir de una semilla usando un Scraper.

El scraper sabe traer UNA pagina; el crawler decide POR CUALES paginas navegar.
Implementa un recorrido a lo ancho (BFS) con varios frenos configurables:

- No repetir paginas ya visitadas (evita bucles).
- No exceder una profundidad maxima (MAX_DEPTH).
- No superar un tope de paginas (MAX_PAGES).
- Quedarse dentro del dominio de la semilla.
- Pausar entre peticiones (cortesia con el servidor).
"""

from collections import deque
from dataclasses import dataclass
from time import sleep
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup

from app.ingest.scraper import Scraper


@dataclass
class Page:
    # Una pagina descargada con exito: su URL, el HTML crudo y a que
    # profundidad se encontro. El HTML se guarda tal cual; la limpieza es una
    # etapa posterior.
    url: str
    html: str
    depth: int


def _mismo_dominio(url: str, dominio_base: str) -> bool:
    # Devuelve True si la URL pertenece al mismo dominio que la semilla.
    # Comparamos el netloc (p. ej. "www.bbva.com.co") ignorando mayusculas.
    return urlparse(url).netloc.lower() == dominio_base.lower()


def _extraer_enlaces(html: str, url_actual: str, dominio_base: str) -> list[str]:
    """Saca los enlaces internos navegables de una pagina.

    Hace tres cosas con cada enlace encontrado:
    1. Lo convierte a absoluto (los href suelen ser relativos, tipo '/tarjetas').
    2. Le quita el fragmento (#seccion), que apunta a la misma pagina.
    3. Lo descarta si no es http(s) o si sale del dominio.
    """
    soup = BeautifulSoup(html, "lxml")
    encontrados: list[str] = []
    vistos_local: set[str] = set()

    for etiqueta in soup.find_all("a", href=True):
        href = etiqueta["href"].strip()
        # Ignoramos anclas vacias, mailto, tel y javascript.
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        # urljoin resuelve un href relativo contra la URL actual: si estamos en
        # https://www.bbva.com.co/personas/ y el href es '/tarjetas', produce
        # https://www.bbva.com.co/tarjetas
        absoluto = urljoin(url_actual, href)
        # urldefrag separa el fragmento: '.../pagina#seccion' -> '.../pagina'
        absoluto, _ = urldefrag(absoluto)

        # Solo http/https y solo el mismo dominio.
        esquema = urlparse(absoluto).scheme
        if esquema not in ("http", "https"):
            continue
        if not _mismo_dominio(absoluto, dominio_base):
            continue

        # Evitamos duplicados dentro de la misma pagina.
        if absoluto not in vistos_local:
            vistos_local.add(absoluto)
            encontrados.append(absoluto)

    return encontrados


class Crawler:
    """Recorre un sitio a lo ancho respetando los frenos configurados."""

    def __init__(
        self,
        scraper: Scraper,
        max_depth: int,
        max_pages: int,
        delay: float,
    ) -> None:
        # El crawler recibe el scraper ya construido (por la Factory). No sabe
        # ni le importa si es requests o playwright: solo llama a fetch().
        self._scraper = scraper
        self._max_depth = max_depth
        self._max_pages = max_pages
        self._delay = delay

    def crawl(self, url_semilla: str) -> list[Page]:
        dominio_base = urlparse(url_semilla).netloc

        # La cola guarda pares (url, profundidad). deque permite sacar por la
        # izquierda de forma eficiente, que es lo que hace el recorrido "a lo
        # ancho": primero se procesan las URLs mas cercanas a la semilla.
        cola: deque[tuple[str, int]] = deque([(url_semilla, 0)])
        visitadas: set[str] = set()
        paginas: list[Page] = []

        while cola and len(paginas) < self._max_pages:
            url, profundidad = cola.popleft()

            # Freno 1: no repetir.
            if url in visitadas:
                continue
            # Freno 2: no exceder profundidad.
            if profundidad > self._max_depth:
                continue

            visitadas.add(url)

            # Descarga usando la estrategia de scraping (Strategy en accion).
            resultado = self._scraper.fetch(url)
            if not resultado.ok or not resultado.html:
                # Manejo de errores: una pagina fallida se salta sin romper el
                # crawl. El scraper ya devolvio ok=False en vez de lanzar.
                continue

            paginas.append(
                Page(url=resultado.url, html=resultado.html, depth=profundidad)
            )

            # Solo seguimos extrayendo enlaces si aun no llegamos al fondo.
            if profundidad < self._max_depth:
                for enlace in _extraer_enlaces(
                    resultado.html, resultado.url, dominio_base
                ):
                    if enlace not in visitadas:
                        cola.append((enlace, profundidad + 1))

            # Cortesia: pausa entre peticiones para no saturar el servidor.
            if self._delay > 0:
                sleep(self._delay)

        return paginas