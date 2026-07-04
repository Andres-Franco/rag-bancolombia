"""Estrategias de scraping (patron Strategy).

Define un contrato comun (la clase abstracta Scraper) y sus implementaciones
concretas. El resto del sistema depende solo del contrato: pide 'fetch(url)' y
recibe el HTML, sin saber ni importarle que estrategia esta detras.

- RequestsScraper: ligero, usa requests. No ejecuta JavaScript. Es el defecto.
- PlaywrightScraper: renderiza JS con un navegador headless. Solo si hace falta.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FetchResult:
    # Resultado de descargar una pagina. Agrupamos la URL final (por si hubo
    # redirecciones), el HTML crudo y el codigo de estado, para que las etapas
    # siguientes (limpieza, descubrimiento de enlaces) tengan todo lo que necesitan.
    url: str
    html: str
    status_code: int
    ok: bool


class Scraper(ABC):
    """Contrato comun de todas las estrategias de scraping.

    Cualquier estrategia concreta DEBE implementar 'fetch'. Al ser abstracta,
    esta clase no se puede instanciar directamente: solo sirve de plantilla.
    """

    @abstractmethod
    def fetch(self, url: str) -> FetchResult:
        # Descarga una unica URL y devuelve su contenido. El 'como' lo define
        # cada estrategia; el 'que' (recibir una URL, devolver un FetchResult)
        # es el contrato que todas comparten.
        ...

    def close(self) -> None:
        # Metodo opcional para liberar recursos (por ejemplo, cerrar el
        # navegador de Playwright). No es abstracto porque no todas las
        # estrategias lo necesitan; por defecto no hace nada.
        return None


class RequestsScraper(Scraper):
    """Estrategia ligera basada en requests.

    Adecuada cuando el contenido informativo viene en el HTML inicial, sin
    depender de JavaScript. Es la opcion por defecto por ser rapida y no
    arrastrar un navegador completo.
    """

    def __init__(self, timeout: float = 20.0, user_agent: str | None = None) -> None:
        # Importamos requests aqui dentro (no arriba del archivo) para que quien
        # use solo la estrategia de Playwright no cargue requests sin necesidad.
        # Es una optimizacion menor, pero mantiene las dependencias acopladas a
        # la estrategia que de verdad las usa.
        import requests

        self._session = requests.Session()
        self._timeout = timeout
        # Un User-Agent de navegador real reduce que el sitio nos bloquee o
        # devuelva una version distinta pensada para bots.
        self._session.headers.update(
            {
                "User-Agent": user_agent
                or (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            }
        )

    def fetch(self, url: str) -> FetchResult:
        # Implementacion concreta del contrato para la estrategia requests.
        try:
            resp = self._session.get(url, timeout=self._timeout)
            return FetchResult(
                url=str(resp.url),
                html=resp.text,
                status_code=resp.status_code,
                ok=resp.ok,
            )
        except Exception:
            # Ante cualquier fallo de red devolvemos un resultado 'no ok' en vez
            # de dejar que la excepcion rompa todo el crawl. El crawler decide
            # que hacer con una pagina fallida (saltarla). Esto es manejo de
            # errores: una pagina rota no debe tumbar la ingesta completa.
            return FetchResult(url=url, html="", status_code=0, ok=False)


class PlaywrightScraper(Scraper):
    """Estrategia que renderiza JavaScript con un navegador headless real.

    Adecuada cuando el sitio bloquea peticiones simples (403 anti-bot) o carga
    su contenido con JavaScript. Un navegador de verdad envia headers y huella
    TLS autenticos, y ejecuta el JS, con lo que sortea ambos problemas.

    Requiere el extra opcional: uv sync --extra playwright, y luego descargar el
    navegador con: uv run playwright install chromium.

    El navegador se lanza UNA vez al crear el scraper y se reutiliza en todas las
    paginas (lanzar un navegador por pagina seria muy costoso). Al terminar hay
    que llamar a close() para liberar el navegador; por eso esta estrategia si
    implementa el metodo close() que la clase base dejaba opcional.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str | None = None,
        wait_until: str = "networkidle",
    ) -> None:
        # Importamos aqui dentro para que el proyecto funcione sin Playwright
        # instalado mientras se use la estrategia de requests. Si falta el
        # paquete, damos un mensaje claro de que instalar.
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "Playwright no esta instalado"
            ) from exc

        self._timeout_ms = int(timeout * 1000)  # Playwright usa milisegundos.
        self._wait_until = wait_until

        # Arrancamos Playwright y lanzamos Chromium una sola vez.
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        # El 'context' es como una sesion de navegador: le damos un User-Agent
        # realista y el idioma, para parecer un usuario de Colombia.
        self._context = self._browser.new_context(
            user_agent=user_agent
            or (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            locale="es-CO",
        )

    def fetch(self, url: str) -> FetchResult:
        try:
            page = self._context.new_page()
            # wait_until='networkidle' espera a que el JavaScript termine de
            # cargar el contenido antes de leer el HTML.
            respuesta = page.goto(
                url, timeout=self._timeout_ms, wait_until=self._wait_until
            )
            html = page.content()  # HTML ya renderizado, con el JS ejecutado.
            estado = respuesta.status if respuesta is not None else 0
            url_final = page.url
            page.close()
            return FetchResult(
                url=url_final,
                html=html,
                status_code=estado,
                ok=200 <= estado < 400,
            )
        except Exception:
            # Mismo criterio que la otra estrategia: una pagina fallida no debe
            # romper el crawl; se reporta con ok=False.
            return FetchResult(url=url, html="", status_code=0, ok=False)

    def close(self) -> None:
        # Liberamos el navegador y Playwright. Importante llamarlo al terminar
        # para no dejar procesos de Chromium colgados.
        try:
            self._context.close()
            self._browser.close()
            self._playwright.stop()
        except Exception:
            pass