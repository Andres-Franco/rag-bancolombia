"""Limpieza y almacenamiento de las paginas scrapeadas.

Cumple el requisito de guardar los datos "crudos y limpios":
- El HTML crudo se guarda tal cual en data/raw (materia prima reutilizable).
- El texto limpio, con sus metadatos, se guarda como JSON en data/clean.

La limpieza quita del HTML todo lo que no es contenido legible (scripts,
estilos, menus de navegacion, pies de pagina) y normaliza el texto resultante.
"""

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup

from app.ingest.crawler import Page


# Etiquetas que NO son contenido: codigo y estructura repetida entre paginas.
_ETIQUETAS_RUIDO = ["script", "style", "noscript", "nav", "header", "footer", "aside"]


@dataclass
class CleanDoc:
    # Una pagina ya limpia, lista para trocear y vectorizar. Guardamos la url
    # para poder citar la fuente, el titulo como contexto, y el texto limpio.
    url: str
    title: str
    text: str


def _id_desde_url(url: str) -> str:
    # Convierte una URL en un identificador corto y seguro para nombre de
    # archivo. Usamos un hash para evitar caracteres invalidos (/, ?, &) y que
    # dos URLs distintas nunca colisionen en el mismo nombre.
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _normalizar_texto(texto: str) -> str:
    # Colapsa espacios y saltos de linea multiples en uno solo, y recorta. Deja
    # el texto compacto y legible en vez de un amasijo con huecos.
    # \s+ = cualquier secuencia de espacios, tabs o saltos de linea.
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n\s*\n+", "\n\n", texto)
    lineas = [linea.strip() for linea in texto.splitlines()]
    return "\n".join(l for l in lineas if l).strip()


# --- Filtrado de ruido especifico de sitios con plantillas JavaScript ---
# Estos sitios (como Bancolombia) dejan en el HTML restos de su motor de
# plantillas y nombres de iconos que no son contenido. Los filtramos por linea.

# Lineas que son EXACTAMENTE una de estas cadenas tecnicas se descartan.
_LINEAS_EXACTAS_RUIDO = {
    "{}",
    "error",
    "deferred modules",
    "${title}",
    "${badge}",
    "${loading}",
    "ver video",
    "conocer",
    "conoce mas",
    "conoce más",
    "ver mas",
    "ver más",
    "ir ahora",
    "conócelas",
    "suscríbete aquí",
    "términos y condiciones",
    "ver tyc",
}

# Si una linea CONTIENE alguno de estos patrones, se descarta.
_PATRONES_RUIDO = re.compile(
    r"(\$\{.*?\}|lorem ipsum|^arrow[\w-]*$|^icon-[\w-]*$|^arrow2-[\w-]*$)",
    re.IGNORECASE,
)


def _filtrar_ruido(texto: str) -> str:
    """Quita lineas que son ruido de plantilla, no contenido real.

    Se aplica linea a linea de forma conservadora: solo elimina lineas que
    coinciden con patrones de ruido conocidos (iconos, placeholders de
    plantilla, textos de relleno, botones repetidos). No toca lineas de
    contenido aunque sean cortas, salvo que sean exactamente ruido conocido.
    """
    lineas_utiles = []
    for linea in texto.splitlines():
        limpia = linea.strip()
        if not limpia:
            continue
        # Descarta coincidencias exactas con ruido conocido (sin distinguir
        # mayusculas).
        if limpia.lower() in _LINEAS_EXACTAS_RUIDO:
            continue
        # Descarta lineas que contienen patrones de ruido (iconos, ${...}, etc.)
        if _PATRONES_RUIDO.search(limpia):
            continue
        lineas_utiles.append(limpia)
    return "\n".join(lineas_utiles)


def limpiar(pagina: Page) -> CleanDoc:
    """Transforma el HTML crudo de una pagina en texto limpio con metadatos."""
    soup = BeautifulSoup(pagina.html, "lxml")

    # 1. El titulo, si existe, antes de borrar nada.
    title = soup.title.get_text(strip=True) if soup.title else ""

    # 2. Eliminar del arbol todas las etiquetas de ruido.
    for nombre in _ETIQUETAS_RUIDO:
        for etiqueta in soup.find_all(nombre):
            etiqueta.decompose()  # decompose() borra la etiqueta y su contenido.

    # 3. Extraer el texto restante y normalizarlo.
    crudo = soup.get_text(separator="\n")
    texto = _normalizar_texto(crudo)

    # 4. Filtrar el ruido de plantillas/iconos que dejan los sitios con JS.
    texto = _filtrar_ruido(texto)

    return CleanDoc(url=pagina.url, title=title, text=texto)


class Almacen:
    """Guarda en disco las paginas crudas y limpias."""

    def __init__(self, raw_dir: Path, clean_dir: Path) -> None:
        self._raw_dir = raw_dir
        self._clean_dir = clean_dir
        # Nos aseguramos de que las carpetas existan antes de escribir.
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._clean_dir.mkdir(parents=True, exist_ok=True)

    def guardar_crudo(self, pagina: Page) -> Path:
        # El HTML crudo se guarda con extension .html, nombrado por el hash de
        # la URL. Se conserva intacto para poder re-limpiar sin re-scrapear.
        ruta = self._raw_dir / f"{_id_desde_url(pagina.url)}.html"
        ruta.write_text(pagina.html, encoding="utf-8")
        return ruta

    def guardar_limpio(self, doc: CleanDoc) -> Path:
        # El documento limpio se guarda como JSON para conservar juntos la url,
        # el titulo y el texto. asdict convierte el dataclass en diccionario.
        ruta = self._clean_dir / f"{_id_desde_url(doc.url)}.json"
        ruta.write_text(
            json.dumps(asdict(doc), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ruta

    def cargar_limpios(self) -> list[CleanDoc]:
        # Lee de vuelta todos los documentos limpios guardados. Lo usara la
        # etapa de chunking, que parte del texto ya limpio en disco.
        docs: list[CleanDoc] = []
        for ruta in sorted(self._clean_dir.glob("*.json")):
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            docs.append(CleanDoc(**datos))
        return docs