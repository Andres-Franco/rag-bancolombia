"""Orquestador de la ingesta: ejecuta el pipeline completo de principio a fin.

Flujo:
  1. Lee la configuracion del .env.
  2. La Factory construye la estrategia de scraping indicada (requests/playwright).
  3. El crawler recorre el sitio desde la URL semilla, respetando los limites.
  4. Cada pagina cruda se guarda en data/raw.
  5. Cada pagina se limpia y se guarda en data/clean.
  6. Los documentos limpios se trocean en fragmentos (chunks).
  7. Se informa un resumen del resultado.

Ejecutar en tu maquina, con el .env ya configurado:
    uv run python src/scripts/run_ingest.py

Requiere que las variables del .env apunten al sitio y estrategia deseados.
Para BBVA se recomienda SCRAPER_TYPE=playwright (el sitio bloquea requests).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.factories import crear_scraper
from app.ingest.chunker import Chunker
from app.ingest.cleaner import Almacen, limpiar
from app.ingest.crawler import Crawler


def main() -> None:
    settings = get_settings()
    settings.asegurar_directorios()

    print("=" * 60)
    print("INGESTA - configuracion")
    print("=" * 60)
    print(f"  sitio semilla : {settings.scrape_base_url}")
    print(f"  estrategia    : {settings.scraper_type}")
    print(f"  profundidad   : {settings.max_depth}")
    print(f"  tope paginas  : {settings.max_pages}")
    print(f"  chunk_size    : {settings.chunk_size}  overlap: {settings.chunk_overlap}")

    # --- Paso 1: la Factory construye el scraper segun la config ---
    scraper = crear_scraper(settings)

    # --- Paso 2: el crawler recorre el sitio ---
    crawler = Crawler(
        scraper=scraper,
        max_depth=settings.max_depth,
        max_pages=settings.max_pages,
        delay=settings.scrape_delay,
    )

    print("\n" + "=" * 60)
    print("Recorriendo el sitio (esto puede tardar segun el tamano)...")
    print("=" * 60)
    try:
        paginas = crawler.crawl(settings.scrape_base_url)
    finally:
        # Liberamos el navegador si la estrategia era Playwright.
        scraper.close()

    print(f"  paginas descargadas: {len(paginas)}")
    if not paginas:
        print("\n  No se descargo ninguna pagina. Revisa conexion, la URL semilla")
        print("  o si el sitio bloquea la estrategia elegida (prueba playwright).")
        return

    # --- Paso 3: guardar crudos y limpiar ---
    almacen = Almacen(raw_dir=settings.raw_data_dir, clean_dir=settings.clean_data_dir)
    docs_limpios = []
    for pagina in paginas:
        almacen.guardar_crudo(pagina)
        doc = limpiar(pagina)
        # Solo guardamos documentos con contenido util (evita paginas vacias).
        if doc.text.strip():
            almacen.guardar_limpio(doc)
            docs_limpios.append(doc)

    print(f"  crudos guardados en : {settings.raw_data_dir}")
    print(f"  limpios guardados en: {settings.clean_data_dir}  ({len(docs_limpios)} con contenido)")

    # --- Paso 4: trocear en fragmentos ---
    chunker = Chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.trocear_todos(docs_limpios)

    print("\n" + "=" * 60)
    print("RESUMEN DE LA INGESTA")
    print("=" * 60)
    print(f"  paginas recorridas   : {len(paginas)}")
    print(f"  documentos con texto : {len(docs_limpios)}")
    print(f"  fragmentos generados : {len(chunks)}")
    if docs_limpios:
        chars = sum(len(d.text) for d in docs_limpios)
        print(f"  caracteres de texto  : {chars:,}")
        print(f"  media chars/fragmento: {chars // max(len(chunks), 1)}")
    print("\n  Ingesta completada. Los fragmentos estan listos para vectorizar")
    print("  (proximo paso: embeddings + ChromaDB).")


if __name__ == "__main__":
    main()