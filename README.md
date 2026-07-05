# Asistente conversacional RAG sobre un sitio bancario

Sistema de **Retrieval-Augmented Generation (RAG)** que responde preguntas sobre
el contenido publicado en el sitio web de un banco. Extrae la informacion del
sitio mediante web scraping, la vectoriza en una base de datos vectorial, y
expone una interfaz conversacional que responde con base en ese contenido,
manteniendo el historial de cada conversacion.

> **Nota sobre el sitio.** El enunciado planteaba BBVA Colombia y permitia usar
> otro banco. Durante el desarrollo se detecto que el sitio de BBVA responde con
> HTTP 403 (proteccion anti-bot) a las peticiones automatizadas. Se implemento
> una estrategia de scraping con navegador (Playwright) para sortearlo, pero
> finalmente se uso **Bancolombia** (`https://www.bancolombia.com/personas`)
> como fuente, ya que permite scraping directo y ofrece contenido informativo
> rico. El sitio es configurable mediante variables de entorno.

---

## Tabla de contenido

- [Arquitectura general](#arquitectura-general)
- [Stack tecnologico y justificacion](#stack-tecnologico-y-justificacion)
- [Patrones de diseno](#patrones-de-diseno)
- [Requisitos previos](#requisitos-previos)
- [Puesta en marcha con Docker](#puesta-en-marcha-con-docker)
- [Puesta en marcha local (sin Docker)](#puesta-en-marcha-local-sin-docker)
- [Uso de la interfaz](#uso-de-la-interfaz)
- [Analisis del historico de conversaciones](#analisis-del-historico-de-conversaciones)
- [Configuracion](#configuracion)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Futuras mejoras](#futuras-mejoras)

---

## Arquitectura general

El sistema tiene dos flujos separados en el tiempo.

**1. Ingesta (offline).** Recorre el sitio, guarda y limpia el contenido, lo
trocea y lo indexa en la base vectorial:

```
scraper -> crawler -> guardado crudo -> limpieza -> guardado limpio
        -> chunking -> embeddings -> ChromaDB
```

**2. Conversacion (online).** Por cada pregunta, un grafo de LangGraph recupera
contexto, arma el prompt con el historial y genera la respuesta con Gemini:

```
pregunta + thread_id -> recuperar (ChromaDB) -> construir prompt
                     -> generar (Gemini) -> respuesta
```

El historial de cada conversacion se persiste por `thread_id` mediante el
checkpointer de LangGraph sobre SQLite.

### Estructura del proyecto

```
src/
  app/
    config.py            Configuracion central (pydantic-settings)
    factories.py         Fabricas de componentes (patron Factory)
    ingest/
      scraper.py         Estrategias de scraping (patron Strategy)
      crawler.py         Recorrido del sitio (BFS con limites)
      cleaner.py         Limpieza de HTML y almacenamiento crudo/limpio
      chunker.py         Troceado hibrido con solape
    rag/
      embeddings.py      Wrapper del modelo e5 (prefijos query/passage)
      vectorstore.py     ChromaDB: indexar y buscar
    graph/
      state.py           Estado del grafo conversacional
      nodes.py           Nodos del grafo (patron Chain of Responsibility)
      build.py           Ensamblado del grafo + checkpointer SQLite
    ui/
      gradio_app.py      Interfaz conversacional
  scripts/
    run_ingest.py        Ejecuta el pipeline de ingesta completo
    chat_consola.py      Chat por consola (para probar el motor)
    metrics_cli.py       Analisis del historico de conversaciones
    diagnostico_scraping.py  Compara estrategias de scraping
```

---

## Stack tecnologico y justificacion

| Componente | Eleccion | Por que |
|------------|----------|---------|
| Orquestacion | **LangGraph** | Modela la conversacion como grafo de nodos y trae persistencia de historial por sesion (checkpointer) integrada. |
| LLM | **Google Gemini** (Flash, tier gratuito) | Gratuito y suficiente para generar respuestas sobre contexto acotado. El nombre del modelo es configurable. |
| Embeddings | **intfloat/multilingual-e5-base** (HuggingFace) | Multilingue con buen rendimiento en espanol y orientado a retrieval. Gratuito y self-hosted. |
| Base vectorial | **ChromaDB** (embebido, persistente) | Tier gratuito / self-hosted, corre en el mismo proceso sin servicios extra, persiste en disco. |
| Interfaz | **Gradio** (serie 5.x) | Interfaz de chat funcional con muy poco codigo; maneja sesiones por navegador. |
| Historial | **SQLite** (checkpointer de LangGraph) | Persistencia sin servicios adicionales; un unico archivo aloja el checkpointer y la analitica. |
| Scraping | **requests + BeautifulSoup**, con **Playwright** opcional | Ligero por defecto; Playwright renderiza JS y sortea proteccion anti-bot cuando hace falta. |
| Entorno | **uv** + Python 3.11 | Instalacion rapida y reproducible (`uv.lock`), gestiona Python y dependencias. |

Todas las piezas de modelos, embeddings y base vectorial son gratuitas o de tier
gratuito, como valora el enunciado.

---

## Patrones de diseno

Se implementaron tres patrones, que emergen de la arquitectura de forma natural.

### 1. Strategy (comportamental) — `src/app/ingest/scraper.py`

Define una interfaz comun `Scraper` (clase abstracta con `fetch`) y dos
implementaciones intercambiables: `RequestsScraper` (ligero) y
`PlaywrightScraper` (renderiza JavaScript). El resto del sistema depende solo del
contrato, sin saber cual estrategia se usa. **Por que:** el sitio objetivo puede
requerir o no renderizado de JS; el patron permite cambiar de estrategia por
configuracion sin tocar el crawler ni el pipeline.

### 2. Factory (creacional) — `src/app/factories.py`

Funciones fabrica (`crear_scraper`, `crear_embedder`, `crear_vectorstore`,
`crear_llm`) que leen la configuracion y construyen la implementacion concreta
adecuada. Centralizan en un unico lugar la decision de que componente instanciar.
**Por que:** evita esparcir logica de construccion por el codigo y es el
complemento del Strategy (la Factory decide que estrategia crear). Se
implementaron como funciones por ser lo idiomatico en Python cuando no hay estado.

### 3. Chain of Responsibility (comportamental) — `src/app/graph/`

El grafo conversacional es una cadena de nodos (`recuperar` ->
`construir_prompt` -> `generar`), donde cada nodo procesa el estado y lo pasa al
siguiente. **Por que:** el flujo de una consulta RAG es naturalmente secuencial;
modelarlo como cadena lo hace explicito y facilita insertar nodos nuevos (por
ejemplo, un reranker) sin reescribir el flujo.

---

## Requisitos previos

**Para ejecutar con Docker (recomendado):**

- Docker y Docker Compose instalados.
- Una clave de API de Google Gemini (gratuita en <https://aistudio.google.com/apikey>).

**Para ejecutar localmente:**

- Python 3.11 y [uv](https://docs.astral.sh/uv/) instalados.
- La misma clave de API de Gemini.

---

## Puesta en marcha con Docker

```bash
# 1. Clonar el repositorio
git clone <URL-del-repo>
cd rag-bbva

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env y poner tu GEMINI_API_KEY

# 3. Construir la imagen (la primera vez tarda: descarga dependencias pesadas)
docker compose build

# 4. Ejecutar la ingesta una vez (puebla la base vectorial)
docker compose run --rm app python src/scripts/run_ingest.py

# 5. Levantar la aplicacion
docker compose up
```

Luego abre <http://localhost:7860> en el navegador.

> Segun tu version de Docker, el comando puede ser `docker compose` (con espacio,
> Compose V2) o `docker-compose` (con guion, version antigua). Ambos funcionan
> con este proyecto.

---

## Puesta en marcha local (sin Docker)

```bash
# 1. Clonar y entrar al proyecto
git clone <URL-del-repo>
cd rag-bbva

# 2. Instalar el entorno (uv descarga Python 3.11 y las dependencias)
uv sync

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env y poner tu GEMINI_API_KEY

# 4. Ejecutar la ingesta (scrapea, procesa e indexa)
uv run python src/scripts/run_ingest.py

# 5. Levantar la interfaz
uv run python -m app.ui.gradio_app
```

Tambien puedes probar el motor por consola sin interfaz web:

```bash
uv run python src/scripts/chat_consola.py
```

> Alternativa con pip: existe un `requirements.txt` generado desde el lock, por
> si se prefiere `pip install -r requirements.txt` en un entorno virtual propio.

---

## Uso de la interfaz

1. Abre <http://localhost:7860>.
2. Escribe una pregunta sobre los productos o servicios del banco (por ejemplo,
   *"Que tarjetas de credito ofrecen?"* o *"Que beneficios tiene la banca
   preferencial?"*).
3. El sistema recupera los fragmentos mas relevantes del contenido indexado, se
   los pasa a Gemini junto con el historial reciente, y responde citando la
   fuente.
4. La conversacion mantiene memoria: puedes hacer preguntas de seguimiento que
   se apoyen en respuestas anteriores. Cada sesion de navegador tiene su propio
   historial aislado.

---

## Analisis del historico de conversaciones

El sistema persiste todas las conversaciones. Para extraer metricas de impacto
del historico:

```bash
# Con Docker
docker compose run --rm app python src/scripts/metrics_cli.py

# Local
uv run python src/scripts/metrics_cli.py
```

Reporta metricas como numero de conversaciones, mensajes totales, promedio de
mensajes por sesion, y las preguntas mas frecuentes.

---

## Configuracion

Todos los parametros se externalizan en el `.env` (ver `.env.example`). Los mas
relevantes:

| Variable | Descripcion | Por defecto |
|----------|-------------|-------------|
| `GEMINI_API_KEY` | Clave de la API de Gemini (obligatoria) | — |
| `GEMINI_MODEL` | Modelo de Gemini a usar | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | Modelo de embeddings de HuggingFace | `intfloat/multilingual-e5-base` |
| `SCRAPE_BASE_URL` | URL semilla del sitio a scrapear | Bancolombia personas |
| `SCRAPER_TYPE` | Estrategia: `requests` o `playwright` | `requests` |
| `MAX_DEPTH` / `MAX_PAGES` | Limites del crawl | `2` / `40` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamano y solape de los fragmentos | `900` / `140` |
| `RETRIEVAL_TOP_K` | Fragmentos recuperados por pregunta | `5` |
| `HISTORY_WINDOW` | Numero N de mensajes previos como contexto | `6` |
| `USE_RERANKER` | Activa el reranker (bonus) | `false` |

---

## Limitaciones conocidas

En linea con la honestidad que pide el enunciado:

- **Fuente cambiada a Bancolombia.** El sitio de BBVA bloquea el scraping con
  HTTP 403. Se dejo implementada la estrategia Playwright para sortearlo, pero
  se opto por Bancolombia por practicidad. El sistema funciona con cualquiera de
  los dos ajustando `.env`.
- **Ruido de plantillas JavaScript.** El sitio de Bancolombia renderiza parte de
  su contenido con JS, lo que deja artefactos en el HTML (nombres de iconos,
  placeholders de plantilla). Se implemento un filtro de limpieza especifico
  para eliminarlos, pero puede quedar algo de ruido residual.
- **Limpieza heuristica.** La eliminacion de menus y pies se hace por etiquetas
  estructurales; en casos atipicos podria descartarse contenido util o colarse
  algo de ruido.
- **Sin lectura de robots.txt.** El crawler no consulta `robots.txt`. Para un
  uso sobre contenido publico informativo es aceptable, pero se deja anotado.
- **Version de Gradio fijada.** Gradio 6.0 introdujo cambios de API que rompian
  la interfaz; se fijo la serie 5.x estable para garantizar reproducibilidad.

---

## Futuras mejoras

- **Reranker (bonus).** El grafo esta preparado para insertar un nodo de
  reranking (cross-encoder) entre la recuperacion y la generacion, activable con
  `USE_RERANKER`. Reordenaria los fragmentos por relevancia antes de pasarlos al
  LLM.
- **Scraping con renderizado completo.** Usar Playwright por defecto para
  obtener el contenido ya renderizado y reducir el ruido en origen.
- **Crawl mas amplio y respeto de robots.txt.**
- **Evaluacion de calidad del retrieval** con un conjunto de preguntas de prueba.
- **Streaming de la respuesta** en la interfaz para mejor experiencia.

---

## Notas de diseno

- **Un solo servicio en Docker.** ChromaDB corre embebido (persistente en disco)
  y SQLite es un archivo, por lo que no se necesitan contenedores separados para
  la base vectorial ni el historial. El `docker-compose.yml` define un unico
  servicio.
- **Historial y analitica comparten el mismo SQLite.** El checkpointer de
  LangGraph y las metricas leen del mismo archivo, sin infraestructura extra.
- **Prefijos de e5.** El modelo de embeddings requiere prefijos `query:` y
  `passage:`; se encapsulan en el wrapper para que el resto del codigo no tenga
  que recordarlos.