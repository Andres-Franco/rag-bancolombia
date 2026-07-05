FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# UV_COMPILE_BYTECODE: precompila a bytecode para arranque mas rapido.
# UV_LINK_MODE=copy: evita warnings al usar cache montada.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# --- Capa 1: dependencias (se cachea si no cambian pyproject/uv.lock) ---
# Copiamos SOLO los archivos de dependencias primero. Asi, si solo cambia el
# codigo, Docker reutiliza esta capa y no reinstala todo (build mas rapido).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# --- Capa 2: el codigo del proyecto ---
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Ponemos el entorno virtual al frente del PATH para usar sus ejecutables.
ENV PATH="/app/.venv/bin:$PATH"

# Puerto de Gradio.
EXPOSE 7860

# HF_HOME apunta a un directorio persistente para cachear el modelo de
# embeddings descargado, y no volver a bajarlo en cada arranque.
ENV HF_HOME=/app/data/hf_cache

# Por defecto, levanta la interfaz Gradio. El pipeline de ingesta se ejecuta
# aparte (ver README) con: docker compose run --rm app python src/scripts/run_ingest.py
CMD ["python", "-m", "app.ui.gradio_app"]