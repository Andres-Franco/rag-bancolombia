"""Interfaz conversacional con Gradio.

Conecta la ventana de chat con el AsistenteRAG. Cada sesion de navegador recibe
un thread_id unico (via gr.State), de modo que la memoria de un usuario no se
mezcla con la de otro. El asistente se construye una sola vez al arrancar
(carga el modelo de embeddings y abre ChromaDB) y se reutiliza en cada mensaje.

Ejecutar:
    uv run python -m app.ui.gradio_app
o mediante el script:
    uv run python src/scripts/run_app.py
"""

from uuid import uuid4

import gradio as gr

from app.config import get_settings
from app.graph.build import construir_asistente


def crear_interfaz():
    settings = get_settings()

    # Se construye una sola vez: es costoso (modelo de embeddings + ChromaDB).
    # Todos los usuarios comparten este asistente, pero cada uno con su thread_id.
    print("Construyendo el asistente (carga el modelo de embeddings)...")
    asistente = construir_asistente(settings)
    print("Asistente listo.")

    def responder(mensaje: str, history: list, thread_id: str) -> str:
        # history y el formato de mensajes los gestiona Gradio; nosotros solo
        # delegamos en el asistente, que maneja su propia memoria por thread_id.
        if not mensaje or not mensaje.strip():
            return "Por favor escribe una pregunta."
        return asistente.preguntar(mensaje, thread_id=thread_id)

    with gr.Blocks(title="Asistente RAG") as demo:

        gr.Markdown(
            "# Asistente virtual\n"
            "Haz preguntas sobre los productos y servicios publicados en el "
            "sitio del banco. Las respuestas se basan en el contenido recuperado."
        )

        # Estado de sesion: un thread_id unico por cada visitante. gr.State con
        # una funcion (uuid4) genera un valor nuevo por sesion de navegador.
        thread_state = gr.State(lambda: str(uuid4())[:12])

        chatbot = gr.Chatbot(type="messages", 
                             height=460, 
                             label="Conversacion",
                             allow_tags=False)

        gr.ChatInterface(
            fn=responder,
            type="messages",
            chatbot=chatbot,
            additional_inputs=[thread_state]
        )

    return demo, asistente


def main() -> None:
    demo, asistente = crear_interfaz()
    try:
        # server_name 0.0.0.0 permite acceder desde fuera del contenedor Docker.
        demo.launch(server_name="0.0.0.0", server_port=7860)
    finally:
        asistente.cerrar()


if __name__ == "__main__":
    main()