"""Ensamblado del grafo conversacional.

Conecta los tres nodos en cadena y anade el checkpointer de SQLite, que da
persistencia del historial por thread_id (el "ID" del requisito 5).

Flujo: START -> recuperar -> construir_prompt -> generar -> END
Esa cadena lineal de nodos es el patron Chain of Responsibility.

El checkpointer guarda el estado de cada conversacion en SQLite. Al invocar el
grafo con un thread_id, LangGraph carga el historial previo de esa sesion y
guarda el nuevo estado al terminar, de forma automatica.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.factories import crear_embedder, crear_llm, crear_vectorstore
from app.graph.nodes import (
    construir_nodo_generar,
    construir_nodo_prompt,
    construir_nodo_recuperar,
)
from app.graph.state import EstadoConversacion


class AsistenteRAG:
    """Envuelve el grafo compilado y expone una forma simple de preguntar."""

    def __init__(self, grafo, conexion_sqlite) -> None:
        self._grafo = grafo
        self._conn = conexion_sqlite

    def preguntar(self, pregunta: str, thread_id: str) -> str:
        # Invoca el grafo para una pregunta dentro de una sesion (thread_id).
        # LangGraph carga el historial previo de ese thread y guarda el nuevo
        # estado automaticamente gracias al checkpointer.
        config = {"configurable": {"thread_id": thread_id}}
        estado_entrada = {"pregunta": pregunta}
        resultado = self._grafo.invoke(estado_entrada, config=config)
        return resultado["respuesta"]

    def historial(self, thread_id: str) -> list:
        # Devuelve los mensajes acumulados de una sesion, leyendo el estado
        # guardado por el checkpointer. Util para mostrar la conversacion.
        config = {"configurable": {"thread_id": thread_id}}
        estado = self._grafo.get_state(config)
        if estado and estado.values:
            return estado.values.get("messages", [])
        return []

    def cerrar(self) -> None:
        # Cierra la conexion SQLite al terminar la aplicacion.
        self._conn.close()


def construir_asistente(settings: Settings) -> AsistenteRAG:
    """Construye el asistente completo: componentes, nodos, grafo y checkpointer."""
    settings.asegurar_directorios()

    # 1. La Factory construye los colaboradores desde la config.
    embedder = crear_embedder(settings)
    vectorstore = crear_vectorstore(settings)
    llm = crear_llm(settings)

    # 2. Construimos los tres nodos, inyectando sus colaboradores.
    nodo_recuperar = construir_nodo_recuperar(
        embedder=embedder, vectorstore=vectorstore, top_k=settings.retrieval_top_k
    )
    nodo_prompt = construir_nodo_prompt(history_window=settings.history_window)
    nodo_generar = construir_nodo_generar(llm=llm)

    # 3. Declaramos el grafo sobre nuestro estado y registramos los nodos.
    builder = StateGraph(EstadoConversacion)
    builder.add_node("recuperar", nodo_recuperar)
    builder.add_node("construir_prompt", nodo_prompt)
    builder.add_node("generar", nodo_generar)

    # 4. Definimos el flujo lineal (Chain of Responsibility).
    builder.add_edge(START, "recuperar")
    builder.add_edge("recuperar", "construir_prompt")
    builder.add_edge("construir_prompt", "generar")
    builder.add_edge("generar", END)

    # 5. Checkpointer de SQLite para persistir el historial por thread_id.
    #    check_same_thread=False permite que Gradio llame desde distintos hilos.
    conn = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()  # crea las tablas del checkpointer si no existen

    # 6. Compilamos el grafo con el checkpointer.
    grafo = builder.compile(checkpointer=checkpointer)

    return AsistenteRAG(grafo=grafo, conexion_sqlite=conn)