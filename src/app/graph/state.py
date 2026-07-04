"""Estado del grafo conversacional.

El estado es la estructura de datos que viaja por todos los nodos del grafo.
Cada nodo lee lo que necesita y escribe su resultado de vuelta. LangGraph usa
este TypedDict para saber que campos existen y como combinar los valores que
cada nodo devuelve.

Detalle clave: el campo 'messages' usa el reducer add_messages. Un reducer le
dice a LangGraph como combinar el valor nuevo con el existente. add_messages
ACUMULA (anade los mensajes nuevos al historial) en vez de reemplazar, que es
lo que queremos para mantener la conversacion. Los demas campos se reemplazan
en cada turno (comportamiento por defecto).
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Importamos el tipo del fragmento recuperado para tipar el contexto.
from app.rag.vectorstore import Recuperado


class EstadoConversacion(TypedDict):
    # La pregunta del usuario en este turno.
    pregunta: str

    # El historial de mensajes de la conversacion. El Annotated con add_messages
    # indica que este campo se ACUMULA: cada turno anade mensajes en vez de
    # sobrescribir. Asi el grafo recuerda lo que se dijo antes.
    messages: Annotated[list[BaseMessage], add_messages]

    # Los fragmentos recuperados de ChromaDB para esta pregunta. Se reemplazan
    # en cada turno (el contexto es especifico de la pregunta actual).
    contexto: list[Recuperado]

    # Mensajes ya ensamblados que el nodo generar enviara al LLM (instruccion de
    # sistema + historial reciente + pregunta con contexto). Es un campo interno
    # de paso entre el nodo prompt y el nodo generar; se reemplaza cada turno.
    mensajes_llm: list[BaseMessage]

    # La respuesta generada por el LLM en este turno.
    respuesta: str