"""Nodos del grafo conversacional (patron Chain of Responsibility).

Cada nodo es una funcion que recibe el estado, hace una tarea, y devuelve las
partes del estado que modifico. LangGraph los encadena en orden.

Los nodos necesitan colaboradores (embedder, vectorstore, llm) que construye la
Factory. Para inyectarlos sin que el nodo los cree por su cuenta, usamos el
patron de "funcion que construye el nodo": una funcion externa recibe los
colaboradores y devuelve la funcion-nodo, que los recuerda por clausura. Asi el
nodo queda desacoplado y se puede probar con componentes simulados.
"""

from app.graph.state import EstadoConversacion


def construir_nodo_recuperar(embedder, vectorstore, top_k: int):
    """Devuelve el nodo que recupera contexto de ChromaDB para la pregunta.

    embedder y vectorstore se reciben aqui y quedan "recordados" por la funcion
    interna (clausura). El nodo solo recibe el estado cuando LangGraph lo ejecuta.
    """

    def recuperar(estado: EstadoConversacion) -> dict:
        pregunta = estado["pregunta"]
        # 1. Convertir la pregunta en vector (el Embedder pone el prefijo query:).
        vector = embedder.embeber_consulta(pregunta)
        # 2. Buscar los top_k fragmentos mas cercanos en ChromaDB.
        recuperados = vectorstore.buscar(vector_consulta=vector, top_k=top_k)
        # 3. Convertir a diccionarios simples para que el checkpointer los
        #    serialice sin problemas (evita guardar tipos personalizados).
        contexto = [
            {"text": r.text, "url": r.url, "title": r.title, "distance": r.distance}
            for r in recuperados
        ]
        return {"contexto": contexto}

    return recuperar


# Instruccion de sistema: define el rol del asistente y las reglas anti-alucinacion.
_SYSTEM_PROMPT = (
    "Eres un asistente virtual del banco que responde preguntas sobre sus "
    "productos y servicios. Responde de forma clara y concisa, SOLO con base en "
    "el CONTEXTO que se te proporciona. Si el contexto no contiene la "
    "informacion necesaria para responder, dilo honestamente en vez de inventar. "
    "Cuando uses informacion del contexto, menciona la fuente (URL) de donde "
    "proviene. Responde siempre en espanol."
)


def _formatear_contexto(recuperados) -> str:
    # Presenta los fragmentos numerados y con su fuente, para que el LLM pueda
    # citarlos. Los fragmentos son diccionarios con claves text, url, title.
    if not recuperados:
        return "(No se encontro contexto relevante para esta pregunta.)"
    bloques = []
    for i, r in enumerate(recuperados, start=1):
        bloques.append(f"[{i}] Fuente: {r['url']}\n{r['text']}")
    return "\n\n".join(bloques)


def construir_nodo_prompt(history_window: int):
    """Devuelve el nodo que arma el prompt con contexto, historial y pregunta.

    history_window controla cuantos mensajes previos se incluyen (los N ultimos).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    def construir_prompt(estado: EstadoConversacion) -> dict:
        pregunta = estado["pregunta"]
        contexto = estado.get("contexto", [])

        # Recorte a los ultimos N mensajes del historial (requisito N configurable).
        # Excluimos el ultimo si es la pregunta actual ya anadida; tomamos el
        # historial previo tal cual esta en el estado.
        historial = estado.get("messages", [])
        historial_reciente = historial[-history_window:] if history_window > 0 else []

        # Construimos el texto del contexto recuperado.
        texto_contexto = _formatear_contexto(contexto)

        # El mensaje humano combina el contexto y la pregunta. El historial se
        # pasa como mensajes previos aparte, para que el LLM vea la conversacion.
        contenido_usuario = (
            f"CONTEXTO:\n{texto_contexto}\n\n"
            f"PREGUNTA: {pregunta}"
        )

        # La lista de mensajes que recibira el LLM: instruccion de sistema,
        # historial reciente, y el mensaje actual con contexto + pregunta.
        mensajes_para_llm = (
            [SystemMessage(content=_SYSTEM_PROMPT)]
            + list(historial_reciente)
            + [HumanMessage(content=contenido_usuario)]
        )

        # Guardamos estos mensajes en un campo del estado para que el nodo
        # generar los use. No usamos 'messages' (que es el historial acumulado)
        # para no mezclar el prompt con el registro de la conversacion.
        return {"mensajes_llm": mensajes_para_llm}

    return construir_prompt


def construir_nodo_generar(llm):
    """Devuelve el nodo que llama al LLM y produce la respuesta.

    llm es el modelo ya construido (ChatGoogleGenerativeAI). Se inyecta para
    poder probar el nodo con un LLM simulado.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    def generar(estado: EstadoConversacion) -> dict:
        mensajes = estado.get("mensajes_llm", [])
        pregunta = estado["pregunta"]
        try:
            # invoke envia los mensajes al modelo y devuelve un AIMessage.
            respuesta_llm = llm.invoke(mensajes)
            texto = respuesta_llm.content
        except Exception as exc:
            # Manejo de errores: si la llamada falla (limite de peticiones, red,
            # etc.), devolvemos un mensaje de disculpa en vez de romper la app.
            texto = (
                "Lo siento, no pude generar una respuesta en este momento por un "
                "problema tecnico. Intenta de nuevo en unos segundos."
            )
            # Registramos el error para depuracion sin exponerlo al usuario.
            print(f"[generar] Error al invocar el LLM: {exc}")

        # Actualizamos dos cosas:
        # 1. 'respuesta': el texto para mostrar al usuario en este turno.
        # 2. 'messages': anadimos la pregunta y la respuesta al historial
        #    acumulado (add_messages las agrega sin borrar lo previo). Esto es
        #    lo que da memoria a la conversacion entre turnos.
        return {
            "respuesta": texto,
            "messages": [HumanMessage(content=pregunta), AIMessage(content=texto)],
        }

    return generar