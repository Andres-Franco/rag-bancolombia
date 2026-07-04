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
        # 3. Devolver solo el campo que este nodo modifica: el contexto.
        return {"contexto": recuperados}

    return recuperar