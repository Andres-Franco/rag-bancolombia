"""Base de datos vectorial sobre ChromaDB (patron Repository para vectores).

Envuelve ChromaDB con dos operaciones claras:
- indexar: guarda fragmentos (texto + vector + metadatos) en la coleccion.
- buscar:  dado el vector de una pregunta, devuelve los fragmentos mas cercanos.

Decisiones:
- PersistentClient: los datos se guardan en disco y sobreviven reinicios.
- embedding_function=None: nosotros pasamos los vectores ya calculados por
  nuestro Embedder (e5 con prefijos), en vez de dejar que Chroma los genere.
  Asi controlamos el modelo y garantizamos que documentos y consultas usan el
  mismo espacio vectorial.
- espacio de distancia 'cosine': adecuado para vectores normalizados de e5.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb


@dataclass
class Recuperado:
    # Un fragmento recuperado de la busqueda, con su distancia a la pregunta
    # (menor distancia = mas parecido).
    text: str
    url: str
    title: str
    distance: float


class VectorStore:
    """Almacena y recupera fragmentos vectorizados en ChromaDB."""

    def __init__(self, ruta: Path, coleccion: str) -> None:
        # PersistentClient guarda en disco en 'ruta'. Se crea la carpeta si no existe.
        ruta.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(ruta))
        # get_or_create es idempotente: si la coleccion ya existe, la reutiliza.
        # embedding_function=None porque aportamos los vectores nosotros.
        self._coleccion = self._client.get_or_create_collection(
            name=coleccion,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def indexar(
        self,
        ids: list[str],
        vectores: list[list[float]],
        textos: list[str],
        metadatos: list[dict],
    ) -> None:
        # Anade los fragmentos a la coleccion. Chroma guarda juntos el id, el
        # vector, el texto original y los metadatos (url, titulo).
        if not ids:
            return
        self._coleccion.add(
            ids=ids,
            embeddings=vectores,
            documents=textos,
            metadatas=metadatos,
        )

    def buscar(self, vector_consulta: list[float], top_k: int) -> list[Recuperado]:
        # Devuelve los top_k fragmentos mas cercanos al vector de la pregunta.
        resultado = self._coleccion.query(
            query_embeddings=[vector_consulta],
            n_results=top_k,
        )
        # Chroma devuelve listas anidadas (una por cada consulta); tomamos la
        # primera porque consultamos de a una pregunta.
        documentos = resultado["documents"][0]
        metadatos = resultado["metadatas"][0]
        distancias = resultado["distances"][0]

        recuperados = []
        for texto, meta, dist in zip(documentos, metadatos, distancias):
            recuperados.append(
                Recuperado(
                    text=texto,
                    url=meta.get("url", ""),
                    title=meta.get("title", ""),
                    distance=dist,
                )
            )
        return recuperados

    def contar(self) -> int:
        # Cuantos fragmentos hay indexados. Util para verificar la ingesta.
        return self._coleccion.count()

    def reiniciar(self) -> None:
        # Borra y recrea la coleccion. Util para re-indexar desde cero sin
        # arrastrar datos viejos.
        nombre = self._coleccion.name
        self._client.delete_collection(nombre)
        self._coleccion = self._client.get_or_create_collection(
            name=nombre,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )