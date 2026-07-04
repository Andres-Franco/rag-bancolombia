"""Troceado (chunking) de los documentos limpios en fragmentos.

Estrategia hibrida: se respeta la estructura del texto (se corta primero por
parrafos, luego por lineas y espacios) y solo se subdivide cuando un bloque
supera el tamano objetivo. Entre fragmentos consecutivos se deja un solape para
no perder contexto en las fronteras.

Nos apoyamos en RecursiveCharacterTextSplitter de langchain-text-splitters, que
implementa justamente ese corte jerarquico, en vez de reescribirlo a mano.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest.cleaner import CleanDoc


@dataclass
class Chunk:
    # Un fragmento listo para vectorizar. Ademas del texto, conserva la url y el
    # titulo de la pagina de origen (para citar la fuente) y un indice que dice
    # que numero de fragmento es dentro de su documento.
    text: str
    url: str
    title: str
    index: int


class Chunker:
    """Trocea documentos limpios en fragmentos con solape."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        # Los separadores van del mas "fuerte" al mas "debil": primero intenta
        # cortar por parrafos (doble salto), luego por linea, luego por espacio,
        # y como ultimo recurso por caracter. Asi respeta la estructura y solo
        # baja de nivel cuando un bloque sigue siendo demasiado grande.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def trocear_documento(self, doc: CleanDoc) -> list[Chunk]:
        # Divide el texto de UN documento en fragmentos, conservando su origen.
        textos = self._splitter.split_text(doc.text)
        return [
            Chunk(text=t, url=doc.url, title=doc.title, index=i)
            for i, t in enumerate(textos)
            if t.strip()  # descartamos fragmentos que queden vacios
        ]

    def trocear_todos(self, docs: list[CleanDoc]) -> list[Chunk]:
        # Trocea una lista de documentos y devuelve todos los fragmentos juntos.
        fragmentos: list[Chunk] = []
        for doc in docs:
            fragmentos.extend(self.trocear_documento(doc))
        return fragmentos