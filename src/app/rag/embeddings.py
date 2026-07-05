"""Wrapper del modelo de embeddings (patron Strategy en la capa de vectorizacion).

Convierte texto en vectores usando un modelo de HuggingFace. Encapsula un detalle
critico de la familia e5: cada texto debe llevar un prefijo segun su rol.
- "passage: " para los fragmentos de contenido que se indexan.
- "query: "   para las preguntas del usuario.

Poner los prefijos mal (u omitirlos) degrada la calidad del retrieval. Al
encapsularlos aqui, el resto del sistema no tiene que recordarlos: llama a
embeber_documentos o a embeber_consulta y los prefijos se aplican solos.
"""

from sentence_transformers import SentenceTransformer


class Embedder:
    """Genera embeddings con prefijos correctos segun el rol del texto."""

    def __init__(self, modelo: str, normalizar: bool = True) -> None:
        # Cargar el modelo descarga los pesos la primera vez (varios cientos de
        # MB) y luego los cachea. Se hace una sola vez al crear el Embedder.
        self._model = SentenceTransformer(modelo)
        # normalize_embeddings=True deja los vectores en norma 1, lo que hace
        # que la similitud por producto/coseno sea consistente. Recomendado
        # para e5.
        self._normalizar = normalizar

    def _codificar(self, textos: list[str]) -> list[list[float]]:
        # Llama al modelo y devuelve listas de floats (no arrays de numpy), para
        # que ChromaDB y el resto del codigo trabajen con tipos simples.
        vectores = self._model.encode(
            textos,
            normalize_embeddings=self._normalizar,
            convert_to_numpy=True,
        )
        return vectores.tolist()

    def embeber_documentos(self, textos: list[str]) -> list[list[float]]:
        # Prefijo "passage: " para el contenido que se indexa.
        con_prefijo = [f"passage: {t}" for t in textos]
        return self._codificar(con_prefijo)

    def embeber_consulta(self, texto: str) -> list[float]:
        # Prefijo "query: " para la pregunta del usuario. Devuelve un solo vector.
        con_prefijo = [f"query: {texto}"]
        return self._codificar(con_prefijo)[0]

    @property
    def dimension(self) -> int:
        # Numero de dimensiones del vector (768 para e5-base). Util para
        # configurar la coleccion de la base vectorial.
        return self._model.get_sentence_embedding_dimension()