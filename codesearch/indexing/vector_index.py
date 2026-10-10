from pathlib import Path
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from codesearch.parsing.parser import FunctionInfo

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_BATCH_SIZE = 4000


@dataclass
class VectorIndex:
    collection: Collection
    model: "SentenceTransformer"
    model_name: str

    @classmethod
    def build(
            cls,
            functions: list[FunctionInfo],
            persist: bool,
            collection_name: str | None = None,
            persist_path: Path | None = None,
            model_name: str = DEFAULT_MODEL_NAME,
    ) -> "VectorIndex":
        from sentence_transformers import SentenceTransformer

        if not functions:
            raise ValueError("Cannot build VectorIndex with an empty list of functions.")
        if persist and not collection_name:
            raise ValueError("collection_name is required when persistent client is used.")
        if persist and persist_path is None:
            raise ValueError("persist_path is required when persist=True.")

        if persist:
            client = chromadb.PersistentClient(path=str(persist_path))
            try:
                client.delete_collection(name=collection_name)
            except NotFoundError:
                pass
            name = collection_name
        else:
            client = chromadb.EphemeralClient()
            name = collection_name or "temp_collection"

        collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

        if model_name == DEFAULT_MODEL_NAME:
            model = SentenceTransformer(model_name)
        else:
            model = SentenceTransformer(model_name)

        ids = [f"{func.file.as_posix()}:{func.name}:{func.line}" for func in functions]
        documents = [func.composite_doc for func in functions]
        embeddings = model.encode(documents, batch_size=256, show_progress_bar=False).tolist()
        metadatas = [
            {
                "file": func.file.as_posix(),
                "name": func.name,
                "line": func.line,
                "doc_string": func.doc_string if func.doc_string is not None else "",
                "class_name": func.class_name if func.class_name is not None else "",
            }
            for func in functions
        ]

        for i in range(0, len(ids), CHROMA_BATCH_SIZE):
            collection.add(
                ids=ids[i:i + CHROMA_BATCH_SIZE],
                embeddings=embeddings[i:i + CHROMA_BATCH_SIZE],
                metadatas=metadatas[i:i + CHROMA_BATCH_SIZE],
                documents=documents[i:i + CHROMA_BATCH_SIZE],
            )

        return cls(collection=collection, model=model, model_name=model_name)

    @classmethod
    def load(
        cls,
        collection_name: str,
        persist_path: Path,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> "VectorIndex":
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(persist_path))

        try:
            collection = client.get_collection(name=collection_name)
        except NotFoundError as e:
            raise ValueError(
                f"No collection named '{collection_name}' found at: {persist_path}"
                f"Try running 'codesearch reindex' to fix."
            ) from e


        loaded_model = SentenceTransformer(model_name)

        return cls(collection=collection, model=loaded_model, model_name=model_name)

    @classmethod
    def delete(cls, collection_name: str, persist_path: Path) -> None:
        client = chromadb.PersistentClient(path=str(persist_path))
        try:
            client.delete_collection(name=collection_name)
        except NotFoundError:
            pass