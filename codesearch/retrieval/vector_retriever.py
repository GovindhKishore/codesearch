from dataclasses import dataclass
from codesearch.indexing.vector_index import VectorIndex
from codesearch.parsing.parser import FunctionInfo
from codesearch.retrieval.types import ScoredFunction
from pathlib import Path


@dataclass
class VectorRetriever:
    index: VectorIndex

    def search(self, query: str, top_k: int = 20, similarity_threshold: float = 0.3) -> list[ScoredFunction]:
        query_embedding = self.index.model.encode([query]).tolist()

        results = self.index.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        distances = results["distances"][0]
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]

        scored_functions = []
        for rank, (dist, doc_id, meta) in enumerate(zip(distances, ids, metadatas), start=1):
            similarity = 1 - dist
            if similarity < similarity_threshold:
                continue

            function = FunctionInfo(
                name=meta["name"],
                file=Path(meta["file"]),
                line=meta["line"],
                doc_string=None,
                params=[],
                return_type=None,
                callees=[],
                callers=[],
            )

            scored_functions.append(ScoredFunction(
                function=function,
                score=similarity,
                rank=rank,
                retriever="vector",
            ))

        return scored_functions

