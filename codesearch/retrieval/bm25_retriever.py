from codesearch.indexing.bm25_index import BM25Index, tokenize
from codesearch.retrieval.types import ScoredFunction

@dataclass
class BM25Retriever:
    index: BM25Index

    def search(self, query: str, top_k: int = 20) -> list[ScoredFunction]:
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.index.bm25.get_scores(tokenized_query).tolist()

        func_score_pair = [(func, score) for func, score in zip(self.index.functions, scores) if score > 0]
        if not func_score_pair:
            return []
        func_score_pair.sort(key=lambda x: x[1], reverse=True)

        top_results = func_score_pair[:top_k]

        return [
            ScoredFunction(function=func, score=score, rank=rank, retriever="bm25")
            for rank, (func, score) in enumerate(top_results, start=1)
        ]