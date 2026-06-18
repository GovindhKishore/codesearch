from codesearch.parsing.parser import FunctionInfo
from codesearch.retrieval.types import ScoredFunction
from dataclasses import dataclass

@dataclass
class Fuser:
    structural_weight: float = 1.0
    k: int = 60

    def fuse(
            self,
            bm25_results: list[ScoredFunction],
            vector_results: list[ScoredFunction],
            structural_results: list[ScoredFunction],
            bm25_weight: float = 1.0,
            vector_weight: float = 1.0,
            structural_weight: float = 1.0,
    ) -> list[ScoredFunction]:
        scores: dict[tuple[str, str, int], float] = {}
        function_map: dict[tuple[str, str, int], FunctionInfo] = {}

        self._accumulate(results=bm25_results, scores=scores, function_map=function_map, weight=bm25_weight)
        self._accumulate(results=vector_results, scores=scores, function_map=function_map, weight=vector_weight)
        self._accumulate(results=structural_results, scores=scores, function_map=function_map, weight=structural_weight)

        sorted_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)

        return [
            ScoredFunction(function=function_map[key], score=scores[key], rank=rank, retriever="fused")
            for rank, key in enumerate(sorted_keys, start=1)
        ]

    def _accumulate_scores(
            self,
            results: list[ScoredFunction],
            scores: dict[tuple[str, str, int], float],
            function_map: dict[tuple[str, str, int], FunctionInfo],
            weight: float = 1.0,
    ) -> None:
        for result in results:
            key = ( result.function.file.as_posix() ,result.function.name, result.function.line)
            scores[key] = scores.get(key, 0) + (weight * (1 / (self.k + result.rank)))
            if key not in function_map:
                function_map[key] = result.function

