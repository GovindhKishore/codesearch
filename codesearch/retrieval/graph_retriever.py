from dataclasses import dataclass
from codesearch.indexing.graph_index import GraphIndex
from codesearch.retrieval.types import ScoredFunction
from pathlib import Path
import math
from codesearch.parsing.parser import FunctionInfo
import networkx as nx


@dataclass
class GraphRetriever:
    index: GraphIndex

    def search(self, seeds: list[ScoredFunction], max_hop: int = 1, decay_factor: float = 0.5) -> list[ScoredFunction]:
            seed_names = {(s.function.name, s.function.class_name) for s in seeds}
            reversed_graph = self.index.graph.reverse()

            seed_to_callees = self._reach_from_seeds(self.index.graph, seed_names, max_hop)
            seed_to_callers = self._reach_from_seeds(reversed_graph, seed_names, max_hop)

            return self._build_scored_function(seed_to_callees, seed_to_callers, decay_factor)

    def _reach_from_seeds(self, graph, seeds, max_hop):
        reach = {}
        for seed in seeds:
            if not graph.has_node(seed):
                continue
            dists = nx.single_source_shortest_path_length(graph, seed, cutoff=max_hop)
            for node, d in dists.items():
                if d == 0:
                    continue
                if node not in reach:
                    reach[node] = {}
                reach[node][seed] = d
        return reach

    def _build_scored_function(self, seed_to_callees, seed_to_callers  , decay_factor: float) -> list[ScoredFunction]:

        merged = {}
        for part in (seed_to_callees, seed_to_callers):
            for node, per_seed in part.items():
                if node not in merged:
                    merged[node] = {}
                for seed, d in per_seed.items():
                    if seed not in merged[node] or d < merged[node][seed]:
                        merged[node][seed] = d

        scored_functions = []
        for (name, class_name), per_seed in merged.items():
            node_data = self.index.graph.nodes[(name, class_name)]
            in_degree = self.index.graph.in_degree((name, class_name))

            score = (sum(decay_factor ** d for d in per_seed.values()))

            function = FunctionInfo(
                name=name,
                file=Path(node_data["file"]),
                line=node_data["line"],
                doc_string=node_data["doc_string"],
                class_name=class_name,
                params=[],
                return_type=None,
                callees=[],
                callers=[],
            )
            scored_functions.append(ScoredFunction(
                function=function,
                score=score,
                rank=0,
                retriever="structural",
            ))

        scored_functions.sort(key=lambda x: x.score, reverse=True)
        for rank, sf in enumerate(scored_functions, start=1):
            sf.rank = rank

        return scored_functions
