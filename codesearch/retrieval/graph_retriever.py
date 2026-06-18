from dataclasses import dataclass
from codesearch.indexing.graph_index import GraphIndex
from codesearch.retrieval.types import ScoredFunction
from pathlib import Path
from codesearch.parsing.parser import FunctionInfo
import networkx as nx


@dataclass
class GraphRetriever:
    index: GraphIndex

    def search(self, seeds: list[ScoredFunction], max_hop: int = 2, decay_factor: float = 0.5) -> list[ScoredFunction]:
            seed_names = {s.function.name for s in seeds}
            reversed_graph = self.index.graph.reverse()
            min_hops: dict[str, int] = {}

            self._update_min_hops(self.index.graph, seed_names, max_hop, min_hops)
            self._update_min_hops(reversed_graph, seed_names, max_hop, min_hops)

            return self._build_scored_function(min_hops, decay_factor)

    def _update_min_hops(self, graph, seed_names: set[str], max_hop: int, min_hops: dict[str, int]) -> None:
        valid_sources = [s for s in seed_names if graph.has_node(s)]
        for current_hop, layer_nodes in enumerate(nx.bfs_layers(graph, valid_sources)):
            if current_hop == 0:
                continue
            if current_hop > max_hop:
                break
            for node in layer_nodes:
                if node not in min_hops or current_hop < min_hops[node]:
                    min_hops[node] = current_hop

    def _build_scored_function(self, min_hops: dict[str, int], decay_factor: float) -> list[ScoredFunction]:
        scored_functions = []
        for name, hop in min_hops.items():
            node_data = self.index.graph.nodes[name]
            function = FunctionInfo(
                name=name,
                file=Path(node_data["file"]),
                line=node_data["line"],
                doc_string=None,
                params=[],
                return_type=None,
                callees=[],
                callers=[],
            )
            scored_functions.append(ScoredFunction(
                function=function,
                score=decay_factor ** hop,
                rank=hop,
                retriever="structural",
            ))
        return scored_functions
