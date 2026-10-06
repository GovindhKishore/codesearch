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
            min_hops: dict[tuple[str, str | None], int] = {}

            self._update_min_hops(self.index.graph, seed_names, max_hop, min_hops)
            self._update_min_hops(reversed_graph, seed_names, max_hop, min_hops)

            seed_connections: dict[tuple[str, str | None], int] = {}

            for seed in seed_names:
                if not self.index.graph.has_node(seed):
                    continue

                reachable = nx.single_source_shortest_path_length(
                    self.index.graph,
                    seed,
                    cutoff=max_hop,
                )

                for node in reachable:
                    if node != seed:
                        seed_connections[node] = seed_connections.get(node, 0) + 1

            return self._build_scored_function(min_hops, decay_factor, seed_connections)

    def _update_min_hops(self, graph, seed_names: set[tuple[str, str | None]], max_hop: int, min_hops: dict[tuple[str, str | None], int]) -> None:
        valid_sources = [s for s in seed_names if graph.has_node(s)]
        for current_hop, layer_nodes in enumerate(nx.bfs_layers(graph, valid_sources)):
            if current_hop == 0:
                continue
            if current_hop > max_hop:
                break
            for node in layer_nodes:
                if node not in min_hops or current_hop < min_hops[node]:
                    min_hops[node] = current_hop

    def _build_scored_function(self, min_hops: dict[tuple[str, str | None], int], decay_factor: float, seed_connections: dict[tuple[str, str | None], int]) -> list[ScoredFunction]:
        scored_functions = []
        for (name, class_name), hop in min_hops.items():
            node_data = self.index.graph.nodes[(name, class_name)]
            in_degree = self.index.graph.in_degree((name, class_name))
            seeds_call_count = seed_connections.get((name, class_name), 0)

            score = seeds_call_count * ((decay_factor ** hop) / math.log(2 + in_degree))

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
