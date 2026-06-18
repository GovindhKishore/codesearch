from pathlib import Path
from dataclasses import dataclass
import pickle
import networkx as nx
from codesearch.parsing.parser import FunctionInfo

@dataclass
class GraphIndex:
    graph: nx.DiGraph

    @classmethod
    def build(cls, functions: list[FunctionInfo]) -> "GraphIndex":
        if not functions:
            raise ValueError("Cannot build GraphIndex with an empty list of functions.")

        graph = nx.DiGraph()

        for func in functions:
            graph.add_node(func.name, file=func.file.as_posix(), line=func.line, doc_string=func.doc_string)

        for func in functions:
            for callee in func.callees:
                if graph.has_node(callee):
                    graph.add_edge(func.name, callee)

        return cls(graph=graph)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "GraphIndex":
        path = Path(path)
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Pickle at {path} did not contain a GraphIndex")
        return obj