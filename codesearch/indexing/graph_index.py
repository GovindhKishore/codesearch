from pathlib import Path
from dataclasses import dataclass
import pickle
import networkx as nx
from codesearch.parsing.parser import FunctionInfo
from collections import Counter

@dataclass
class GraphIndex:
    graph: nx.DiGraph

    @classmethod
    def build(cls, functions: list[FunctionInfo]) -> "GraphIndex":
        if not functions:
            raise ValueError("Cannot build GraphIndex with an empty list of functions.")

        graph = nx.DiGraph()

        func_counts = Counter((func.name, func.class_name) for func in functions)
        unique_functions = [f for f in functions if func_counts[(f.name, f.class_name)] == 1]
        unique_keys = {(f.name, f.class_name) for f in unique_functions}

        for func in unique_functions:
            key = (func.name, func.class_name)
            graph.add_node(key, file=func.file.as_posix(), line=func.line, doc_string=func.doc_string)

        for func in unique_functions:
            source_key = (func.name, func.class_name)
            for callee_name, callee_class in func.callees:
                if callee_class == "UNKNOWN":
                    continue
                target_key = (callee_name, callee_class)
                if target_key in unique_keys and target_key != source_key:
                    graph.add_edge(source_key, target_key)

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
            try:
                obj = pickle.load(f)
            except (EOFError, pickle.UnpicklingError, AttributeError, ModuleNotFoundError, FileNotFoundError) as e:
                raise ValueError(f"Index at {path} is corrupted or outdated. Try running 'codesearch reindex' to fix.") from e

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle at {path} did not contain a GraphIndex")
        return obj