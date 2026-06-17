from dataclasses import dataclass
from codesearch.parsing.parser import FunctionInfo

@dataclass
class ScoredFunction:
    function: FunctionInfo
    score: float
    rank: int
    retriever: str