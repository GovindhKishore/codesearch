import re
from dataclasses import dataclass
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from codesearch.parsing.parser import FunctionInfo

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "else", "this", "that", "these",
    "those", "it", "its", "self", "cls", "none", "true", "false",
}

_TOKEN_REGEX = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    if not text:
        return []

    raw_tokens = _TOKEN_REGEX.findall(text.lower())

    tokens = [
        token for token in raw_tokens
        if (token not in STOPWORDS) and
           (not token.isdigit()) and
           (len(token) > 1)
    ]

    return tokens

@dataclass
class BM25Index:
    bm25: BM25Okapi
    functions: list[FunctionInfo]
    tokenized_corpus: list[list[str]]

    @classmethod
    def build(cls, functions: list[FunctionInfo]) -> "BM25Index":
        if not functions:
            raise ValueError("Cannot build BM25Index with an empty list of functions.")

        tokenized_corpus = [tokenize(func.composite_doc) for func in functions]

        bm25 = BM25Okapi(tokenized_corpus)

        return cls(bm25=bm25, functions=functions, tokenized_corpus=tokenized_corpus)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        path = Path(path)

        with path.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle at {path} did not contain a BM25Index")

        return obj
