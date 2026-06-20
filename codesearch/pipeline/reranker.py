import json
import random
from pathlib import Path
import ast

from codesearch.providers.base import BaseProvider
from codesearch.retrieval.types import ScoredFunction

def extract_function_source(file: Path, line: int) -> str | None:
    try:
        source = file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, ValueError, UnicodeDecodeError, OSError):
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == line:
            return ast.get_source_segment(source, node)

    return None

class Reranker:
    def __init__(self, provider: BaseProvider, top_n: int = 10):
        self.provider = provider
        self.top_n = top_n

    def rerank(self, query: str, candidates: list[ScoredFunction]) -> list[ScoredFunction]:
        fallback = candidates[: self.top_n]

        prompt = self._build_prompt(query, candidates)
        response = self.provider.generate(prompt)

        try:
            response_dict = json.loads(response)
            results = response_dict["results"]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return fallback

        candidates_by_file_line: dict[tuple[str, int], ScoredFunction] = {}
        candidates_by_file_name: dict[tuple[str, str], list[ScoredFunction]] = {}
        for candidate in candidates:
            key_candidate_by_file_line = (candidate.function.file.as_posix(), candidate.function.line)
            candidates_by_file_line[key_candidate_by_file_line] = candidate

            key_candidate_by_file_name = (candidate.function.file.as_posix(), candidate.function.name)
            if key_candidate_by_file_name not in candidates_by_file_name:
                candidates_by_file_name[key_candidate_by_file_name] = []
            candidates_by_file_name[key_candidate_by_file_name].append(candidate)


        matched: list[ScoredFunction] = []
        for result in results:
            if not isinstance(result, dict):
                continue

            file = result.get("file")
            if file is None:
                continue

            line = result.get("line")
            try:
                line = int(line)
            except (TypeError, ValueError):
                line = None

            match = None
            if line is not None:
                key = (file, line)
                match = candidates_by_file_line.get(key)

            if match is None:
                name = result.get("name")
                if name is None:
                    continue
                key = (file, name)
                match = candidates_by_file_name.get(key, [])
                if len(match) == 1:
                    match = match[0]
                else:
                    match = None

            if match is None:
                continue

            matched.append(ScoredFunction(
                function=match.function,
                score=None,
                rank=len(matched) + 1,
                retriever="reranked",
                explanation=result.get("explanation"),
            ))

        if not matched:
            return fallback
        return matched[: self.top_n]






    def _build_prompt(self, query: str, candidates: list[ScoredFunction]) -> str:
        shuffled = candidates.copy()
        random.shuffle(shuffled)

        candidate_blocks = []
        for candidate in shuffled:
            source = extract_function_source(candidate.function.file, candidate.function.line)
            if not source:
                continue

            doc_string = candidate.function.doc_string or "No docstring available."
            name = candidate.function.name
            file = candidate.function.file.as_posix()
            line = candidate.function.line

            block = f"""Function: {name}
            File: {file}
            Line: {line}
            Docstring: {doc_string}
            Source: {source}
            """

            candidate_blocks.append(block)

        candidates_text = "\n\n-----\n\n".join(candidate_blocks)

        schema_example = '{"results": [{"name": "...", "file": "...", "line": ..., "explanation": "one sentence why this is relevant"}]}'

        return f"""You are ranking code search results by relevance to a query.
Query: {query}

Rank the top {self.top_n} most relevant candidates below, ordered from most to least relevant. 
The order of objects in the "results" array IS the ranking - the first object is rank 1. 
Return ONLY valid JSON, no other text, in this exact format:
{schema_example}

CANDIDATES:

{candidates_text}"""
            
