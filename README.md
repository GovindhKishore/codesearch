# codesearch

A hybrid lexical + semantic + structural code search CLI tool for Python codebases. Point it at any folder and search using natural language from your terminal.

---

## Why This Exists

Existing code search approaches have a fundamental tension:

- **Lexical search (grep, ripgrep)** fails when your vocabulary doesn't match the code's naming conventions.
- **Semantic search (embeddings)** surfaces functions that are plausibly relevant but structurally irrelevant - dead code, deprecated modules, things nothing ever calls.
- **Neither understands codebase architecture** - a function can be semantically perfect but useless if it's architecturally isolated.

`codesearch` fuses all three signals into one ranked result.

---

## How It Works

**Three retrievers run in parallel:**

1. **Lexical (BM25)** - tokenizes the query and scores against a composite document per function (name, docstring, params, return type, caller/callee names, module path). Fast, exact-keyword recall.

2. **Semantic (ChromaDB + sentence-transformers)** - embeds the query using `all-MiniLM-L6-v2` and does approximate nearest-neighbor lookup. Handles vocabulary mismatch.

3. **Structural (NetworkX call graph)** - builds an AST-derived call graph of the codebase. Seeds a BFS from the top BM25/semantic results and surfaces architecturally adjacent functions that neither lexical nor semantic search would find.

**Fusion:** All three retrievers return ranked lists. Reciprocal Rank Fusion (RRF) merges them without needing to normalize across incompatible score scales:

```
RRF_score(function) = Σ weight_i / (60 + rank_in_retriever_i)
```

Each retriever's weight is independently tunable (`--bm25-weight`, `--vector-weight`, `--structural-weight`).

**Reranking (optional):** The fused top-20 candidates are sent to an LLM (Gemini or Ollama) with full source code per function. The LLM reorders them and writes a one-sentence explanation per result. Falls back to fused results if no API key is configured.

---

## Installation

```bash
git clone https://github.com/GovindhKishore/codesearch.git
cd codesearch
pip install -e .
```

Requires Python 3.10+.

---

## Usage

```bash
# Index a codebase (builds BM25, vector, and call graph indexes)
codesearch index /path/to/your/project

# Search it
codesearch search "functions that handle HTTP request validation" /path/to/your/project

# Tune retrieval weights for jargon-heavy or framework-style codebases
codesearch search "parse incoming payload" /path/to/your/project \
    --vector-weight 1.5 --structural-weight 0.0

# Use a local LLM for reranking (no data leaves your machine)
codesearch search "authenticate user token" /path/to/your/project --provider ollama

# Force a full rebuild
codesearch reindex /path/to/your/project

# Clear indexes
codesearch clear /path/to/your/project
```

### API key setup (for Gemini reranking)

```bash
codesearch set-api-key gemini
```

Keys are stored in your OS's native credential store (Windows Credential Manager, macOS Keychain, Linux Secret Service) - never in plaintext files.

---

## Tech Stack

| Layer | Technology |
|---|---|
| AST parsing | Python `ast` module |
| Lexical retrieval | `rank-bm25` (BM25Okapi) |
| Vector store | ChromaDB (cosine HNSW, persistent) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Call graph | NetworkX DiGraph + BFS |
| Fusion | Reciprocal Rank Fusion (RRF) |
| LLM reranking | Google Gemini or Ollama (local) |
| CLI | Typer |
| Credential storage | `keyring` |
| Packaging | Hatchling / pyproject.toml |

---

## Evaluation

The project includes an evaluation framework in `eval/` that measures retrieval quality using standard IR metrics across four configurations:

| Configuration | MRR@5 | Recall@10 | Precision@5 |
|---|---:|---:|---:|
| BM25 only | 0.2844 | 0.4833 | 0.0800 |
| Semantic only | 0.4344 | 0.5667 | 0.1217 |
| Structural only | 0.0167 | 0.0333 | 0.0067 |
| Hybrid fusion | 0.4278 | 0.6667 | 0.1200 |

*Numbers pending - evaluation running against a hand-labeled benchmark of 30 queries on scikit-learn's `model_selection` and `preprocessing` subpackages (~1,000+ indexed functions).*

The ablation table is the project's central claim: each retriever's independent contribution is measured, not just the final fused result.

---

## Known Limitations

**Structural retrieval (call graph):** The graph indexes functions by bare name, without class or file context. In codebases where many classes share same method names, node collisions occur - two functions with the same name share one graph node, causing phantom edges. This degrades structural retrieval quality for those codebases. Use `--structural-weight 0.0` to disable it in that case.

A v2 fix (class-aware node keys + scoped caller resolution) is planned once the evaluation framework exists to measure the before/after improvement.

---

## Project Structure

```
codesearch/
├── codesearch/
│   ├── parsing/parser.py         # AST parsing, FunctionInfo, composite doc building
│   ├── indexing/
│   │   ├── bm25_index.py         # BM25Okapi index, tokenizer
│   │   ├── vector_index.py       # ChromaDB collection management
│   │   └── graph_index.py        # NetworkX DiGraph construction
│   ├── retrieval/
│   │   ├── types.py              # ScoredFunction dataclass
│   │   ├── bm25_retriever.py
│   │   ├── vector_retriever.py
│   │   └── graph_retriever.py    # Multi-source BFS, hop-decay scoring
│   ├── pipeline/
│   │   ├── fusion.py             # Fuser dataclass, RRF
│   │   └── reranker.py           # LLM reranking, source extraction, response parsing
│   ├── providers/
│   │   ├── base.py               # BaseProvider ABC
│   │   ├── gemini.py
│   │   └── ollama.py
│   ├── config.py                 # keyring-backed credential management
│   └── cli.py                    # Typer CLI
└── eval/
    ├── queries.json              # 30 hand-labeled ground-truth queries
    └── run_eval.py               # MRR@5, Recall@10, Precision@5 harness
```

---

## Roadmap

- [ ] Finalize ablation table with real numbers
- [ ] PyPI publication (rename from `codesearch` working title first)
- [ ] Class-aware graph node keys (v2 structural retrieval fix)
- [ ] Support for languages beyond Python (JavaScript/TypeScript via tree-sitter)
