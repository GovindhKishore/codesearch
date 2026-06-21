import typer
from pathlib import Path
import hashlib, json
import os
from codesearch.parsing.parser import CodebaseParser
from codesearch.indexing.bm25_index import BM25Index
from codesearch.indexing.vector_index import VectorIndex
from codesearch.indexing.graph_index import GraphIndex
from codesearch.retrieval.bm25_retriever import BM25Retriever
from codesearch.retrieval.vector_retriever import VectorRetriever
from codesearch.retrieval.graph_retriever import GraphRetriever
from codesearch.pipeline.fusion import Fuser
from codesearch.pipeline.reranker import Reranker
from codesearch.providers.gemini import GeminiProvider
from codesearch.providers.ollama import OllamaProvider
from codesearch import config

REGISTRY_PATH = Path.home() / ".codesearch" / "registry.json"
BM25_DIR = Path.home() / ".codesearch" / "bm25"
GRAPH_DIR = Path.home() / ".codesearch" / "graphs"
CHROMA_DIR = Path.home() / ".codesearch" / "chroma"

VALID_KEY_PROVIDERS = {"gemini"}

def compute_project_hash(folder: Path) -> str:
    return hashlib.sha256(folder.as_posix().encode("utf-8")).hexdigest()

def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        typer.echo(f"Warning: {REGISTRY_PATH} is corrupted and could not be read. Treating as empty.")
        return {}

def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

def get_file_mtimes(folder: Path) -> dict[str, float]:
    skip_dirs = {
        "__pycache__", ".git", "venv", ".venv", "node_modules", "dist", "build",
        ".eggs", "egg-info", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
        ".tox",
    }

    file_mtimes: dict[str, float] = {}

    for dirpath, sub_dirnames, filenames in os.walk(folder):
        sub_dirnames[:] = [d for d in sub_dirnames if d not in skip_dirs]

        for filename in filenames:
            if filename.endswith(".py"):
                filepath = Path(dirpath) / filename
                file_mtimes[filepath.as_posix()] = filepath.stat().st_mtime

    return file_mtimes

def get_indexes(folder: Path, no_index: bool) -> tuple[BM25Index, VectorIndex, GraphIndex]:
    if not no_index:
        project_hash = compute_project_hash(folder)

        registry = load_registry()
        if project_hash not in registry:
            typer.echo(f"{folder} is not indexed. Run 'codesearch index {folder}' first.")
            raise typer.Exit(code=1)

        indexed_project = registry[project_hash]

        current_mtimes = get_file_mtimes(folder)
        if current_mtimes != indexed_project["file_mtimes"]:
            typer.echo(f"Files in {folder} have changed since last index.")
            proceed = typer.confirm("Search anyway with the existing index?", default=True)
            if not proceed:
                typer.echo(f"Run 'codesearch reindex {folder}' to update.")
                raise typer.Exit(code=0)

        try:
            bm25_index = BM25Index.load(BM25_DIR / f"{project_hash}.pkl")
            vector_index = VectorIndex.load(
                collection_name=project_hash,
                persist_path=CHROMA_DIR,
                model_name=indexed_project["vector_index_embedding_model"],
            )
            graph_index = GraphIndex.load(GRAPH_DIR / f"{project_hash}.pkl")
        except Exception as e:
            typer.echo(f"Failed to load indexes: {e}")
            raise typer.Exit(code=1)

        return bm25_index, vector_index, graph_index

    else:
        parser = CodebaseParser()
        functions = parser.parse_dir(folder)

        if not functions:
            typer.echo(f"No functions found in {folder}. Nothing to index.")
            raise typer.Exit(code=1)

        try:
            bm25_index = BM25Index.build(functions)
            vector_index = VectorIndex.build(functions, persist=False)
            graph_index = GraphIndex.build(functions)
        except Exception as e:
            typer.echo(f"Failed to build indexes: {e}")
            raise typer.Exit(code=1)

        return bm25_index, vector_index, graph_index



app = typer.Typer()

@app.command()
def index(folder: Path):
    folder = folder.resolve()
    project_hash = compute_project_hash(folder)

    registry = load_registry()
    if project_hash in registry:
        typer.echo(f"Already indexed: {folder}. Run 'codesearch reindex' to rebuild.")
        typer.Exit(code=0)

    parser = CodebaseParser()
    functions = parser.parse_dir(folder)

    if not functions:
        typer.echo(f"No functions found in {folder}. Nothing to index.")
        raise typer.Exit(code=1)

    typer.echo(f"Parsed {len(functions)} functions. Building indexes...")

    try:
        bm25_index = BM25Index.build(functions)
        vector_index = VectorIndex.build(
            functions,
            persist=True,
            collection_name=project_hash,
            persist_path=CHROMA_DIR,
        )
        graph_index = GraphIndex.build(functions)

        bm25_index.save(BM25_DIR / f"{project_hash}.pkl")
        graph_index.save(GRAPH_DIR / f"{project_hash}.pkl")
    except Exception as e:
        typer.echo(f"Failed to build indexes: {e}")
        raise typer.Exit(code=1)

    file_mtimes = get_file_mtimes(folder)
    registry[project_hash] = {
        "folder": folder.as_posix(),
        "vector_index_embedding_model": vector_index.model_name,
        "file_mtimes": file_mtimes,
    }

    try:
        save_registry(registry)
    except (OSError, TypeError) as e:
        typer.echo(f"Indexes were built but failed to update registry: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"Indexed and saved {len(functions)} functions from {folder}.")


@app.command()
def search(
        query: str,
        folder: Path,
        no_index: bool = False,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        structural_weight: float = 0.3,
        max_hop: int = 2,
        decay_factor: float = 0.5,
        provider: str | None = None,
        top_n: int = 10,
    ):
    folder = folder.resolve()

    bm25_index, vector_index, graph_index = get_indexes(folder, no_index=no_index)

    bm25_retriever = BM25Retriever(bm25_index)
    vector_retriever = VectorRetriever(vector_index)
    graph_retriever = GraphRetriever(graph_index)

    bm25_results = bm25_retriever.search(query)
    try:
        vector_results = vector_retriever.search(query)
    except Exception as e:
        typer.echo(f"Vector search failed ({e}), continuing with keyword + structural search only.")
        vector_results = []

    seeds = bm25_results[:10] + vector_results[:10]
    graph_results = graph_retriever.search(seeds, max_hop=max_hop, decay_factor=decay_factor)

    fuser = Fuser(
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        structural_weight=structural_weight,
    )
    fused_results = fuser.fuse(bm25_results, vector_results, graph_results)
    rerank_candidates = fused_results[:20]

    if provider is None:
        typer.echo("No provider specified. Skipping reranking, showing fused results.")
        final_results = rerank_candidates[:top_n]
    elif provider == "gemini":
        api_key = config.get_api_key("gemini")
        if api_key is None:
            typer.echo("No Gemini API key found. Skipping reranking, showing fused results.")
            final_results = rerank_candidates[:top_n]
        else:
            gemini_provider = GeminiProvider(api_key=api_key)
            reranker = Reranker(provider=gemini_provider, top_n=top_n)
            final_results = reranker.rerank(query, rerank_candidates)
    elif provider == "ollama":
        ollama_provider = OllamaProvider()
        reranker = Reranker(provider=ollama_provider, top_n=top_n)
        final_results = reranker.rerank(query, rerank_candidates)
    else:
        typer.echo(f"Unknown provider: {provider}. Skipping reranking, showing fused results.")
        final_results = rerank_candidates[:top_n]


    if not final_results:
        typer.echo("No relevant results found.")
        return

    typer.echo(f"\nResults for: \"{query}\"\n")
    for i, result in enumerate(final_results, start=1):
        func = result.function
        typer.echo(f"{i}. {func.name}    {func.file.as_posix()}:{func.line}")
        if result.explanation:
            typer.echo(f"   {result.explanation}")
        typer.echo()



@app.command()
def set_api_key(provider: str, api_key: str):
    if provider not in VALID_KEY_PROVIDERS:
        typer.echo(f"Unknown provider: {provider}. Valid options: {', '.join(VALID_KEY_PROVIDERS)}")
        raise typer.Exit(code=1)

    if config.set_api_key(provider, api_key):
        typer.echo(f"API key saved for {provider}.")
    else:
        typer.echo(f"Failed to save API key for {provider}.")
        raise typer.Exit(code=1)

@app.command()
def get_api_key(provider: str):
    if provider not in VALID_KEY_PROVIDERS:
        typer.echo(f"Unknown provider: {provider}. Valid providers: {', '.join(VALID_KEY_PROVIDERS)}")
        raise typer.Exit(code=1)

    api_key = config.get_api_key(provider)
    if api_key is None:
        typer.echo(f"No API key configured for {provider}.")
    else:
        typer.echo(f"{provider} API key: {api_key}")

@app.command()
def clear_api_key(provider: str):
    if provider not in VALID_KEY_PROVIDERS:
        typer.echo(f"Unknown provider: {provider}. Valid providers: {', '.join(VALID_KEY_PROVIDERS)}")
        raise typer.Exit(code=1)

    if config.clear_api_key(provider):
        typer.echo(f"API key cleared for {provider}.")
    else:
        typer.echo(f"No API key was set for {provider}, or it could not be cleared.")

@app.command()
def clear():
    pass

@app.command()
def reindex(folder: Path):
    folder = folder.resolve()
    project_hash = compute_project_hash(folder)

    parser = CodebaseParser()
    functions = parser.parse_dir(folder)

    if not functions:
        typer.echo(f"No functions found in {folder}. Nothing to index.")
        raise typer.Exit(code=1)

    typer.echo(f"Parsed {len(functions)} functions. Building indexes...")

    try:
        bm25_index = BM25Index.build(functions)
        vector_index = VectorIndex.build(
            functions,
            persist=True,
            collection_name=project_hash,
            persist_path=CHROMA_DIR,
        )
        graph_index = GraphIndex.build(functions)

        bm25_index.save(BM25_DIR / f"{project_hash}.pkl")
        graph_index.save(GRAPH_DIR / f"{project_hash}.pkl")
    except Exception as e:
        typer.echo(f"Failed to build indexes: {e}")
        raise typer.Exit(code=1)

    file_mtimes = get_file_mtimes(folder)
    registry = load_registry()

    registry[project_hash] = {
        "folder": folder.as_posix(),
        "vector_index_embedding_model": vector_index.model_name,
        "file_mtimes": file_mtimes,
    }

    try:
        save_registry(registry)
    except (OSError, TypeError) as e:
        typer.echo(f"Indexes were built but failed to update registry: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"Indexed and saved {len(functions)} functions from {folder}.")


if __name__ == "__main__":
    app()