import typer
from pathlib import Path
import hashlib, json
import os
import time

from rich.console import Console
from rich.markup import escape
from codesearch.parsing.parser import CodebaseParser
from codesearch.indexing.bm25_index import BM25Index
from codesearch.indexing.vector_index import VectorIndex
from codesearch.indexing.graph_index import GraphIndex
from codesearch.retrieval.bm25_retriever import BM25Retriever
from codesearch.retrieval.types import ScoredFunction
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

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

console = Console(highlight=False, soft_wrap=True)
error_console = Console(stderr=True, highlight=False, soft_wrap=True)

def compute_project_hash(folder: Path) -> str:
    return hashlib.sha256(folder.as_posix().encode("utf-8")).hexdigest()

def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        error_console.print(f"[bold yellow]Warning: [/bold yellow] {REGISTRY_PATH} is corrupted and could not be read.Treating as empty.")
        return {}

def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

def get_file_mtimes(folder: Path, include_tests: bool) -> dict[str, float]:
    skip_dirs = {
        "__pycache__", ".git", "venv", ".venv", "node_modules", "dist", "build",
        ".eggs", "egg-info", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
        ".tox",
    }

    file_mtimes: dict[str, float] = {}

    for dirpath, sub_dirnames, filenames in os.walk(folder):
        sub_dirnames[:] = [
            d for d in sub_dirnames
            if (d not in skip_dirs and (include_tests or d not in {"tests", "test"}))
        ]

        for filename in filenames:
            if filename.endswith(".py"):
                if not include_tests and (filename.startswith("test_") or filename.endswith("_test.py")):
                    continue
                filepath = Path(dirpath) / filename
                file_mtimes[filepath.as_posix()] = filepath.stat().st_mtime

    return file_mtimes

def get_indexes(folder: Path, no_index: bool) -> tuple[BM25Index, VectorIndex, GraphIndex, float, float, float]:
    if not no_index:
        project_hash = compute_project_hash(folder)

        registry = load_registry()
        if project_hash not in registry:
            error_console.print(f"[bold yellow]Warning: [/bold yellow]{escape(str(folder))} is not indexed. Run 'codesearch index {folder}' first.")
            raise typer.Exit(code=1)

        indexed_project = registry[project_hash]
        include_tests = indexed_project["include_tests"]

        current_mtimes = get_file_mtimes(folder, include_tests=include_tests)
        if current_mtimes != indexed_project["file_mtimes"]:
            error_console.print(f"[bold yellow]Warning: [/bold yellow]Files in {escape(str(folder))} have changed since last index.")
            proceed = typer.confirm("Search anyway with the existing index?", default=True)
            if not proceed:
                console.print(f"Run 'codesearch reindex {escape(str(folder))}' to update.")
                raise typer.Exit(code=0)

        try:
            tb0 = time.perf_counter()
            bm25_index = BM25Index.load(BM25_DIR / f"{project_hash}.pkl")
            tb1 = time.perf_counter()

            tv0 = time.perf_counter()
            vector_index = VectorIndex.load(
                collection_name=project_hash,
                persist_path=CHROMA_DIR,
                model_name=indexed_project["vector_index_embedding_model"],
            )
            tv1 = time.perf_counter()

            tg0 = time.perf_counter()
            graph_index = GraphIndex.load(GRAPH_DIR / f"{project_hash}.pkl")
            tg1 = time.perf_counter()

        except Exception as e:
            error_console.print(f"[b red]Error:[/b red] Failed to load indexes: {e}")
            raise typer.Exit(code=1)

        return bm25_index, vector_index, graph_index, tb1 - tb0, tv1 - tv0, tg1 - tg0

    else:
        parser = CodebaseParser()
        functions = parser.parse_dir(folder)

        if not functions:
            console.print(f"[bold yellow]Warning: [/bold yellow]No functions found in {escape(str(folder))}. Nothing to index.")
            raise typer.Exit(code=1)

        try:
            ttb0 = time.perf_counter()
            bm25_index = BM25Index.build(functions)
            ttb1 = time.perf_counter()

            ttv0 = time.perf_counter()
            vector_index = VectorIndex.build(functions, persist=False)
            ttv1 = time.perf_counter()

            ttg0 = time.perf_counter()
            graph_index = GraphIndex.build(functions)
            ttg1 = time.perf_counter()

        except Exception as e:
            error_console.print(f"[b red]Error:[/b red] Failed to build indexes: {e}")
            raise typer.Exit(code=1)

        return bm25_index, vector_index, graph_index, ttb1 - ttb0, ttv1 - ttv0, ttg1 - ttg0


def search_helper(bm25_retriever: BM25Retriever,
                  vector_retriever: VectorRetriever,
                  graph_retriever: GraphRetriever,
                  fuser: Fuser,
                  query: str,
                  max_hop: int,
                  decay_factor: float,
                  timings : bool = False) -> list[ScoredFunction]:
    tbs0 = time.perf_counter()
    bm25_results = bm25_retriever.search(query)
    tbs1 = time.perf_counter()

    tvs0 = 0.0
    tvs1 = 0.0
    try:
        tvs0 = time.perf_counter()
        vector_results = vector_retriever.search(query)
        tvs1 = time.perf_counter()
    except Exception as e:
        error_console.print(f"[bold yellow]Warning: [/bold yellow]Vector search failed ({e}), continuing with keyword + structural search only.")
        vector_results = []

    seeds = bm25_results[:5] + vector_results[:5]

    tgs0 = time.perf_counter()
    graph_results = graph_retriever.search(seeds, max_hop=max_hop, decay_factor=decay_factor)
    tgs1 = time.perf_counter()

    if timings:
        console.print()
        console.print(f"[dim]BM25 search took [cyan]{tbs1 - tbs0:.2f}[/cyan] seconds.[/dim]")
        if tvs1 != 0.0:
            console.print(f"[dim]Vector search took [cyan]{tvs1 - tvs0:.2f}[/cyan] seconds.[/dim]")
        console.print(f"[dim]Graph search took [cyan]{tgs1 - tgs0:.2f}[/cyan] seconds.[/dim]")

    fused_results = fuser.fuse(bm25_results, vector_results, graph_results)
    rerank_candidates = fused_results[:15]

    return rerank_candidates


def rerank_results(query: str, candidates: list[ScoredFunction], reranker: Reranker, skip_reason: str, top_n: int, timings: bool) -> list[ScoredFunction]:
    if reranker is None:
        console.print()
        error_console.print(f"[bold yellow]Warning: [/bold yellow]{skip_reason} Skipping reranking, showing fused results.")
        return candidates[:top_n]

    tr0 = time.perf_counter()
    final = reranker.rerank(query, candidates)
    tr1 = time.perf_counter()

    if timings:
        console.print(f"[dim]Reranking took [cyan]{tr1 - tr0:.2f}[/cyan] seconds.[/dim]")
    return final


def print_results(query: str, final_results: list[ScoredFunction]) -> None:
    console.print(f"\nResults for: \"{query}\"\n")
    for i, result in enumerate(final_results, start=1):
        func = result.function
        console.print(f"[cyan]{i}[/cyan]. {func.name}    {escape(func.file.as_posix())}:{func.line}")
        if result.explanation:
            console.print(f"   {result.explanation}")
        console.print()


app = typer.Typer()


@app.command()
def index(folder: Path, timings: bool = False, include_tests: bool = False):
    folder = folder.resolve()
    project_hash = compute_project_hash(folder)

    registry = load_registry()
    if project_hash in registry:
        console.print(f"Already indexed: {escape(str(folder))}. Run 'codesearch reindex' to rebuild.")
        raise typer.Exit(code=0)

    parser = CodebaseParser(include_tests=include_tests)

    tp0 = time.perf_counter()
    functions = parser.parse_dir(folder)
    tp1 = time.perf_counter()

    if not functions:
        console.print()
        error_console.print(f"[bold yellow]Warning: [/bold yellow]No functions found in {escape(str(folder))}. Nothing to index.")
        raise typer.Exit(code=1)

    console.print()
    console.print(f"Parsed [cyan]{len(functions)}[/cyan] functions. Building indexes...")
    if timings:
        console.print(f"[dim]Parsing took [cyan]{tp1 - tp0:.2f}[/cyan] seconds.[/dim]")

    try:
        tb0 = time.perf_counter()
        bm25_index = BM25Index.build(functions)
        tb1 = time.perf_counter()

        tv0 = time.perf_counter()
        vector_index = VectorIndex.build(
            functions,
            persist=True,
            collection_name=project_hash,
            persist_path=CHROMA_DIR,
        )
        tv1 = time.perf_counter()

        tg0 = time.perf_counter()
        graph_index = GraphIndex.build(functions)
        tg1 = time.perf_counter()

        if timings:
            console.print(f"[dim]BM25 index built in [cyan]{tb1 - tb0:.2f}[/cyan] seconds.[/dim]")
            console.print(f"[dim]Vector index built in [cyan]{tv1 - tv0:.2f}[/cyan] seconds.[/dim]")
            console.print(f"[dim]Graph index built in [cyan]{tg1 - tg0:.2f}[/cyan] seconds.[/dim]")

        bm25_index.save(BM25_DIR / f"{project_hash}.pkl")
        graph_index.save(GRAPH_DIR / f"{project_hash}.pkl")
    except Exception as e:
        error_console.print(f"[b red]Error: [/b red]Failed to build indexes: {e}")
        raise typer.Exit(code=1)

    file_mtimes = get_file_mtimes(folder, include_tests)
    registry[project_hash] = {
        "folder": folder.as_posix(),
        "vector_index_embedding_model": vector_index.model_name,
        "file_mtimes": file_mtimes,
        "include_tests": include_tests,
    }

    try:
        save_registry(registry)
    except (OSError, TypeError) as e:
        error_console.print(f"[b red]Error: [/b red]Indexes were built but failed to update registry: {e}")
        raise typer.Exit(code=1)

    console.print(f"[b green]Success: [/b green]Indexed and saved [cyan]{len(functions)}[/cyan] functions from {escape(str(folder))}.")


@app.command()
def search(
        folder: Path,
        no_index: bool = False,
        bm25_weight: float = 0.7,
        vector_weight: float = 1.0,
        structural_weight: float = 0.1,
        max_hop: int = 2,
        decay_factor: float = 0.5,
        provider: str | None = None,
        top_n: int = 10,
        timings: bool = False,
    ):
    folder = folder.resolve()

    bm25_index, vector_index, graph_index, blt, vlt, glt = get_indexes(folder, no_index=no_index)

    if timings:
        console.print()
        console.print(f"[dim]BM25 index load took [cyan]{blt:.2f}[/cyan] seconds.[/dim]")
        console.print(f"[dim]Vector index load took [cyan]{vlt:.2f}[/cyan] seconds.[/dim]")
        console.print(f"[dim]Graph index load took [cyan]{glt:.2f}[cyan] seconds.[/dim]")

    bm25_retriever = BM25Retriever(bm25_index)
    vector_retriever = VectorRetriever(vector_index)
    graph_retriever = GraphRetriever(graph_index)
    fuser = Fuser(
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        structural_weight=structural_weight,
    )

    reranker, skip_reason = None, None

    if provider is None:
        skip_reason = "No provider specified."
    elif provider == "gemini":
        api_key = config.get_api_key("gemini")
        if api_key is None:
            skip_reason = "No Gemini API key found."
        else:
            reranker = Reranker(provider=GeminiProvider(api_key=api_key), top_n=top_n)
    elif provider == "ollama":
        reranker = Reranker(provider=OllamaProvider(), top_n=top_n)
    else:
        skip_reason = f"Unknown provider: {provider}."

    while True:
        try:
            console.print()
            query = typer.prompt("Query").strip()
        except typer.Abort:
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        rerank_candidates = search_helper(bm25_retriever, vector_retriever, graph_retriever, fuser, query, max_hop, decay_factor, timings)

        final_results = rerank_results(query, rerank_candidates, reranker, skip_reason, top_n, timings)

        if not final_results:
            console.print()
            console.print("No relevant results found.")
            continue

        print_results(query, final_results)





@app.command()
def set_api_key(provider: str, api_key: str):
    if provider not in VALID_KEY_PROVIDERS:
        error_console.print(f"[b red]Error: [/b red]Unknown provider: {provider}. Valid options: {', '.join(VALID_KEY_PROVIDERS)}")
        raise typer.Exit(code=1)

    if config.set_api_key(provider, api_key):
        console.print(f"[b green]Success: [/b green]API key saved for {provider}.")
    else:
        error_console.print(f"[b red]Error:[/b red] Failed to save API key for {provider}.")
        raise typer.Exit(code=1)


@app.command()
def get_api_key(provider: str):
    if provider not in VALID_KEY_PROVIDERS:
        error_console.print(f"[b red]Error: [/b red]Unknown provider: {provider}. Valid providers: {', '.join(VALID_KEY_PROVIDERS)}")
        raise typer.Exit(code=1)

    api_key = config.get_api_key(provider)
    if api_key is None:
        error_console.print(f"No API key configured for {provider}.")
    else:
        masked = "..." + api_key[-4: ] if len(api_key) > 8 else "..."
        console.print(f"[b green]Success: [/b green]{provider} API key Found: {masked}")


@app.command()
def clear_api_key(provider: str):
    if provider not in VALID_KEY_PROVIDERS:
        error_console.print(f"[b red]Error: [/b red]Unknown provider: {provider}. Valid providers: {', '.join(VALID_KEY_PROVIDERS)}")
        raise typer.Exit(code=1)

    if config.clear_api_key(provider):
        console.print(f"[b green]Success: [/b green]API key cleared for {provider}.")
    else:
        error_console.print(f"[b yellow]Warning: [/b yellow]No API key was set for {provider}, or it could not be cleared.")


@app.command()
def clear(folder: Path | None = None, all_items: bool = False):
    if not folder and not all_items:
        error_console.print("[b yellow]Warning: [/ b yellow]Provide a folder to clear, or use --all-items to clear everything.")
        raise typer.Exit(code=1)
    if folder and all_items:
        error_console.print("[b yellow]Warning: [/b yellow]Provide either a folder or --all-items.")
        raise typer.Exit(code=1)

    registry = load_registry()

    if all_items:
        hashes_to_clear = list(registry.keys())
    else:
        folder = folder.resolve()
        project_hash = compute_project_hash(folder)
        if project_hash not in registry:
            error_console.print(f"[b yellow]Warning: [/b yellow]{escape(str(folder))} is not indexed. Nothing to clear.")
            raise typer.Exit(code=0)
        hashes_to_clear = [project_hash]

    for project_hash in hashes_to_clear:
        (BM25_DIR / f"{project_hash}.pkl").unlink(missing_ok=True)
        VectorIndex.delete(collection_name=project_hash, persist_path=CHROMA_DIR)
        (GRAPH_DIR / f"{project_hash}.pkl").unlink(missing_ok=True)
        del registry[project_hash]

    try:
        save_registry(registry)
    except (OSError, TypeError) as e:
        error_console.print(f"[b red]Error: [/b red]Indexes were cleared but failed to update registry: {e}")
        raise typer.Exit(code=1)

    console.print(f"[b green]Success: [/b green]Cleared [cyan]{len(hashes_to_clear)}[/cyan] project(s).")


@app.command()
def reindex(folder: Path, timings: bool = False, include_tests: bool = False):
    folder = folder.resolve()
    project_hash = compute_project_hash(folder)

    parser = CodebaseParser(include_tests=include_tests)

    tp0 = time.perf_counter()
    functions = parser.parse_dir(folder)
    tp1 = time.perf_counter()

    if not functions:
        error_console.print(f"[bold yellow]Warning: [/bold yellow]No functions found in {escape(str(folder))}. Nothing to index.")
        raise typer.Exit(code=1)


    console.print(f"Parsed [cyan]{len(functions)}[/cyan] functions. Building indexes...")
    if timings:
        console.print(f"[dim]Parsing took [cyan]{tp1 - tp0:.2f}[/cyan] seconds.[/dim]")

    try:
        tb0 = time.perf_counter()
        bm25_index = BM25Index.build(functions)
        tb1 = time.perf_counter()

        tv0 = time.perf_counter()
        vector_index = VectorIndex.build(
            functions,
            persist=True,
            collection_name=project_hash,
            persist_path=CHROMA_DIR,
        )
        tv1 = time.perf_counter()

        tg0 = time.perf_counter()
        graph_index = GraphIndex.build(functions)
        tg1 = time.perf_counter()

        if timings:
            console.print(f"[dim]BM25 index built in [cyan]{tb1 - tb0:.2f}[/cyan] seconds.[/dim]")
            console.print(f"[dim]Vector index built in [cyan]{tv1 - tv0:.2f}[/cyan] seconds.[/dim]")
            console.print(f"[dim]Graph index built in [cyan]{tg1 - tg0:.2f}[/cyan] seconds.[/dim]")

        bm25_index.save(BM25_DIR / f"{project_hash}.pkl")
        graph_index.save(GRAPH_DIR / f"{project_hash}.pkl")
    except Exception as e:
        error_console.print(f"[b red]Error: [/b red]Failed to build indexes: {e}")
        raise typer.Exit(code=1)

    file_mtimes = get_file_mtimes(folder, include_tests)
    registry = load_registry()

    registry[project_hash] = {
        "folder": folder.as_posix(),
        "vector_index_embedding_model": vector_index.model_name,
        "file_mtimes": file_mtimes,
        "include_tests": include_tests,
    }

    try:
        save_registry(registry)
    except (OSError, TypeError) as e:
        error_console.print(f"[b red]Error: [/b red]Indexes were built but failed to update registry: {e}")
        raise typer.Exit(code=1)

    console.print(f"[b green]Success: [/b green]Indexed and saved [cyan]{len(functions)}[/cyan] functions from {escape(str(folder))}.")


if __name__ == "__main__":
    app()