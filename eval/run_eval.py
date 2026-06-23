import json
from pathlib import Path

from codesearch.cli import get_indexes
from codesearch.retrieval.bm25_retriever import BM25Retriever
from codesearch.retrieval.vector_retriever import VectorRetriever
from codesearch.retrieval.graph_retriever import GraphRetriever
from codesearch.pipeline.fusion import Fuser
import typer


EVAL_DIR = Path(__file__).parent
TARGET_FOLDER = Path(r"C:\Users\govin\PycharmProjects\codesearch\eval\sklearn-eval-core")
QUERIES_PATH = EVAL_DIR / "queries.json"


def load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_file_path(file_path: str) -> str:
    parts = Path(file_path).as_posix()
    idx = parts.find("sklearn-eval-core")
    if idx == -1:
        return parts
    return parts[idx:]


def setup() -> tuple[BM25Retriever, VectorRetriever, GraphRetriever]:
    try:
        bm25_index, vector_index, graph_index = get_indexes(TARGET_FOLDER, no_index=False)
    except typer.Exit as e:
        if e.exit_code == 0:
            raise RuntimeError("Evaluation cancelled: index is stale and user declined to proceed.")
        raise RuntimeError(f"{TARGET_FOLDER} is not indexed. Run 'codesearch index {TARGET_FOLDER}' first.")

    bm25_retriever = BM25Retriever(bm25_index)
    vector_retriever = VectorRetriever(vector_index)
    graph_retriever = GraphRetriever(graph_index)

    return bm25_retriever, vector_retriever, graph_retriever


def run_queries(
    queries: list[dict],
    bm25_retriever: BM25Retriever,
    vector_retriever: VectorRetriever,
    graph_retriever: GraphRetriever,
    fuser: Fuser,
) -> list[dict]:
    all_results = []

    for q in queries:
        query_text = q["query"]

        bm25_results = bm25_retriever.search(query_text)

        try:
            vector_results = vector_retriever.search(query_text)
        except Exception:
            vector_results = []

        seeds = bm25_results[:10] + vector_results[:10]
        structural_results = graph_retriever.search(seeds)
        structural_results.sort(key=lambda r: r.score or 0.0, reverse=True)

        fused_results = fuser.fuse(bm25_results, vector_results, structural_results)

        all_results.append({
            "bm25": bm25_results,
            "vector": vector_results,
            "structural": structural_results,
            "fused": fused_results,
        })

    return all_results


def compute_metrics(
    results: list,
    ground_truth_keys: set[tuple[str, int]],
) -> dict[str, float]:
    def result_key(r):
        return (normalize_file_path(r.function.file.as_posix()), r.function.line)

    top5_keys = [result_key(r) for r in results[:5]]
    top10_keys = [result_key(r) for r in results[:10]]

    # MRR@5
    mrr = 0.0
    for i, key in enumerate(top5_keys, start=1):
        if key in ground_truth_keys:
            mrr = 1 / i
            break

    # Recall@10
    hits_in_10 = sum(1 for key in top10_keys if key in ground_truth_keys)
    recall = hits_in_10 / len(ground_truth_keys)

    # Precision@5
    if not results:
        precision = 0.0
    else:
        hits_in_5 = sum(1 for key in top5_keys if key in ground_truth_keys)
        precision = hits_in_5 / min(5, len(results))

    return {"mrr": mrr, "recall": recall, "precision": precision}


def evaluate(queries: list[dict], all_results: list[dict]) -> dict[str, dict[str, float]]:
    rows = ["bm25", "vector", "structural", "fused"]
    totals = {row: {"mrr": 0.0, "recall": 0.0, "precision": 0.0} for row in rows}

    for q, results_for_query in zip(queries, all_results):
        ground_truth_keys = set(
            zip(q["files"], q["lines"])
        )

        for row in rows:
            metrics = compute_metrics(results_for_query[row], ground_truth_keys)
            for metric, value in metrics.items():
                totals[row][metric] += value

    n = len(queries)
    return {
        row: {metric: round(total / n, 4) for metric, total in metrics.items()}
        for row, metrics in totals.items()
    }


def print_table(results: dict[str, dict[str, float]]) -> None:
    row_labels = {
        "bm25": "BM25 only",
        "vector": "Semantic only",
        "structural": "Structural only",
        "fused": "Hybrid fusion",
    }

    header = f"{'Retriever':<20} {'MRR@5':>8} {'Recall@10':>10} {'Precision@5':>12}"
    print("\n" + header)
    print("-" * len(header))

    for row, label in row_labels.items():
        m = results[row]
        print(f"{label:<20} {m['mrr']:>8.4f} {m['recall']:>10.4f} {m['precision']:>12.4f}")
    print()


def main():
    queries = load_queries(QUERIES_PATH)
    bm25_retriever, vector_retriever, graph_retriever = setup()
    fuser = Fuser(bm25_weight=0.8, vector_weight=1.0, structural_weight=0.0)
    all_results = run_queries(queries, bm25_retriever, vector_retriever, graph_retriever, fuser)
    final_metrics = evaluate(queries, all_results)
    print_table(final_metrics)

if __name__ == "__main__":
    main()