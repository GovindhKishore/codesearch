import json
from pathlib import Path
import math

from codesearch.cli import get_indexes
from codesearch.retrieval.bm25_retriever import BM25Retriever
from codesearch.retrieval.vector_retriever import VectorRetriever
from codesearch.retrieval.graph_retriever import GraphRetriever
from codesearch.pipeline.fusion import Fuser
import typer


EVAL_DIR = Path(__file__).parent
TARGET_FOLDER = Path(r"C:\Users\govin\PycharmProjects\codesearch\eval\sklearn-eval-core")
QUERIES_PATH = EVAL_DIR / "new_set_queries_hcse_eval.json"


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
        bm25_index, vector_index, graph_index, t0, t1, t2 = get_indexes(TARGET_FOLDER, no_index=False, include_tests=False)
    except typer.Exit as e:
        if e.exit_code == 0:
            raise RuntimeError("Evaluation cancelled")
        raise RuntimeError(f"{TARGET_FOLDER} is not indexed. Run 'codesearch index {TARGET_FOLDER}' first.")

    bm25_retriever = BM25Retriever(bm25_index)
    vector_retriever = VectorRetriever(vector_index)
    graph_retriever = GraphRetriever(graph_index)

    return bm25_retriever, vector_retriever, graph_retriever


GRAPH_MAX_HOP = 1
GRAPH_CAP = 10
SEED_COUNT = 10

def run_queries(
    queries: list[dict],
    bm25_retriever: BM25Retriever,
    vector_retriever: VectorRetriever,
    graph_retriever: GraphRetriever,
    fuser: Fuser,
    fuser_no_graph: Fuser,
) -> list[dict]:
    all_results = []

    for q in queries:
        query_text = q["query"]

        bm25_results = bm25_retriever.search(query_text)

        try:
            vector_results = vector_retriever.search(query_text)
        except Exception as e:
            print(f"Vector search failed for query: {query_text} ({e}). Continuing without it.")
            vector_results = []

        seeds = bm25_results[:SEED_COUNT] + vector_results[:SEED_COUNT]
        structural_results = graph_retriever.search(seeds, max_hop=GRAPH_MAX_HOP)

        fused_results = fuser.fuse(bm25_results, vector_results, structural_results[:GRAPH_CAP])
        fused_results_no_graph = fuser_no_graph.fuse(bm25_results, vector_results, [])

        print(f"\nQuery: {query_text}")
        for row_name, row_results in (
            ("bm25", bm25_results),
            ("vector", vector_results),
            ("structural", structural_results),
            ("fused", fused_results),
            ("fused_no_graph", fused_results_no_graph)
        ):
            print(f"  [{row_name}] ({len(row_results)} results)")
            for r in row_results[:10]:
                print(f"    {r.function.file.as_posix()}:{r.function.line}  {r.function.name}")

        all_results.append({
            "bm25": bm25_results,
            "vector": vector_results,
            "structural": structural_results,
            "fused": fused_results,
            "fused_no_graph": fused_results_no_graph,
        })

    return all_results


RECALL_KS = (10, 15, 20)


def compute_metrics(
    results: list,
    grades: dict[tuple[str, int], int],
) -> dict[str, float]:

    def result_key(r):
        return normalize_file_path(r.function.file.as_posix()), r.function.line

    # Drop repeated functions, keeping the first occurrence
    keys: list[tuple[str, int]] = []
    seen = set()
    for r in results:
        k = result_key(r)
        if k not in seen:
            seen.add(k)
            keys.append(k)

    # MRR@5
    mrr = 0.0
    for i, k in enumerate(keys[:5], start=1):
        if grades.get(k, 0) == 2:
            mrr = 1 / i
            break

    # Precision@5
    precision = sum(1 for k in keys[:5] if k in grades) / 5

    metrics = {"mrr": mrr, "precision": precision}

    # Recall@10/15/20
    total = len(grades)
    for k_cut in RECALL_KS:
        found = sum(1 for k in keys[:k_cut] if k in grades)
        metrics[f"recall{k_cut}"] = found / total if total else 0.0


    # nDCG@10 (linear gain, log2 discount)
    dcg = sum(
        grades.get(k, 0) / math.log2(i + 1)
        for i, k in enumerate(keys[:10], start=1)
    )
    ideal_grades = sorted(grades.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_grades, start=1))
    metrics["ndcg10"] = dcg / idcg if idcg > 0 else 0.0


    return metrics


def build_grades(q: dict) -> dict[tuple[str, int], int]:
    grades: dict[tuple[str, int], int] = {}
    for f, line in zip(q.get("related_files", []), q.get("related_lines", [])):
        grades[(normalize_file_path(f), line)] = 1
    for f, line in zip(q["relevant_files"], q["relevant_lines"]):
        grades[(normalize_file_path(f), line)] = 2  # exact overrides related
    return grades


METRIC_KEYS = ["mrr", "precision", "recall10", "recall15", "recall20", "ndcg10"]

def evaluate(queries: list[dict], all_results: list[dict]) -> dict[str, dict[str, float]]:
    rows = ["bm25", "vector", "structural", "fused", "fused_no_graph"]
    totals = {row: {m: 0.0 for m in METRIC_KEYS} for row in rows}

    for q, results_for_query in zip(queries, all_results):
        grades = build_grades(q)

        for row in rows:
            metrics = compute_metrics(results_for_query[row], grades)
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
        "fused_no_graph": "Hybrid (no graph)",
    }

    header = (
        f"{'Retriever':<26} {'MRR@5':>8} {'Prec@5':>8} {'nDCG@10':>9} "
        f"{'Recall@10':>10} {'Recall@15':>10} {'Recall@20':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for row, label in row_labels.items():
        m = results[row]
        print(
            f"{label:<26} {m['mrr']:>8.4f} {m['precision']:>8.4f} {m['ndcg10']:>9.4f} "
            f"{m['recall10']:>10.4f} {m['recall15']:>10.4f} {m['recall20']:>10.4f}"
        )
    print()


def main():
    queries = load_queries(QUERIES_PATH)
    bm25_retriever, vector_retriever, graph_retriever = setup()
    fuser = Fuser(bm25_weight=0.9, vector_weight=1.0, structural_weight=0.3)
    fuser_no_graph = Fuser(bm25_weight=0.9, vector_weight=1.0, structural_weight=0.0)
    all_results = run_queries(queries, bm25_retriever, vector_retriever, graph_retriever, fuser, fuser_no_graph)
    final_metrics = evaluate(queries, all_results)
    print_table(final_metrics)

if __name__ == "__main__":
    main()