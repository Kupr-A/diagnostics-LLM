import json
from pathlib import Path

from src.config import (
    TOP_K,
    N_RUNS,
    EVAL_TEST_CASES_FILE,
    EVAL_RESULTS_FILE,
)
from src.retrieval.search import SemanticSearcher
from src.llm.client import OpenRouterClient
from src.llm.prompts import build_baseline_prompt
from src.llm.parse_output import parse_llm_output
from src.consensus.consensus import build_consensus

METHODS = ["frequency", "weighted", "embedding", "hybrid"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def matches_expected(predicted_name: str, expected) -> bool:
    predicted = normalize_text(predicted_name)

    if isinstance(expected, str):
        expected = [expected]

    expected = [normalize_text(x) for x in expected if str(x).strip()]

    for target in expected:
        if target in predicted or predicted in target:
            return True

    return False


def evaluate_one_case(searcher, llm_client, case: dict) -> dict:
    patient_query = case.get("patient_query") or case.get("query") or ""
    expected = case.get("expected_diagnosis") or case.get("diagnosis") or []

    retrieved = searcher.search(patient_query, top_k=TOP_K)
    prompt = build_baseline_prompt(patient_query, retrieved)

    parsed_outputs = []
    raw_outputs = []

    for _ in range(N_RUNS):
        raw = llm_client.generate(prompt=prompt, temperature=0.7, max_tokens=1200)
        raw_outputs.append(raw)
        parsed_outputs.append(parse_llm_output(raw))

    method_results = {}

    for method in METHODS:
        consensus = build_consensus(parsed_outputs, method=method)
        top_name = ""
        if consensus.get("hypotheses"):
            top_name = consensus["hypotheses"][0].get("name", "")

        method_results[method] = {
            "top_prediction": top_name,
            "top1_match": matches_expected(top_name, expected),
            "consensus": consensus,
        }

    return {
        "patient_query": patient_query,
        "expected_diagnosis": expected,
        "retrieved_cases": retrieved,
        "method_results": method_results,
    }


def summarize(results: list[dict]) -> dict:
    summary = {}

    for method in METHODS:
        total = len(results)
        correct = 0

        for item in results:
            if item["method_results"][method]["top1_match"]:
                correct += 1

        summary[method] = {
            "cases": total,
            "top1_accuracy": round(correct / total, 4) if total else 0.0,
        }

    return summary


def main():
    if not EVAL_TEST_CASES_FILE.exists():
        raise FileNotFoundError(
            f"Файл тестовых кейсов не найден: {EVAL_TEST_CASES_FILE}"
        )

    test_cases = load_json(EVAL_TEST_CASES_FILE)

    searcher = SemanticSearcher()
    llm_client = OpenRouterClient()

    results = []

    for idx, case in enumerate(test_cases, start=1):
        try:
            print(f"[{idx}/{len(test_cases)}] RUN")
            item_result = evaluate_one_case(searcher, llm_client, case)
            results.append(item_result)
            print(f"[{idx}/{len(test_cases)}] OK")
        except Exception as e:
            print(f"[{idx}/{len(test_cases)}] ERROR | {e}")

    summary = summarize(results)

    output = {
        "summary": summary,
        "results": results,
    }

    save_json(EVAL_RESULTS_FILE, output)

    print("\nГотово:")
    print(f"Результаты evaluation → {EVAL_RESULTS_FILE}")
    print("\nSummary:")
    for method, metrics in summary.items():
        print(f"{method}: top1_accuracy={metrics['top1_accuracy']}")


if __name__ == "__main__":
    main()