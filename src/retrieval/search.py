import json
import numpy as np
from pathlib import Path

from src.config import EMBEDDINGS_FINAL_NPY_FILE, EMBEDDINGS_FINAL_META_FILE, TOP_K
from src.retrieval.embed_utils import (
    build_hf_client,
    get_query_embedding,
    cosine_similarity_scores,
    l2_normalize_matrix,
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SemanticSearcher:
    def __init__(self):
        if not EMBEDDINGS_FINAL_NPY_FILE.exists():
            raise FileNotFoundError(f"Файл не найден: {EMBEDDINGS_FINAL_NPY_FILE}")

        if not EMBEDDINGS_FINAL_META_FILE.exists():
            raise FileNotFoundError(f"Файл не найден: {EMBEDDINGS_FINAL_META_FILE}")

        self.embeddings = np.load(EMBEDDINGS_FINAL_NPY_FILE).astype(np.float32)
        self.meta = load_json(EMBEDDINGS_FINAL_META_FILE)

        if len(self.embeddings) != len(self.meta):
            raise ValueError(
                f"Несоответствие длины embeddings и meta: "
                f"{len(self.embeddings)} vs {len(self.meta)}"
            )

        self.embeddings = l2_normalize_matrix(self.embeddings)
        self.client = build_hf_client()

    def search(self, query_text: str, top_k: int = TOP_K) -> list[dict]:
        query_vector = get_query_embedding(query_text, client=self.client)
        scores = cosine_similarity_scores(query_vector, self.embeddings)

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            item = dict(self.meta[int(idx)])
            item["rank"] = rank
            item["similarity"] = float(scores[int(idx)])
            results.append(item)

        return results

    @staticmethod
    def pretty_print(results: list[dict]) -> None:
        for item in results:
            print(f"[{item['rank']}] similarity={item['similarity']:.4f}")
            print(f"case_id: {item.get('case_id', '')}")

            complaint_clean = item.get("complaint_clean", "")
            if complaint_clean:
                print(f"Жалоба (clean): {complaint_clean}")

            summary_clean = item.get("summary_llm_clean", "")
            if summary_clean:
                print(f"Краткое описание: {summary_clean}")

            diagnosis = item.get("diagnosis", "")
            if diagnosis:
                print(f"Диагноз: {diagnosis}")

            findings_clean = item.get("important_findings_clean", [])
            if findings_clean:
                print("Симптомы: " + "; ".join(findings_clean))

            print("-" * 80)