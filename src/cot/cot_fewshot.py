import json
import numpy as np
from pathlib import Path

from src.config import (
    COT_EMBEDDINGS_NPY_FILE,
    COT_EMBEDDINGS_META_FILE,
    COT_TOP_K,
    N_RUNS,
    DEFAULT_CONSENSUS_METHOD,
)
from src.retrieval.embed_utils import (
    build_hf_client,
    get_query_embedding,
    cosine_similarity_scores,
    l2_normalize_matrix,
)
from src.llm.client import OpenRouterClient
from src.llm.prompts import build_cot_prompt
from src.llm.parse_output import parse_llm_output
from src.consensus.consensus import build_consensus


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CoTSearcher:
    def __init__(self):
        self.embeddings = np.load(COT_EMBEDDINGS_NPY_FILE).astype(np.float32)
        self.meta = load_json(COT_EMBEDDINGS_META_FILE)
        self.embeddings = l2_normalize_matrix(self.embeddings)
        self.client = build_hf_client()

    def search(self, query_text: str, top_k: int = COT_TOP_K) -> list[dict]:
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


def run_cot_fewshot(
    patient_query: str,
    top_k: int = COT_TOP_K,
    n_runs: int = N_RUNS,
    consensus_method: str = DEFAULT_CONSENSUS_METHOD,
):
    searcher = CoTSearcher()
    llm_client = OpenRouterClient()

    retrieved_cases = searcher.search(patient_query, top_k=top_k)
    prompt = build_cot_prompt(patient_query, retrieved_cases)

    raw_outputs = []
    parsed_outputs = []

    for i in range(1, n_runs + 1):
        try:
            raw = llm_client.generate(prompt=prompt, temperature=0.7, max_tokens=1200)
            parsed = parse_llm_output(raw)

            raw_outputs.append(raw)
            parsed_outputs.append(parsed)
            print(f"[CoT LLM {i}/{n_runs}] OK")
        except Exception as e:
            print(f"[CoT LLM {i}/{n_runs}] ERROR | {e}")

    consensus = build_consensus(parsed_outputs, method=consensus_method)

    return {
        "patient_query": patient_query,
        "retrieved_cases": retrieved_cases,
        "prompt": prompt,
        "raw_outputs": raw_outputs,
        "parsed_outputs": parsed_outputs,
        "consensus": consensus,
    }