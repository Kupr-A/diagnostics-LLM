import os
import numpy as np
from huggingface_hub import InferenceClient

from src.config import EMBED_MODEL_NAME


def build_hf_client() -> InferenceClient:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "Переменная окружения HF_TOKEN не установлена."
        )

    return InferenceClient(
        provider="hf-inference",
        api_key=token,
    )


def add_e5_prefix(text: str, kind: str) -> str:
    text = (text or "").strip()

    if kind not in {"query", "passage"}:
        raise ValueError("kind должен быть 'query' или 'passage'")

    return f"{kind}: {text}"


def l2_normalize_vector(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def l2_normalize_matrix(matrix) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def get_embedding(text: str, kind: str, client: InferenceClient | None = None) -> np.ndarray:
    own_client = client or build_hf_client()
    prepared_text = add_e5_prefix(text, kind)

    embedding = own_client.feature_extraction(
        prepared_text,
        model=EMBED_MODEL_NAME,
    )

    return l2_normalize_vector(embedding)


def get_passage_embedding(text: str, client: InferenceClient | None = None) -> np.ndarray:
    return get_embedding(text=text, kind="passage", client=client)


def get_query_embedding(text: str, client: InferenceClient | None = None) -> np.ndarray:
    return get_embedding(text=text, kind="query", client=client)


def cosine_similarity_scores(query_vector, matrix) -> np.ndarray:
    query = l2_normalize_vector(query_vector)
    matrix = l2_normalize_matrix(matrix)
    return matrix @ query