import numpy as np

from src.retrieval.embed_utils import build_hf_client, get_passage_embedding


def build_output_embedding_text(parsed_output: dict) -> str:
    parts = []

    hypotheses = parsed_output.get("hypotheses", [])
    if hypotheses:
        hyp_lines = []
        for hyp in hypotheses:
            name = hyp.get("name", "")
            prob = hyp.get("probability", 0.0)
            reason = hyp.get("reason", "")
            hyp_lines.append(f"{name} | p={prob} | {reason}")
        parts.append("Гипотезы: " + " || ".join(hyp_lines))

    red_flags = parsed_output.get("red_flags", [])
    if red_flags:
        parts.append("Red flags: " + "; ".join(red_flags))

    next_questions = parsed_output.get("next_questions", [])
    if next_questions:
        parts.append("Уточняющие вопросы: " + "; ".join(next_questions))

    final_summary = parsed_output.get("final_summary", "")
    if final_summary:
        parts.append("Итог: " + final_summary)

    return ". ".join(parts).strip()


def vectorize_outputs(parsed_outputs: list[dict]) -> np.ndarray:
    client = build_hf_client()
    vectors = []

    for output in parsed_outputs:
        text = build_output_embedding_text(output)
        vec = get_passage_embedding(text, client=client)
        vectors.append(vec)

    return np.asarray(vectors, dtype=np.float32)


def build_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms

    return normalized @ normalized.T


def mean_off_diagonal_similarity(sim_matrix: np.ndarray) -> np.ndarray:
    n = sim_matrix.shape[0]
    if n == 0:
        return np.asarray([], dtype=np.float32)
    if n == 1:
        return np.asarray([1.0], dtype=np.float32)

    result = []
    for i in range(n):
        row = np.delete(sim_matrix[i], i)
        result.append(float(np.mean(row)) if len(row) else 1.0)

    return np.asarray(result, dtype=np.float32)


def select_central_answer(parsed_outputs: list[dict]) -> dict:
    if not parsed_outputs:
        return {
            "central_index": None,
            "central_output": None,
            "similarity_matrix": [],
            "centralities": [],
        }

    vectors = vectorize_outputs(parsed_outputs)
    sim_matrix = build_similarity_matrix(vectors)
    centralities = mean_off_diagonal_similarity(sim_matrix)

    central_index = int(np.argmax(centralities))

    return {
        "central_index": central_index,
        "central_output": parsed_outputs[central_index],
        "similarity_matrix": sim_matrix.tolist(),
        "centralities": centralities.tolist(),
    }