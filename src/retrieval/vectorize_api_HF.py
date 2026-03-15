import json
import numpy as np
from pathlib import Path

from src.config import (
    FINAL_PROCESSED_FILE,
    EMBEDDINGS_NPY_FILE,
    EMBEDDINGS_META_FILE,
    SAVE_EVERY,
)
from src.retrieval.embed_utils import build_hf_client, get_passage_embedding


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_vectors_and_meta():
    meta = []
    vectors = []

    if EMBEDDINGS_META_FILE.exists():
        meta = load_json(EMBEDDINGS_META_FILE)

    if EMBEDDINGS_NPY_FILE.exists():
        vectors = np.load(EMBEDDINGS_NPY_FILE).tolist()

    if meta and vectors and len(meta) != len(vectors):
        raise ValueError(
            f"Несоответствие между meta и npy: meta={len(meta)}, vectors={len(vectors)}"
        )

    if meta and not vectors:
        raise ValueError("Есть embeddings_meta.json, но нет embeddings.npy.")

    if vectors and not meta:
        raise ValueError("Есть embeddings.npy, но нет embeddings_meta.json.")

    return vectors, meta


def save_outputs(vectors: list, meta: list) -> None:
    np.save(EMBEDDINGS_NPY_FILE, np.asarray(vectors, dtype=np.float32))
    save_json(EMBEDDINGS_META_FILE, meta)


def main():
    if not FINAL_PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {FINAL_PROCESSED_FILE}")

    cases = load_json(FINAL_PROCESSED_FILE)
    vectors, meta = load_existing_vectors_and_meta()

    done_ids = {str(item["case_id"]) for item in meta}
    client = build_hf_client()

    total = len(cases)
    processed_since_save = 0

    print(f"Всего кейсов: {total}")
    print(f"Уже векторизовано: {len(done_ids)}")

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", idx))

        if case_id in done_ids:
            print(f"[{idx}/{total}] SKIP | case_id={case_id}")
            continue

        text = (case.get("text_for_embedding_v2") or "").strip()
        if not text:
            print(f"[{idx}/{total}] EMPTY | case_id={case_id}")
            continue

        try:
            embedding = get_passage_embedding(text, client=client)

            vectors.append(embedding.tolist())
            meta.append({
                "case_id": case_id,
                "row_index": case.get("row_index", idx),
                "source": case.get("source", ""),
                "source_row_index": case.get("source_row_index"),
                "complaint": case.get("complaint", ""),
                "complaint_clean": case.get("complaint_clean", ""),
                "anamnesis": case.get("anamnesis", ""),
                "anamnesis_clean": case.get("anamnesis_clean", ""),
                "important_findings": case.get("important_findings", []),
                "important_findings_clean": case.get("important_findings_clean", []),
                "diagnosis": case.get("diagnosis", ""),
                "age": case.get("age"),
                "sex": case.get("sex", ""),
                "summary_llm": case.get("summary_llm", ""),
                "summary_llm_clean": case.get("summary_llm_clean", ""),
                "text_for_embedding_v2": text,
                "embedding_dim": int(len(embedding)),
                "matrix_index": len(vectors) - 1,
            })

            done_ids.add(case_id)
            processed_since_save += 1

            print(f"[{idx}/{total}] OK | case_id={case_id} | dim={len(embedding)}")

            if processed_since_save >= SAVE_EVERY:
                save_outputs(vectors, meta)
                print(f"Промежуточное сохранение → {EMBEDDINGS_NPY_FILE}")
                print(f"Промежуточное сохранение → {EMBEDDINGS_META_FILE}")
                processed_since_save = 0

        except Exception as e:
            print(f"[{idx}/{total}] ERROR | case_id={case_id} | {e}")

    save_outputs(vectors, meta)

    print("\nГотово:")
    print(f"Embeddings (npy)  → {EMBEDDINGS_NPY_FILE}")
    print(f"Embeddings meta   → {EMBEDDINGS_META_FILE}")
    print(f"Итого векторов: {len(vectors)}")


if __name__ == "__main__":
    main()