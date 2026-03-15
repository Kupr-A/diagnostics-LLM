import json
import numpy as np
from pathlib import Path

from src.config import (
    COT_CASES_FILE,
    COT_EMBEDDINGS_NPY_FILE,
    COT_EMBEDDINGS_META_FILE,
    SAVE_EVERY,
)
from src.retrieval.embed_utils import build_hf_client, get_passage_embedding


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing():
    meta = []
    vectors = []

    if COT_EMBEDDINGS_META_FILE.exists():
        meta = load_json(COT_EMBEDDINGS_META_FILE)

    if COT_EMBEDDINGS_NPY_FILE.exists():
        vectors = np.load(COT_EMBEDDINGS_NPY_FILE).tolist()

    return vectors, meta


def save_outputs(vectors: list, meta: list):
    np.save(COT_EMBEDDINGS_NPY_FILE, np.asarray(vectors, dtype=np.float32))
    save_json(COT_EMBEDDINGS_META_FILE, meta)


def main():
    if not COT_CASES_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {COT_CASES_FILE}")

    cases = load_json(COT_CASES_FILE)
    vectors, meta = load_existing()

    done_ids = {str(item["case_id"]) for item in meta}
    client = build_hf_client()
    processed_since_save = 0

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", idx))

        if case_id in done_ids:
            print(f"[{idx}/{len(cases)}] SKIP | case_id={case_id}")
            continue

        text = (case.get("cot_text_for_embedding") or "").strip()
        if not text:
            print(f"[{idx}/{len(cases)}] EMPTY | case_id={case_id}")
            continue

        try:
            emb = get_passage_embedding(text, client=client)

            vectors.append(emb.tolist())
            meta.append({
                "case_id": case_id,
                "complaint": case.get("complaint", ""),
                "complaint_clean": case.get("complaint_clean", ""),
                "diagnosis": case.get("diagnosis", ""),
                "cot": case.get("cot", ""),
                "final_hypothesis": case.get("final_hypothesis", ""),
                "cot_text_for_embedding": text,
                "matrix_index": len(vectors) - 1,
            })

            done_ids.add(case_id)
            processed_since_save += 1
            print(f"[{idx}/{len(cases)}] OK | case_id={case_id}")

            if processed_since_save >= SAVE_EVERY:
                save_outputs(vectors, meta)
                print(f"Промежуточное сохранение → {COT_EMBEDDINGS_NPY_FILE}")
                print(f"Промежуточное сохранение → {COT_EMBEDDINGS_META_FILE}")
                processed_since_save = 0

        except Exception as e:
            print(f"[{idx}/{len(cases)}] ERROR | case_id={case_id} | {e}")

    save_outputs(vectors, meta)

    print("\nГотово:")
    print(f"COT embeddings → {COT_EMBEDDINGS_NPY_FILE}")
    print(f"COT meta       → {COT_EMBEDDINGS_META_FILE}")


if __name__ == "__main__":
    main()