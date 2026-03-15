import json
from pathlib import Path
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm

from src.config import (
    NORMALIZED_CASES_FILE,
    PROCESSED_LLM_FILE,
    SUMMARY_MODEL_NAME,
    SAVE_EVERY,
    MAX_INPUT_LEN,
    MAX_OUTPUT_LEN,
    NO_REPEAT_NGRAM_SIZE,
    NUM_BEAMS,
    EARLY_STOPPING,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
BATCH_SIZE = 8 if DEVICE == "cuda" else 2


def safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_processed(path: Path) -> dict:
    if not path.exists():
        return {}

    data = load_json(path)
    result = {}

    for item in data:
        case_id = str(item["case_id"])
        result[case_id] = item

    return result


def extract_symptoms(case: dict) -> str:
    parts = []

    complaint = safe_str(case.get("complaint"))
    if complaint:
        parts.append(f"Жалобы: {complaint}")

    anamnesis = safe_str(case.get("anamnesis"))
    if anamnesis:
        parts.append(f"Анамнез: {anamnesis}")

    findings = case.get("important_findings", [])
    if isinstance(findings, list) and findings:
        clean_findings = []
        seen = set()

        for f in findings:
            f = safe_str(f)
            if not f or f.isdigit():
                continue
            key = f.lower()
            if key not in seen:
                clean_findings.append(f)
                seen.add(key)

        if clean_findings:
            parts.append("Клинические находки: " + ", ".join(clean_findings))

    return " ".join(parts).strip()


def build_embedding_text(summary: str, full_text: str) -> str:
    return (
        f"Краткое клиническое описание: {summary}. "
        f"Симптомы и признаки: {full_text}"
    ).strip()


def chunked(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class SummaryGenerator:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

    def make_summaries(self, texts: list[str]) -> list[str]:
        batch = self.tokenizer(
            texts,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if USE_AMP else nullcontext()
        )

        with torch.inference_mode():
            with amp_ctx:
                output_ids = self.model.generate(
                    **batch,
                    max_length=MAX_OUTPUT_LEN,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    num_beams=NUM_BEAMS,
                    early_stopping=EARLY_STOPPING,
                )

        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output_ids
        ]


def print_torch_diagnostics():
    print(f"Устройство: {DEVICE}")
    print(f"AMP: {USE_AMP}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"torch version: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        try:
            print(f"cuda device name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Не удалось получить имя CUDA-устройства: {e}")


def main():
    if not NORMALIZED_CASES_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {NORMALIZED_CASES_FILE}")

    cases = load_json(NORMALIZED_CASES_FILE)
    processed_by_id = load_existing_processed(PROCESSED_LLM_FILE)

    print_torch_diagnostics()
    print(f"Всего кейсов: {len(cases)}")
    print(f"Уже обработано: {len(processed_by_id)}")

    summarizer = SummaryGenerator(SUMMARY_MODEL_NAME, DEVICE)

    pending_items = []
    skipped_existing = 0
    skipped_empty = 0

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", idx))

        if case_id in processed_by_id:
            skipped_existing += 1
            tqdm.write(f"[{idx}/{len(cases)}] SKIP | case_id={case_id}")
            continue

        full_text = extract_symptoms(case)
        if not full_text:
            skipped_empty += 1
            tqdm.write(f"[{idx}/{len(cases)}] EMPTY | case_id={case_id}")
            continue

        pending_items.append({
            "idx": idx,
            "case_id": case_id,
            "case": case,
            "full_text": full_text,
        })

    print(f"К обработке: {len(pending_items)}")
    print(f"Пропущено как уже готовые: {skipped_existing}")
    print(f"Пропущено пустых: {skipped_empty}")

    processed_since_save = 0

    progress = tqdm(
        chunked(pending_items, BATCH_SIZE),
        total=(len(pending_items) + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="Summarizing cases",
        unit="batch"
    )

    for batch_items in progress:
        texts = [item["full_text"] for item in batch_items]

        try:
            summaries = summarizer.make_summaries(texts)

            for item, summary in zip(batch_items, summaries):
                idx = item["idx"]
                case_id = item["case_id"]
                case = item["case"]
                full_text = item["full_text"]

                text_for_embedding = build_embedding_text(summary, full_text)

                processed_by_id[case_id] = {
                    "case_id": case_id,
                    "row_index": idx,
                    "source": safe_str(case.get("source")),
                    "source_row_index": case.get("source_row_index", idx),
                    "complaint": safe_str(case.get("complaint")),
                    "anamnesis": safe_str(case.get("anamnesis")),
                    "important_findings": case.get("important_findings", []),
                    "diagnosis": safe_str(case.get("diagnosis")),
                    "age": case.get("age"),
                    "sex": safe_str(case.get("sex")),
                    "summary_llm": summary,
                    "text_for_embedding": text_for_embedding,
                }

                processed_since_save += 1
                tqdm.write(f"[{idx}/{len(cases)}] OK | case_id={case_id}")

            progress.set_postfix({
                "done": len(processed_by_id),
                "new": processed_since_save
            })

            if processed_since_save >= SAVE_EVERY:
                save_json(PROCESSED_LLM_FILE, list(processed_by_id.values()))
                tqdm.write(f"Промежуточное сохранение → {PROCESSED_LLM_FILE}")
                processed_since_save = 0

        except Exception as e:
            tqdm.write(f"Ошибка на батче: {e}")
            for item in batch_items:
                tqdm.write(f"[{item['idx']}/{len(cases)}] ERROR | case_id={item['case_id']}")

    save_json(PROCESSED_LLM_FILE, list(processed_by_id.values()))

    print("\nГотово:")
    print(f"Processed cases → {PROCESSED_LLM_FILE}")
    print(f"Итого обработано записей: {len(processed_by_id)}")


if __name__ == "__main__":
    main()