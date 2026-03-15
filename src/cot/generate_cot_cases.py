import json
import re
from pathlib import Path

from src.config import FINAL_PROCESSED_FILE, COT_CASES_FILE, SAVE_EVERY
from src.llm.client import OpenRouterClient


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json_block(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("JSON блок не найден")


def build_cot_generation_prompt(case: dict) -> str:
    complaint = case.get("complaint_clean") or case.get("complaint") or ""
    summary = case.get("summary_llm_clean") or case.get("summary_llm") or ""
    findings = case.get("important_findings_clean") or case.get("important_findings") or []
    diagnosis = case.get("diagnosis") or ""

    return f"""
Сформируй краткое клиническое рассуждение для обучающего few-shot примера.

Жалоба: {complaint}
Симптомы: {"; ".join(findings[:12])}
Краткое описание: {summary}
Известный диагноз кейса: {diagnosis}

Верни строго JSON формата:
{{
  "cot": "5-7 коротких шагов рассуждения",
  "differential": ["гипотеза 1", "гипотеза 2"],
  "final_hypothesis": "{diagnosis}",
  "teaching_points": ["ключевой признак 1", "ключевой признак 2"]
}}
""".strip()


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    items = load_json(path)
    return {str(item["case_id"]): item for item in items}


def main():
    if not FINAL_PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {FINAL_PROCESSED_FILE}")

    cases = load_json(FINAL_PROCESSED_FILE)
    existing = load_existing(COT_CASES_FILE)
    client = OpenRouterClient()

    processed_since_save = 0

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", idx))
        diagnosis = (case.get("diagnosis") or "").strip()

        if not diagnosis:
            print(f"[{idx}/{len(cases)}] SKIP | case_id={case_id} | no diagnosis")
            continue

        if case_id in existing:
            print(f"[{idx}/{len(cases)}] SKIP | case_id={case_id} | already exists")
            continue

        prompt = build_cot_generation_prompt(case)

        try:
            raw = client.generate(prompt=prompt, temperature=0.4, max_tokens=1000)
            data = json.loads(extract_json_block(raw))

            cot = str(data.get("cot", "")).strip()
            differential = data.get("differential", [])
            if not isinstance(differential, list):
                differential = []

            final_hypothesis = str(data.get("final_hypothesis", diagnosis)).strip()
            teaching_points = data.get("teaching_points", [])
            if not isinstance(teaching_points, list):
                teaching_points = []

            complaint = case.get("complaint_clean") or case.get("complaint") or ""
            cot_text_for_embedding = (
                f"Жалоба: {complaint}. "
                f"Рассуждение: {cot}. "
                f"Диагноз: {final_hypothesis}"
            ).strip()

            existing[case_id] = {
                "case_id": case_id,
                "complaint": case.get("complaint", ""),
                "complaint_clean": case.get("complaint_clean", ""),
                "diagnosis": diagnosis,
                "cot": cot,
                "differential": differential,
                "final_hypothesis": final_hypothesis,
                "teaching_points": teaching_points,
                "cot_text_for_embedding": cot_text_for_embedding,
            }

            processed_since_save += 1
            print(f"[{idx}/{len(cases)}] OK | case_id={case_id}")

            if processed_since_save >= SAVE_EVERY:
                save_json(COT_CASES_FILE, list(existing.values()))
                print(f"Промежуточное сохранение → {COT_CASES_FILE}")
                processed_since_save = 0

        except Exception as e:
            print(f"[{idx}/{len(cases)}] ERROR | case_id={case_id} | {e}")

    save_json(COT_CASES_FILE, list(existing.values()))

    print("\nГотово:")
    print(f"CoT cases → {COT_CASES_FILE}")


if __name__ == "__main__":
    main()