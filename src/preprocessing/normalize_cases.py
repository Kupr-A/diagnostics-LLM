from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, Optional

from src.config import RAW_DIR, PROCESSED_DIR


INPUT_FILE = RAW_DIR / "RuMedPrimeData.tsv"
OUTPUT_FILE = PROCESSED_DIR / "RuMedPrimeData_normalized_full.json"


COLUMN_ALIASES = {
    "case_id": [
        "case_id",
        "id",
        "record_id",
        "uid",
        "caseid",
        "index",
        "idx",
    ],
    "complaint": [
        "complaint",
        "complaints",
        "query",
        "question",
        "patient_query",
        "chief_complaint",
        "жалобы",
        "жалоба",
        "complaint_text",
    ],
    "anamnesis": [
        "anamnesis",
        "history",
        "anamnesis_morbi",
        "medical_history",
        "анамнез",
        "anamnesis_text",
    ],
    "important_findings": [
        "important_findings",
        "findings",
        "key_findings",
        "clinical_findings",
        "symptoms",
        "symptom_list",
        "важные_находки",
        "клинические_находки",
        "признаки",
        "симптомы",
    ],
    "diagnosis": [
        "diagnosis",
        "final_diagnosis",
        "target",
        "label",
        "answer",
        "output",
        "диагноз",
    ],
    "age": [
        "age",
        "patient_age",
        "возраст",
    ],
    "sex": [
        "sex",
        "gender",
        "patient_sex",
        "пол",
    ],
}


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_key(key: str) -> str:
    return normalize_text(key).lower()


def find_existing_column(row: dict, aliases: Iterable[str]) -> Optional[str]:
    normalized_to_original = {normalize_key(k): k for k in row.keys()}
    for alias in aliases:
        candidate = normalized_to_original.get(normalize_key(alias))
        if candidate is not None:
            return candidate
    return None


def get_first_value(row: dict, aliases: Iterable[str], default: str = "") -> str:
    col = find_existing_column(row, aliases)
    if col is None:
        return default
    return normalize_text(row.get(col, default))


def split_findings_text(text: str) -> list[str]:
    if not text:
        return []

    parts = re.split(r"[;\n\r|]+|,\s*(?=[А-ЯA-Zа-яa-z0-9])", text)
    cleaned = []
    seen = set()

    for part in parts:
        item = normalize_text(part)
        if not item:
            continue
        if item.isdigit():
            continue
        key = item.lower()
        if key not in seen:
            cleaned.append(item)
            seen.add(key)

    return cleaned


def parse_possible_json_list(value: str) -> Optional[list[str]]:
    if not value:
        return None

    if not (value.startswith("[") and value.endswith("]")):
        return None

    try:
        parsed = json.loads(value)
    except Exception:
        return None

    if not isinstance(parsed, list):
        return None

    result = []
    seen = set()

    for item in parsed:
        text = normalize_text(item)
        if not text or text.isdigit():
            continue
        key = text.lower()
        if key not in seen:
            result.append(text)
            seen.add(key)

    return result


def parse_important_findings(raw_value: str) -> list[str]:
    raw_value = normalize_text(raw_value)
    if not raw_value:
        return []

    parsed_json = parse_possible_json_list(raw_value)
    if parsed_json is not None:
        return parsed_json

    return split_findings_text(raw_value)


def normalize_sex(value: str) -> str:
    value = normalize_text(value).lower()

    mapping = {
        "м": "male",
        "муж": "male",
        "мужской": "male",
        "male": "male",
        "f": "female",
        "ж": "female",
        "жен": "female",
        "женский": "female",
        "female": "female",
    }
    return mapping.get(value, value)


def normalize_age(value: str):
    value = normalize_text(value)
    if not value:
        return None

    match = re.search(r"\d{1,3}", value)
    if not match:
        return None

    try:
        age = int(match.group(0))
        if 0 <= age <= 120:
            return age
    except ValueError:
        return None

    return None


def row_to_case(row: dict, row_index: int) -> dict:
    complaint = get_first_value(row, COLUMN_ALIASES["complaint"])
    anamnesis = get_first_value(row, COLUMN_ALIASES["anamnesis"])
    findings_raw = get_first_value(row, COLUMN_ALIASES["important_findings"])
    diagnosis = get_first_value(row, COLUMN_ALIASES["diagnosis"])
    age_raw = get_first_value(row, COLUMN_ALIASES["age"])
    sex_raw = get_first_value(row, COLUMN_ALIASES["sex"])
    case_id = get_first_value(row, COLUMN_ALIASES["case_id"], default=str(row_index))

    important_findings = parse_important_findings(findings_raw)

    # Если явного списка находок нет, но complaint/anamnesis есть,
    # просто не дублируем туда весь текст: findings оставляем пустым.
    return {
        "case_id": case_id or str(row_index),
        "source": "RuMedPrimeData",
        "source_row_index": row_index,
        "complaint": complaint,
        "anamnesis": anamnesis,
        "important_findings": important_findings,
        "diagnosis": diagnosis,
        "age": normalize_age(age_raw),
        "sex": normalize_sex(sex_raw) if sex_raw else "",
    }


def detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        return dialect.delimiter
    except Exception:
        return "\t"


def load_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = detect_delimiter(sample)
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_FILE}")

    rows = load_rows(INPUT_FILE)
    normalized_cases = []

    skipped = 0

    for row_index, row in enumerate(rows, start=1):
        case = row_to_case(row, row_index=row_index)

        # Минимальный фильтр: пропускаем совсем пустые записи
        if not case["complaint"] and not case["anamnesis"] and not case["important_findings"]:
            skipped += 1
            continue

        normalized_cases.append(case)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized_cases, f, ensure_ascii=False, indent=2)

    print("Готово:")
    print(f"Источник: {INPUT_FILE}")
    print(f"Сохранено: {OUTPUT_FILE}")
    print(f"Всего строк в источнике: {len(rows)}")
    print(f"Нормализовано кейсов: {len(normalized_cases)}")
    print(f"Пропущено пустых строк: {skipped}")


if __name__ == "__main__":
    main()