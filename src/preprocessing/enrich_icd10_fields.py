import csv
import json
from pathlib import Path

from src.config import (
    RAW_TSV_FILE,
    ICD10_DICT_FILE,
    PROCESSED_LLM_FILE,
    FINAL_PROCESSED_FILE,
    EMBEDDINGS_META_FILE,
    EMBEDDINGS_FINAL_META_FILE,
    COT_CASES_FILE,
    COT_EMBEDDINGS_META_FILE,
)


TARGET_FILES = [
    PROCESSED_LLM_FILE,
    FINAL_PROCESSED_FILE,
    EMBEDDINGS_META_FILE,
    EMBEDDINGS_FINAL_META_FILE,
    COT_CASES_FILE,
    COT_EMBEDDINGS_META_FILE,
]


ICD10_COLUMN_ALIASES = [
    "icd10",
    "ICD10",
    "icd_10",
    "icd",
    "diag",
    "diagnosis_code",
]


def normalize_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        return dialect.delimiter
    except Exception:
        return "\t"


def load_icd10_dict(path: Path) -> dict:
    if not path.exists():
        print(f"WARNING: словарь ICD-10 не найден: {path}")
        return {}

    data = load_json(path)

    if isinstance(data, dict):
        return {normalize_text(k): normalize_text(v) for k, v in data.items()}

    result = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            code = normalize_text(item.get("code") or item.get("icd10") or item.get("icd10_code"))
            label = normalize_text(item.get("label") or item.get("name") or item.get("diagnosis"))
            if code:
                result[code] = label

    return result


def find_icd10_column(fieldnames: list[str]) -> str | None:
    normalized = {normalize_text(name).lower(): name for name in fieldnames}
    for alias in ICD10_COLUMN_ALIASES:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def build_icd10_map_from_tsv(tsv_path: Path, icd10_dict: dict) -> dict:
    if not tsv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {tsv_path}")

    with open(tsv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        delimiter = detect_delimiter(sample)
        reader = csv.DictReader(f, delimiter=delimiter)

        if not reader.fieldnames:
            raise ValueError("Не удалось прочитать заголовки TSV")

        icd_col = find_icd10_column(reader.fieldnames)
        if not icd_col:
            raise ValueError(f"Не найдена колонка icd10. Заголовки: {reader.fieldnames}")

        mapping = {}

        for row_index, row in enumerate(reader, start=1):
            code = normalize_text(row.get(icd_col))
            if not code:
                continue

            parent = code.split(".")[0] if "." in code else code
            label = icd10_dict.get(code, "")
            diagnosis = label if label else code
            diagnosis_display = f"{code} — {label}" if label else code

            mapping[str(row_index)] = {
                "icd10_code": code,
                "icd10_parent": parent,
                "icd10_label": label,
                "diagnosis": diagnosis,
                "diagnosis_display": diagnosis_display,
            }

    return mapping


def get_row_key(item: dict) -> str:
    """
    Привязываем по source_row_index / row_index / case_id.
    Для твоего пайплайна обычно source_row_index или row_index совпадают с номером строки TSV.
    """
    for key in ["source_row_index", "row_index", "case_id"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def enrich_one_item(item: dict, icd_info: dict) -> dict:
    enriched = dict(item)

    enriched["icd10_code"] = icd_info["icd10_code"]
    enriched["icd10_parent"] = icd_info["icd10_parent"]
    enriched["icd10_label"] = icd_info["icd10_label"]
    enriched["diagnosis"] = icd_info["diagnosis"]
    enriched["diagnosis_display"] = icd_info["diagnosis_display"]

    return enriched


def enrich_file(path: Path, icd_map: dict):
    if not path.exists():
        print(f"SKIP | file not found: {path}")
        return

    data = load_json(path)
    if not isinstance(data, list):
        print(f"SKIP | unsupported JSON structure: {path}")
        return

    updated = []
    enriched_count = 0
    missing_count = 0

    for item in data:
        if not isinstance(item, dict):
            updated.append(item)
            continue

        row_key = get_row_key(item)
        icd_info = icd_map.get(row_key)

        if not icd_info:
            updated.append(item)
            missing_count += 1
            continue

        updated.append(enrich_one_item(item, icd_info))
        enriched_count += 1

    save_json(path, updated)

    print(f"UPDATED | {path}")
    print(f"  enriched: {enriched_count}")
    print(f"  missing:  {missing_count}")


def main():
    icd10_dict = load_icd10_dict(ICD10_DICT_FILE)
    icd_map = build_icd10_map_from_tsv(RAW_TSV_FILE, icd10_dict)

    print(f"ICD rows loaded from TSV: {len(icd_map)}")
    print(f"ICD labels loaded from dict: {len(icd10_dict)}")

    for path in TARGET_FILES:
        enrich_file(path, icd_map)


if __name__ == "__main__":
    main()