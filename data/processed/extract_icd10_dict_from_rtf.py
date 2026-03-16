import json
import re
from pathlib import Path

from src.config import ICD10_PDF_FILE, ICD10_DICT_FILE


CODE_PATTERN = re.compile(
    r"^\s*([A-TV-Z][0-9]{2}(?:\.[0-9])?)([+*]?)\s+(.+?)\s*$"
)


def normalize_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(path: Path) -> str:
    """
    Сначала пробуем pdfplumber, если его нет — pypdf.
    """
    try:
        import pdfplumber

        pages_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)

        text = "\n".join(pages_text).strip()
        if text:
            return text
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(page_text)

        text = "\n".join(pages_text).strip()
        if text:
            return text
    except Exception as e:
        raise RuntimeError(
            "Не удалось извлечь текст из PDF. "
            "Поставь pdfplumber или pypdf, либо проверь, что PDF текстовый, а не скан."
        ) from e

    raise RuntimeError("PDF прочитан, но текст не извлечён")


def is_service_line(line: str) -> bool:
    low = line.lower()

    prefixes = [
        "включено:",
        "исключено:",
        "примечание:",
        "класс ",
        "часть ",
        "предисловие",
        "российская академия",
        "московский центр",
        "международная классификация болезней",
        "краткий вариант",
        "дата введения",
        "использование в работе",
        "полный перечень трехзначных рубрик",
        "кишечные инфекции",
    ]

    return any(low.startswith(prefix) for prefix in prefixes)


def should_append_to_previous(line: str) -> bool:
    if not line:
        return False

    if is_service_line(line):
        return False

    if CODE_PATTERN.match(line):
        return False

    # перенос продолжения описания
    if line[0].islower():
        return True

    if line.startswith("(") or line.startswith("-"):
        return True

    return False


def parse_icd10_dict(text: str) -> dict:
    lines = [normalize_spaces(x) for x in text.splitlines()]
    lines = [x for x in lines if x]

    icd_map = {}
    current_code = None

    for line in lines:
        if is_service_line(line):
            continue

        match = CODE_PATTERN.match(line)
        if match:
            code = match.group(1).strip()
            label = normalize_spaces(match.group(3)).strip(" .;,")

            # защитимся от ложных срабатываний на диапазоны классов
            if re.match(r"^[A-TV-Z][0-9]{2}(?:\.[0-9])?$", code) and label:
                icd_map[code] = label
                current_code = code
            continue

        if current_code and should_append_to_previous(line):
            icd_map[current_code] = normalize_spaces(icd_map[current_code] + " " + line)

    return icd_map


def main():
    if not ICD10_PDF_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {ICD10_PDF_FILE}")

    text = extract_text_from_pdf(ICD10_PDF_FILE)
    icd_map = parse_icd10_dict(text)

    if not icd_map:
        raise RuntimeError(
            "Не удалось извлечь ни одного ICD-10 кода из PDF. "
            "Возможно, PDF является сканом без текстового слоя."
        )

    ICD10_DICT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ICD10_DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(icd_map, f, ensure_ascii=False, indent=2)

    print("Готово:")
    print(f"Извлечено кодов: {len(icd_map)}")
    print(f"Словарь сохранён в: {ICD10_DICT_FILE}")

    print("\nПримеры:")
    for code in list(icd_map.keys())[:15]:
        print(f"{code} -> {icd_map[code]}")


if __name__ == "__main__":
    main()