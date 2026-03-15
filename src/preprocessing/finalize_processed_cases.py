import json
import re
from pathlib import Path

from src.config import PROCESSED_LLM_FILE, PROCESSED_DIR


INPUT_FILE = PROCESSED_LLM_FILE
OUTPUT_FILE = PROCESSED_DIR / "RuMedPrimeData_processed_final.json"


PLACEHOLDER_PATTERNS = [
    r"\*ДАТА\*",
    r"\*СТРАНА\*",
    r"\*ГОРОД\*",
]

COMMON_REPLACEMENTS = {
    "фое": "фоне",
    "дышка": "одышка",
    "пояились": "появились",
    "постянные": "непостоянные",
    "пррием": "прием",
    "осмотор": "осмотр",
    "ч/з": "через",
    "поседние": "последние",
    "сненбольши": "с небольшим",
    "принитмал": "принимал",
    "Вышепиеречисленные": "вышеперечисленные",
    "деэрадиккацию": "деэрадикацию",
    "холецитси": "холецистит",
    "назофарингит рекомендовано обследованиее": "назофарингит",
    "Онофарингит": "назофарингит",
    "Жалобв": "Жалобы",
    "вышеизложенные": "указанные",
    "с-м": "синдром",
    "рез-м": "результатам",
    "рез-ми": "результатами",
    "обр - е": "образование",
    "обр - я": "образования",
    "стр - ры": "структуры",
    "эхоплотные вкл - я": "эхоплотные включения",
    "щит. ж-зы": "щитовидной железы",
    "щита=9,9": "щитовидная железа 9,9",
    "на изжога": "изжога",
}

FINDING_PREFIXES = [
    "жалобы:",
    "жалобы гб:",
    "клинические находки:",
    "на ",
    "- ",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:])([^\s])", r"\1 \2", text)
    text = re.sub(r"\.\.+", ".", text)
    text = re.sub(r",\s*,+", ", ", text)
    return text.strip(" ,.;")


def remove_placeholders(text: str) -> str:
    for pattern in PLACEHOLDER_PATTERNS:
        text = re.sub(pattern, "", text)
    return normalize_spaces(text)


def apply_common_replacements(text: str) -> str:
    for src, dst in COMMON_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def cleanup_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()
    text = apply_common_replacements(text)
    text = remove_placeholders(text)

    # убрать явный мусор
    text = re.sub(r"\bдо С\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bдень болезни\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bвыезд за границу\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bпрививка от гриппа\b", "", text, flags=re.IGNORECASE)

    # лишние пробелы/знаки
    text = normalize_spaces(text)

    return text


def strip_prefixes(text: str) -> str:
    lowered = text.lower()
    changed = True

    while changed:
        changed = False
        lowered = text.lower()
        for prefix in FINDING_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip(" -:,.")
                changed = True
                break

    return text.strip()


def simplify_parenthetical_choices(text: str) -> str:
    """
    Примеры:
    'кашель (сухой' + 'влажный)' -> позже схлопнется до 'кашель'
    'боль в горле (умеренная' -> 'боль в горле'
    """
    text = re.sub(r"\([^)]*$", "", text).strip()
    text = re.sub(r"^[^(]*\)", "", text).strip()
    text = re.sub(r"\([^)]*\)", "", text).strip()
    return normalize_spaces(text)


def split_into_candidate_parts(text: str) -> list[str]:
    if not text:
        return []

    # сначала мягко разделим по точкам с последующим большим фрагментом
    text = text.replace(";", ",")
    text = re.sub(r"\s+", " ", text)

    parts = re.split(r",|\.\s+|:\s+", text)
    result = []

    for part in parts:
        part = normalize_spaces(part)
        if not part:
            continue
        result.append(part)

    return result


def clean_one_finding(text: str) -> str:
    text = cleanup_text(text)
    text = strip_prefixes(text)
    text = simplify_parenthetical_choices(text)

    text = re.sub(r"\bс \d+[-–]?\d* С\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bпо ВАШ \d+ баллов\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bв пред \d+/?\d*\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bЧСС \d+\b", "", text, flags=re.IGNORECASE)

    # частные нормализации симптомов
    text = re.sub(r"\bпокашливание\b", "кашель", text, flags=re.IGNORECASE)
    text = re.sub(r"\bлегкая заложенность носа\b", "заложенность носа", text, flags=re.IGNORECASE)
    text = re.sub(r"\bощущение нехватки воздуха\b", "нехватка воздуха", text, flags=re.IGNORECASE)
    text = re.sub(r"\bчувство нехватки воздуха\b", "нехватка воздуха", text, flags=re.IGNORECASE)
    text = re.sub(r"\bпериодически потливость\b", "потливость", text, flags=re.IGNORECASE)
    text = re.sub(r"\bлабильность настроения\b", "перепады настроения", text, flags=re.IGNORECASE)
    text = re.sub(r"\bплаксивость\b", "плаксивость", text, flags=re.IGNORECASE)
    text = re.sub(r"\bсердцебиение до \d+\b", "сердцебиение", text, flags=re.IGNORECASE)
    text = re.sub(r"\bтахикардия до \d+\b", "тахикардия", text, flags=re.IGNORECASE)
    text = re.sub(r"\bповышение температуры до ([\d.,\- ]+)\s*С\b", r"температура \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bповышение температуры\b", "температура", text, flags=re.IGNORECASE)

    text = normalize_spaces(text)
    text = text.strip(" ,.;:-")

    # отфильтровать слишком шумные куски
    if not text:
        return ""
    if len(text) < 2:
        return ""
    if text.isdigit():
        return ""

    return text


def deduplicate_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def clean_findings(findings: list[str]) -> list[str]:
    if not isinstance(findings, list):
        return []

    cleaned = []

    for raw in findings:
        raw = str(raw).strip()
        if not raw:
            continue

        raw = cleanup_text(raw)

        # если элемент явно большой и слепленный — делим дополнительно
        parts = split_into_candidate_parts(raw)

        for part in parts:
            part_clean = clean_one_finding(part)

            if not part_clean:
                continue

            # выкидываем явные варианты выбора
            lowered = part_clean.lower()
            if lowered in {"сильная", "умеренная", "небольшая", "влажный", "сухой"}:
                continue

            cleaned.append(part_clean)

    # убираем мусорные хвосты
    filtered = []
    for item in cleaned:
        low = item.lower()
        if low in {"были но на фоне лечения стало лучше", "после предыдущего приема самочувствие удовлетворительное"}:
            continue
        filtered.append(item)

    return deduplicate_keep_order(filtered)


def extract_temperature(text: str) -> str:
    match = re.search(r"(\d{2}(?:[.,]\d+)?\s*[-–]\s*\d{2}(?:[.,]\d+)?)\s*с", text.lower())
    if match:
        return f"температура {match.group(1).replace(' ', '')}"

    match = re.search(r"температура\s*(\d{2}(?:[.,]\d+)?)", text.lower())
    if match:
        return f"температура {match.group(1)}"

    match = re.search(r"повышение температуры до\s*(\d{2}(?:[.,]\d+)?)", text.lower())
    if match:
        return f"температура {match.group(1)}"

    return ""


def build_complaint_clean(complaint: str, findings_clean: list[str], anamnesis_clean: str) -> str:
    complaint = cleanup_text(complaint)

    if complaint:
        return complaint

    top = findings_clean[:6]

    temp = extract_temperature(anamnesis_clean)
    if temp and temp not in [x.lower() for x in top]:
        top.append(temp)

    if not top:
        # fallback на первые слова анамнеза
        sentences = re.split(r"[.!?]", anamnesis_clean)
        for sent in sentences:
            sent = normalize_spaces(sent)
            if len(sent) >= 10:
                return sent[:180]

    return ", ".join(top)


def build_short_anamnesis(anamnesis: str) -> str:
    text = cleanup_text(anamnesis)

    # выкидываем слишком технические куски про назначения и часть лаборатории
    text = re.sub(r"принимает [^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"рекомендована терапия[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"по результатам[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"узи[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ттг[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"т4[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"актг[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"кортизол[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"метанефрины[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"норметанефрины[^.]+", "", text, flags=re.IGNORECASE)

    text = normalize_spaces(text)

    # берём первые 1-2 содержательные фразы
    sentences = [normalize_spaces(x) for x in re.split(r"[.!?]", text) if normalize_spaces(x)]
    if not sentences:
        return ""

    short = ". ".join(sentences[:2])
    return short[:300].strip(" .")


def clean_summary(summary: str, complaint_clean: str, findings_clean: list[str], anamnesis_short: str) -> str:
    summary = cleanup_text(summary)

    bad_markers = [
        "клинические находки:",
        "прививка от гриппа в испанию",
        "диагностирована тяжело",
        "отрицает выезд за границу",
    ]

    low = summary.lower()
    if not summary or any(marker in low for marker in bad_markers):
        summary = ""

    if len(summary) < 20:
        summary = ""

    # Если summary выглядит странно, строим свой короткий summary
    if not summary:
        parts = []
        if complaint_clean:
            parts.append(complaint_clean)
        if anamnesis_short:
            parts.append(anamnesis_short)
        summary = ". ".join(parts)

    # Слишком длинный / шумный summary режем
    summary = normalize_spaces(summary)
    summary = summary[:350].strip(" .")

    return summary


def build_text_for_embedding_v2(
    complaint_clean: str,
    findings_clean: list[str],
    anamnesis_short: str,
    summary_clean: str,
) -> str:
    sections = []

    if complaint_clean:
        sections.append(f"Жалоба: {complaint_clean}")

    if findings_clean:
        sections.append("Симптомы: " + "; ".join(findings_clean[:12]))

    if anamnesis_short:
        sections.append(f"Анамнез: {anamnesis_short}")

    if summary_clean:
        sections.append(f"Краткое описание: {summary_clean}")

    text = ". ".join(sections)
    text = normalize_spaces(text)
    return text.strip(" .")


def finalize_case(case: dict) -> dict:
    complaint = case.get("complaint", "")
    anamnesis = case.get("anamnesis", "")
    findings = case.get("important_findings", [])
    summary_llm = case.get("summary_llm", "")

    anamnesis_clean = build_short_anamnesis(anamnesis)
    findings_clean = clean_findings(findings)
    complaint_clean = build_complaint_clean(complaint, findings_clean, anamnesis_clean)
    summary_llm_clean = clean_summary(summary_llm, complaint_clean, findings_clean, anamnesis_clean)
    text_for_embedding_v2 = build_text_for_embedding_v2(
        complaint_clean=complaint_clean,
        findings_clean=findings_clean,
        anamnesis_short=anamnesis_clean,
        summary_clean=summary_llm_clean,
    )

    result = dict(case)
    result["complaint_clean"] = complaint_clean
    result["anamnesis_clean"] = anamnesis_clean
    result["important_findings_clean"] = findings_clean
    result["summary_llm_clean"] = summary_llm_clean
    result["text_for_embedding_v2"] = text_for_embedding_v2

    return result


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_FILE}")

    cases = load_json(INPUT_FILE)
    finalized = []

    for idx, case in enumerate(cases, start=1):
        try:
            final_case = finalize_case(case)
            finalized.append(final_case)
            print(f"[{idx}/{len(cases)}] OK | case_id={case.get('case_id', idx)}")
        except Exception as e:
            print(f"[{idx}/{len(cases)}] ERROR | case_id={case.get('case_id', idx)} | {e}")

    save_json(OUTPUT_FILE, finalized)

    print("\nГотово:")
    print(f"Финальный файл → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()