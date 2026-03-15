import json
import re


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

    raise ValueError("JSON блок не найден в ответе модели")


def repair_common_json_issues(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def to_float_prob(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value < 0:
        return 0.0
    if value > 1:
        if value <= 100:
            return round(value / 100.0, 4)
        return 1.0

    return round(value, 4)


def normalize_hypothesis(item: dict) -> dict:
    return {
        "name": str(item.get("name", "")).strip(),
        "probability": to_float_prob(item.get("probability", 0.0)),
        "reason": str(item.get("reason", "")).strip(),
    }


def deduplicate_text_list(items: list[str], limit: int | None = None) -> list[str]:
    result = []
    seen = set()

    for item in items:
        item = str(item).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)

    if limit is not None:
        return result[:limit]
    return result


def parse_llm_output(raw_text: str) -> dict:
    json_text = extract_json_block(raw_text)
    json_text = repair_common_json_issues(json_text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Не удалось распарсить JSON из ответа модели: {e}\n{json_text}")

    hypotheses = data.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        hypotheses = []

    normalized_hypotheses = []
    for item in hypotheses[:3]:
        if not isinstance(item, dict):
            continue
        hyp = normalize_hypothesis(item)
        if hyp["name"]:
            normalized_hypotheses.append(hyp)

    red_flags = data.get("red_flags", [])
    if not isinstance(red_flags, list):
        red_flags = []

    next_questions = data.get("next_questions", [])
    if not isinstance(next_questions, list):
        next_questions = []

    final_summary = str(data.get("final_summary", "")).strip()

    return {
        "hypotheses": normalized_hypotheses,
        "red_flags": deduplicate_text_list(red_flags, limit=10),
        "next_questions": deduplicate_text_list(next_questions, limit=5),
        "final_summary": final_summary,
        "raw_text": raw_text,
    }