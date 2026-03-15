import json


OUTPUT_SCHEMA = {
    "hypotheses": [
        {
            "name": "Название гипотезы",
            "probability": 0.0,
            "reason": "Краткое обоснование на основе текущей жалобы и похожих кейсов"
        }
    ],
    "red_flags": [],
    "next_questions": [],
    "final_summary": "Краткий итог в 1-2 предложениях"
}


def format_case_for_prompt(case: dict, idx: int) -> str:
    parts = [f"[Похожий кейс {idx}]"]

    complaint = case.get("complaint_clean") or case.get("complaint") or ""
    summary = case.get("summary_llm_clean") or case.get("summary_llm") or ""
    diagnosis = case.get("diagnosis") or ""
    findings = case.get("important_findings_clean") or case.get("important_findings") or []

    if complaint:
        parts.append(f"Жалоба: {complaint}")

    if findings:
        parts.append("Симптомы: " + "; ".join(findings[:10]))

    if summary:
        parts.append(f"Краткое описание: {summary}")

    if diagnosis:
        parts.append(f"Диагноз похожего кейса: {diagnosis}")

    return "\n".join(parts)


def build_baseline_prompt(patient_query: str, retrieved_cases: list[dict]) -> str:
    few_shot_block = "\n\n".join(
        format_case_for_prompt(case, idx + 1)
        for idx, case in enumerate(retrieved_cases)
    )

    instructions = [
        "Проанализируй жалобу текущего пациента и похожие кейсы из базы.",
        "Предложи до 3 наиболее вероятных диагностических гипотез.",
        "Гипотезы должны быть отсортированы по убыванию вероятности.",
        "Поле probability должно быть числом от 0 до 1.",
        "Не копируй диагноз похожего кейса автоматически: используй его только как ориентир.",
        "reason должен кратко объяснять, какие симптомы говорят в пользу гипотезы.",
        "red_flags заполняй только если в текущей жалобе действительно есть тревожные признаки.",
        "next_questions должен содержать только уточняющие вопросы, которые реально помогают различить гипотезы.",
        "Если данных мало, всё равно верни осторожные гипотезы и укажи неопределённость в final_summary.",
        "Верни строго один JSON-объект и ничего кроме JSON.",
    ]

    return f"""
Ниже приведены похожие клинические кейсы из базы.

{few_shot_block}

[Текущий пациент]
Жалоба: {patient_query}

Требования:
{chr(10).join(f"- {item}" for item in instructions)}

Формат JSON:
{json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def build_cot_prompt(patient_query: str, retrieved_cases_with_cot: list[dict]) -> str:
    blocks = []

    for idx, case in enumerate(retrieved_cases_with_cot, start=1):
        complaint = case.get("complaint_clean") or case.get("complaint") or ""
        cot = case.get("cot") or ""
        diagnosis = case.get("diagnosis") or ""

        block = [f"[Кейс {idx}]"]
        if complaint:
            block.append(f"Жалоба: {complaint}")
        if cot:
            block.append(f"Рассуждение: {cot}")
        if diagnosis:
            block.append(f"Диагноз: {diagnosis}")

        blocks.append("\n".join(block))

    return f"""
Ниже приведены похожие клинические кейсы с рассуждениями.

{'\n\n'.join(blocks)}

[Текущий пациент]
Жалоба: {patient_query}

Сделай анализ по аналогии с примерами, но не копируй диагноз механически.
Верни до 3 вероятных гипотез, red_flags, уточняющие вопросы и краткий итог.
Верни строго один JSON-объект без markdown.

Формат JSON:
{json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()