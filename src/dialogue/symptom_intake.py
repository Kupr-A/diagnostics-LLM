import re


INTAKE_TEMPLATES = {
    "chest_pain": [
        {
            "id": "chest_pain_onset",
            "question": "Когда началась боль в груди?",
            "type": "single-choice",
            "options": ["внезапно", "постепенно", "несколько дней назад", "давно/хронически"],
        },
        {
            "id": "chest_pain_trigger",
            "question": "С чем связана боль?",
            "type": "multi-choice",
            "options": ["с физической нагрузкой", "с дыханием", "с движением", "после еды", "непонятно"],
        },
        {
            "id": "chest_pain_associated",
            "question": "Есть ли сопутствующие симптомы?",
            "type": "multi-choice",
            "options": ["одышка", "тошнота", "потливость", "головокружение", "отдает в руку/челюсть"],
        },
    ],
    "abdominal_pain": [
        {
            "id": "abd_location",
            "question": "Где именно болит живот?",
            "type": "single-choice",
            "options": ["вверху", "внизу", "справа", "слева", "по центру", "по всему животу"],
        },
        {
            "id": "abd_associated",
            "question": "Что есть вместе с болью?",
            "type": "multi-choice",
            "options": ["тошнота", "рвота", "понос", "запор", "температура", "вздутие"],
        },
        {
            "id": "abd_food",
            "question": "Связано ли с приемом пищи?",
            "type": "single-choice",
            "options": ["усиливается после еды", "уменьшается после еды", "не связано", "не знаю"],
        },
    ],
    "respiratory": [
        {
            "id": "resp_temp",
            "question": "Есть ли температура?",
            "type": "single-choice",
            "options": ["нет", "до 37.5", "37.5-38.5", "выше 38.5", "не измерял(а)"],
        },
        {
            "id": "resp_cough",
            "question": "Какой кашель?",
            "type": "single-choice",
            "options": ["нет кашля", "сухой", "с мокротой", "редкое покашливание"],
        },
        {
            "id": "resp_extra",
            "question": "Есть ли еще симптомы?",
            "type": "multi-choice",
            "options": ["боль в горле", "насморк", "заложенность носа", "одышка", "боль в груди"],
        },
    ],
    "headache": [
        {
            "id": "headache_onset",
            "question": "Как началась головная боль?",
            "type": "single-choice",
            "options": ["внезапно", "постепенно", "после стресса", "после нагрузки"],
        },
        {
            "id": "headache_location",
            "question": "Где болит голова?",
            "type": "single-choice",
            "options": ["лоб", "затылок", "висок", "вся голова", "односторонне"],
        },
        {
            "id": "headache_extra",
            "question": "Есть ли сопутствующие симптомы?",
            "type": "multi-choice",
            "options": ["тошнота", "головокружение", "нарушение зрения", "онемение", "температура"],
        },
    ],
    "general": [
        {
            "id": "duration",
            "question": "Как давно это началось?",
            "type": "single-choice",
            "options": ["сегодня", "1-3 дня", "около недели", "несколько недель", "давно"],
        },
        {
            "id": "severity",
            "question": "Насколько выражены симптомы?",
            "type": "single-choice",
            "options": ["легкие", "умеренные", "сильные", "очень сильные"],
        },
    ],
}


DOMAIN_KEYWORDS = {
    "chest_pain": ["боль в груди", "за грудиной", "жжение в груди", "давит в груди"],
    "abdominal_pain": ["болит живот", "боль в животе", "в животе", "желудок", "тошнит"],
    "respiratory": ["кашель", "насморк", "горло", "температура", "одышка"],
    "headache": ["головная боль", "головокружение", "болит голова", "затылок", "лоб"],
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def detect_domains(patient_text: str) -> list[str]:
    text = normalize_text(patient_text)
    detected = []

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            detected.append(domain)

    if not detected:
        detected.append("general")

    if "general" not in detected:
        detected.append("general")

    return detected


def generate_intake_questions(patient_text: str, max_questions: int = 5) -> list[dict]:
    domains = detect_domains(patient_text)

    questions = []
    seen_ids = set()

    for domain in domains:
        for q in INTAKE_TEMPLATES.get(domain, []):
            if q["id"] not in seen_ids:
                questions.append(q)
                seen_ids.add(q["id"])

    return questions[:max_questions]


def enrich_query_with_intake(patient_text: str, answers: dict) -> str:
    additions = []

    for question_id, answer in answers.items():
        if isinstance(answer, list):
            answer_text = ", ".join(str(x) for x in answer if str(x).strip())
        else:
            answer_text = str(answer).strip()

        if answer_text:
            additions.append(f"{question_id}: {answer_text}")

    if not additions:
        return patient_text.strip()

    return patient_text.strip() + ". Уточнения: " + "; ".join(additions)