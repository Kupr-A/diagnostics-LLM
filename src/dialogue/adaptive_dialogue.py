from src.config import (
    TOP_K,
    N_RUNS,
    DEFAULT_CONSENSUS_METHOD,
    DIALOGUE_MAX_TURNS,
    DIALOGUE_CONFIDENCE_THRESHOLD,
)
from src.retrieval.search import SemanticSearcher
from src.llm.client import OpenRouterClient
from src.llm.prompts import build_baseline_prompt
from src.llm.parse_output import parse_llm_output
from src.consensus.consensus import build_consensus
from src.dialogue.dialogue_state import DialogueState
from src.dialogue.symptom_intake import (
    generate_intake_questions,
    enrich_query_with_intake,
)


TARGETED_QUESTION_RULES = {
    "гиперто": [
        "Какое обычно давление и до каких значений оно повышается?",
        "Есть ли головная боль, шум в ушах, тошнота при повышении давления?",
    ],
    "орви": [
        "Есть ли температура и какая максимальная?",
        "Какой кашель: сухой или с мокротой?",
    ],
    "грипп": [
        "Есть ли выраженная ломота в мышцах и озноб?",
        "Как быстро началось заболевание?",
    ],
    "гэрб": [
        "Есть ли изжога или отрыжка?",
        "Связаны ли симптомы с едой или положением лежа?",
    ],
    "стенокард": [
        "Возникает ли боль при физической нагрузке?",
        "Отдает ли боль в руку, плечо или челюсть?",
    ],
}


def estimate_top_confidence(consensus: dict) -> float:
    hypotheses = consensus.get("hypotheses", [])
    if not hypotheses:
        return 0.0
    return float(hypotheses[0].get("confidence", 0.0))


def analyze_query(patient_query: str, top_k: int = TOP_K, n_runs: int = N_RUNS, consensus_method: str = DEFAULT_CONSENSUS_METHOD):
    searcher = SemanticSearcher()
    llm_client = OpenRouterClient()

    retrieved_cases = searcher.search(patient_query, top_k=top_k)
    prompt = build_baseline_prompt(patient_query, retrieved_cases)

    parsed_outputs = []
    for _ in range(n_runs):
        raw = llm_client.generate(prompt=prompt, temperature=0.7, max_tokens=1200)
        parsed_outputs.append(parse_llm_output(raw))

    consensus = build_consensus(parsed_outputs, method=consensus_method)

    return {
        "retrieved_cases": retrieved_cases,
        "consensus": consensus,
        "parsed_outputs": parsed_outputs,
        "prompt": prompt,
    }


def start_dialogue(patient_query: str, intake_answers: dict | None = None) -> DialogueState:
    intake_answers = intake_answers or {}
    enriched_query = enrich_query_with_intake(patient_query, intake_answers)

    analysis = analyze_query(enriched_query)

    state = DialogueState(
        original_query=patient_query,
        enriched_query=enriched_query,
        intake_answers=intake_answers,
        current_retrieved_cases=analysis["retrieved_cases"],
        current_consensus=analysis["consensus"],
        max_turns=DIALOGUE_MAX_TURNS,
    )

    return state


def should_stop(state: DialogueState) -> bool:
    if state.turn_count >= state.max_turns:
        state.stop_reason = "max_turns"
        return True

    red_flags = state.current_consensus.get("red_flags", [])
    if red_flags:
        state.stop_reason = "red_flags"
        return True

    top_conf = estimate_top_confidence(state.current_consensus)
    if top_conf >= DIALOGUE_CONFIDENCE_THRESHOLD:
        state.stop_reason = "confidence"
        return True

    return False


def generate_targeted_questions(state: DialogueState, max_questions: int = 3) -> list[str]:
    if should_stop(state):
        return []

    hypotheses = state.current_consensus.get("hypotheses", [])
    asked_set = {q.lower().strip() for q in state.asked_questions}
    result = []

    for hyp in hypotheses:
        name = hyp.get("name", "").lower()

        for key, questions in TARGETED_QUESTION_RULES.items():
            if key in name:
                for q in questions:
                    if q.lower().strip() not in asked_set and q not in result:
                        result.append(q)

    if not result:
        fallback = [
            "Когда начались симптомы?",
            "Что усиливает или уменьшает симптомы?",
            "Есть ли температура, слабость, тошнота или другие сопутствующие симптомы?",
        ]
        for q in fallback:
            if q.lower().strip() not in asked_set and q not in result:
                result.append(q)

    return result[:max_questions]


def apply_dialogue_answer(state: DialogueState, question: str, answer: str) -> DialogueState:
    state.asked_questions.append(question)
    state.answer_history.append({"question": question, "answer": answer})
    state.turn_count += 1

    state.enriched_query = (
        state.enriched_query.strip()
        + f". Дополнительная информация: {question} Ответ: {answer}"
    )

    analysis = analyze_query(state.enriched_query)
    state.current_retrieved_cases = analysis["retrieved_cases"]
    state.current_consensus = analysis["consensus"]

    return state


def build_intake_then_dialogue(patient_query: str) -> dict:
    intake_questions = generate_intake_questions(patient_query)
    state = start_dialogue(patient_query)

    return {
        "intake_questions": intake_questions,
        "dialogue_state": state,
        "next_targeted_questions": generate_targeted_questions(state),
    }