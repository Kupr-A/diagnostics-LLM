import argparse

from src.config import (
    TOP_K,
    N_RUNS,
    DEFAULT_CONSENSUS_METHOD,
)
from src.retrieval.search import SemanticSearcher
from src.llm.client import OpenRouterClient
from src.llm.prompts import build_baseline_prompt
from src.llm.parse_output import parse_llm_output
from src.consensus.consensus import build_consensus
from src.cot.cot_fewshot import run_cot_fewshot
from src.dialogue.symptom_intake import (
    generate_intake_questions,
    enrich_query_with_intake,
)
from src.dialogue.adaptive_dialogue import (
    start_dialogue,
    should_stop,
    generate_targeted_questions,
    apply_dialogue_answer,
)


def print_retrieved_cases(retrieved_cases: list[dict]) -> None:
    if not retrieved_cases:
        print("\nПохожие кейсы не найдены.\n")
        return

    print("\nТоп похожих кейсов:\n")
    SemanticSearcher.pretty_print(retrieved_cases)


def print_parsed_output(parsed_output: dict, run_idx: int) -> None:
    print(f"\n[LLM {run_idx}] parsed output:")

    hypotheses = parsed_output.get("hypotheses", [])
    if not hypotheses:
        print("  Гипотезы не найдены")
    else:
        for i, hyp in enumerate(hypotheses, start=1):
            print(
                f"  {i}. {hyp['name']} | prob={hyp['probability']:.4f}\n"
                f"     reason: {hyp['reason']}"
            )

    red_flags = parsed_output.get("red_flags", [])
    if red_flags:
        print("  red_flags:", "; ".join(red_flags))

    next_questions = parsed_output.get("next_questions", [])
    if next_questions:
        print("  next_questions:")
        for q in next_questions:
            print(f"    - {q}")

    final_summary = parsed_output.get("final_summary", "")
    if final_summary:
        print("  final_summary:", final_summary)


def print_consensus(consensus: dict, show_similarity: bool = False) -> None:
    print("\n" + "=" * 80)
    print(f"CONSENSUS METHOD: {consensus.get('method', 'unknown')}")
    print("=" * 80)

    hypotheses = consensus.get("hypotheses", [])
    if not hypotheses:
        print("Консенсусные гипотезы не получены.")
    else:
        print("\nКонсенсусные гипотезы:\n")
        for idx, hyp in enumerate(hypotheses, start=1):
            print(f"{idx}. {hyp.get('name', '')}")

            if "confidence" in hyp:
                print(f"   confidence: {hyp['confidence']}")
            if "score" in hyp:
                print(f"   score: {hyp['score']}")
            if "mentions" in hyp:
                print(f"   mentions: {hyp['mentions']}")
            if "freq_score" in hyp:
                print(f"   freq_score: {hyp['freq_score']}")
            if "weighted_score" in hyp:
                print(f"   weighted_score: {hyp['weighted_score']}")
            if "centrality_score" in hyp:
                print(f"   centrality_score: {hyp['centrality_score']}")
            if hyp.get("reason"):
                print(f"   reason: {hyp['reason']}")
            print()

    red_flags = consensus.get("red_flags", [])
    if red_flags:
        print("Red flags:")
        for flag in red_flags:
            print(f"- {flag}")
        print()

    next_questions = consensus.get("next_questions", [])
    if next_questions:
        print("Уточняющие вопросы:")
        for q in next_questions:
            print(f"- {q}")
        print()

    central_index = consensus.get("central_index")
    if central_index is not None:
        print(f"Central answer index: {central_index}")

    centralities = consensus.get("centralities", [])
    if centralities:
        print("Centralities:", [round(x, 4) for x in centralities])

    if show_similarity:
        sim_matrix = consensus.get("similarity_matrix", [])
        if sim_matrix:
            print("\nSimilarity matrix:")
            for row in sim_matrix:
                print([round(x, 4) for x in row])


def run_baseline_pipeline(
    patient_query: str,
    top_k: int = TOP_K,
    n_runs: int = N_RUNS,
    consensus_method: str = DEFAULT_CONSENSUS_METHOD,
    temperature: float = 0.7,
    show_prompt: bool = False,
    show_similarity: bool = False,
) -> dict:
    searcher = SemanticSearcher()
    llm_client = OpenRouterClient()

    print("\nПоиск похожих кейсов...")
    retrieved_cases = searcher.search(patient_query, top_k=top_k)
    print_retrieved_cases(retrieved_cases)

    print("\nСборка prompt...")
    prompt = build_baseline_prompt(patient_query, retrieved_cases)

    if show_prompt:
        print("\n" + "=" * 80)
        print("PROMPT")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

    raw_outputs = []
    parsed_outputs = []

    print(f"\nЗапуск LLM ({n_runs} прогонов)...")
    for i in range(1, n_runs + 1):
        try:
            raw_text = llm_client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=1200,
            )
            raw_outputs.append(raw_text)

            parsed = parse_llm_output(raw_text)
            parsed_outputs.append(parsed)

            print(f"[LLM {i}/{n_runs}] OK")
            print_parsed_output(parsed, i)

        except Exception as e:
            print(f"[LLM {i}/{n_runs}] ERROR | {e}")

    if not parsed_outputs:
        raise RuntimeError("Не удалось получить ни одного валидного ответа от LLM.")

    consensus = build_consensus(parsed_outputs, method=consensus_method)
    print_consensus(consensus, show_similarity=show_similarity)

    return {
        "mode": "baseline",
        "patient_query": patient_query,
        "retrieved_cases": retrieved_cases,
        "raw_outputs": raw_outputs,
        "parsed_outputs": parsed_outputs,
        "consensus": consensus,
        "prompt": prompt,
    }


def ask_intake_answers(patient_query: str) -> tuple[dict, str]:
    intake_questions = generate_intake_questions(patient_query)
    answers = {}

    if not intake_questions:
        return answers, patient_query

    print("\nПредварительное уточнение симптомов:")
    print("-" * 80)

    for idx, q in enumerate(intake_questions, start=1):
        print(f"\n{idx}. {q['question']}")
        options = q.get("options", [])
        if options:
            print("Варианты:", " | ".join(options))
        user_answer = input("> ").strip()
        if user_answer:
            answers[q["id"]] = user_answer

    enriched_query = enrich_query_with_intake(patient_query, answers)

    print("\nУточнённая жалоба:")
    print(enriched_query)

    return answers, enriched_query


def run_dialogue_mode(
    patient_query: str,
    show_similarity: bool = False,
) -> dict:
    intake_answers, enriched_query = ask_intake_answers(patient_query)

    state = start_dialogue(patient_query, intake_answers=intake_answers)

    print("\nНачальный анализ:")
    print_consensus(state.current_consensus, show_similarity=show_similarity)

    while not should_stop(state):
        questions = generate_targeted_questions(state)
        if not questions:
            state.stop_reason = "no_more_questions"
            break

        question = questions[0]
        print("\nСледующий вопрос:")
        print(question)

        answer = input("> ").strip()
        if not answer:
            state.stop_reason = "user_stopped"
            break

        state = apply_dialogue_answer(state, question, answer)

        print("\nОбновлённый анализ:")
        print_consensus(state.current_consensus, show_similarity=show_similarity)

    print("\nДиалог завершён.")
    print(f"Причина остановки: {state.stop_reason or 'unknown'}")

    return {
        "mode": "dialogue",
        "patient_query": patient_query,
        "intake_answers": intake_answers,
        "enriched_query": enriched_query,
        "state": state,
        "consensus": state.current_consensus,
    }


def run_cot_mode(
    patient_query: str,
    consensus_method: str = DEFAULT_CONSENSUS_METHOD,
    show_similarity: bool = False,
) -> dict:
    result = run_cot_fewshot(
        patient_query=patient_query,
        consensus_method=consensus_method,
    )

    print("\nCoT few-shot retrieved cases:\n")
    for item in result["retrieved_cases"]:
        print(f"[{item['rank']}] similarity={item['similarity']:.4f}")
        print(f"case_id: {item.get('case_id', '')}")
        if item.get("complaint_clean"):
            print(f"Жалоба: {item['complaint_clean']}")
        if item.get("diagnosis"):
            print(f"Диагноз: {item['diagnosis']}")
        if item.get("cot"):
            print(f"CoT: {item['cot']}")
        print("-" * 80)

    print_consensus(result["consensus"], show_similarity=show_similarity)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostics LLM pipeline: baseline / cot / dialogue"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "cot", "dialogue"],
        help="Режим запуска",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Жалоба пациента",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Сколько похожих кейсов брать",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help="Сколько прогонов LLM делать",
    )
    parser.add_argument(
        "--consensus",
        type=str,
        default=DEFAULT_CONSENSUS_METHOD,
        choices=["frequency", "weighted", "embedding", "hybrid"],
        help="Метод consensus",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Температура генерации",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Печатать полный prompt",
    )
    parser.add_argument(
        "--show-similarity",
        action="store_true",
        help="Печатать similarity matrix и centralities для embedding/hybrid consensus",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Diagnostics LLM")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Consensus: {args.consensus}")

    patient_query = args.query.strip()
    if not patient_query:
        patient_query = input("\nВведите жалобу пациента:\n> ").strip()

    if not patient_query:
        print("Пустой запрос. Завершение.")
        return

    try:
        if args.mode == "baseline":
            run_baseline_pipeline(
                patient_query=patient_query,
                top_k=args.top_k,
                n_runs=args.n_runs,
                consensus_method=args.consensus,
                temperature=args.temperature,
                show_prompt=args.show_prompt,
                show_similarity=args.show_similarity,
            )
        elif args.mode == "cot":
            run_cot_mode(
                patient_query=patient_query,
                consensus_method=args.consensus,
                show_similarity=args.show_similarity,
            )
        elif args.mode == "dialogue":
            run_dialogue_mode(
                patient_query=patient_query,
                show_similarity=args.show_similarity,
            )
        else:
            raise ValueError(f"Неизвестный режим: {args.mode}")

    except Exception as e:
        print(f"\nОшибка пайплайна: {e}")


if __name__ == "__main__":
    main()