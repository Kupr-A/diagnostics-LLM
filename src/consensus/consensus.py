from collections import Counter, defaultdict

from src.consensus.similarity_utils import select_central_answer


def normalize_name(name: str) -> str:
    return " ".join(name.lower().strip().split())


def deduplicate_keep_order(items: list[str]) -> list[str]:
    result = []
    seen = set()

    for item in items:
        key = item.lower().strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(item.strip())

    return result


def collect_side_fields(parsed_outputs: list[dict]) -> dict:
    red_flags = []
    next_questions = []

    for output in parsed_outputs:
        red_flags.extend(output.get("red_flags", []))
        next_questions.extend(output.get("next_questions", []))

    return {
        "red_flags": deduplicate_keep_order(red_flags)[:10],
        "next_questions": deduplicate_keep_order(next_questions)[:5],
    }


def frequency_consensus(parsed_outputs: list[dict]) -> dict:
    total_runs = max(len(parsed_outputs), 1)
    counter = Counter()
    reasons = defaultdict(list)

    for output in parsed_outputs:
        for hyp in output.get("hypotheses", []):
            name = normalize_name(hyp.get("name", ""))
            if not name:
                continue
            counter[name] += 1
            if hyp.get("reason"):
                reasons[name].append(hyp["reason"])

    hypotheses = []
    for name, count in counter.most_common(3):
        hypotheses.append({
            "name": name,
            "mentions": count,
            "confidence": round(count / total_runs, 4),
            "reason": reasons[name][0] if reasons[name] else "",
        })

    side = collect_side_fields(parsed_outputs)

    return {
        "method": "frequency",
        "hypotheses": hypotheses,
        "red_flags": side["red_flags"],
        "next_questions": side["next_questions"],
    }


def weighted_consensus(parsed_outputs: list[dict]) -> dict:
    total_runs = max(len(parsed_outputs), 1)

    scores = defaultdict(float)
    reasons = defaultdict(list)
    mentions = defaultdict(int)

    for output in parsed_outputs:
        for hyp in output.get("hypotheses", []):
            name = normalize_name(hyp.get("name", ""))
            if not name:
                continue

            scores[name] += hyp.get("probability", 0.0)
            mentions[name] += 1

            if hyp.get("reason"):
                reasons[name].append(hyp["reason"])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    hypotheses = []
    for name, score in ranked:
        hypotheses.append({
            "name": name,
            "score": round(score, 4),
            "mentions": mentions[name],
            "confidence": round(score / total_runs, 4),
            "reason": reasons[name][0] if reasons[name] else "",
        })

    side = collect_side_fields(parsed_outputs)

    return {
        "method": "weighted",
        "hypotheses": hypotheses,
        "red_flags": side["red_flags"],
        "next_questions": side["next_questions"],
    }


def embedding_consensus(parsed_outputs: list[dict]) -> dict:
    central = select_central_answer(parsed_outputs)
    central_output = central["central_output"]

    if not central_output:
        return {
            "method": "embedding",
            "hypotheses": [],
            "red_flags": [],
            "next_questions": [],
            "central_index": None,
            "similarity_matrix": [],
            "centralities": [],
        }

    hypotheses = []
    for hyp in central_output.get("hypotheses", [])[:3]:
        hypotheses.append({
            "name": normalize_name(hyp.get("name", "")),
            "confidence": hyp.get("probability", 0.0),
            "reason": hyp.get("reason", ""),
        })

    return {
        "method": "embedding",
        "hypotheses": hypotheses,
        "red_flags": central_output.get("red_flags", []),
        "next_questions": central_output.get("next_questions", []),
        "central_index": central["central_index"],
        "similarity_matrix": central["similarity_matrix"],
        "centralities": central["centralities"],
    }


def hybrid_consensus(parsed_outputs: list[dict]) -> dict:
    total_runs = max(len(parsed_outputs), 1)
    central = select_central_answer(parsed_outputs)
    central_output = central["central_output"]
    centralities = central["centralities"]

    freq = Counter()
    prob_sum = defaultdict(float)
    reasons = defaultdict(list)
    mentions = defaultdict(int)
    centrality_support = defaultdict(list)

    for output_idx, output in enumerate(parsed_outputs):
        output_centrality = centralities[output_idx] if output_idx < len(centralities) else 0.0

        for hyp in output.get("hypotheses", []):
            name = normalize_name(hyp.get("name", ""))
            if not name:
                continue

            freq[name] += 1
            mentions[name] += 1
            prob_sum[name] += hyp.get("probability", 0.0)
            centrality_support[name].append(output_centrality)

            if hyp.get("reason"):
                reasons[name].append(hyp["reason"])

    central_names = set()
    if central_output:
        central_names = {
            normalize_name(h.get("name", ""))
            for h in central_output.get("hypotheses", [])
            if h.get("name")
        }

    hypotheses = []
    for name in freq.keys():
        freq_score = freq[name] / total_runs
        weighted_score = prob_sum[name] / total_runs
        centrality_score = (
            sum(centrality_support[name]) / len(centrality_support[name])
            if centrality_support[name] else 0.0
        )

        hybrid_score = (
            0.4 * freq_score +
            0.4 * weighted_score +
            0.2 * centrality_score
        )

        if name in central_names:
            hybrid_score = min(hybrid_score + 0.05, 1.0)

        hypotheses.append({
            "name": name,
            "mentions": mentions[name],
            "confidence": round(hybrid_score, 4),
            "reason": reasons[name][0] if reasons[name] else "",
            "freq_score": round(freq_score, 4),
            "weighted_score": round(weighted_score, 4),
            "centrality_score": round(centrality_score, 4),
        })

    hypotheses = sorted(hypotheses, key=lambda x: x["confidence"], reverse=True)[:3]

    side = collect_side_fields(parsed_outputs)

    return {
        "method": "hybrid",
        "hypotheses": hypotheses,
        "red_flags": side["red_flags"],
        "next_questions": side["next_questions"],
        "central_index": central["central_index"],
        "similarity_matrix": central["similarity_matrix"],
        "centralities": central["centralities"],
    }


def build_consensus(parsed_outputs: list[dict], method: str = "hybrid") -> dict:
    if method == "frequency":
        return frequency_consensus(parsed_outputs)
    if method == "weighted":
        return weighted_consensus(parsed_outputs)
    if method == "embedding":
        return embedding_consensus(parsed_outputs)
    if method == "hybrid":
        return hybrid_consensus(parsed_outputs)

    raise ValueError(f"Неизвестный метод consensus: {method}")