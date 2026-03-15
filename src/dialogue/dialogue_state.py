from dataclasses import dataclass, field


@dataclass
class DialogueState:
    original_query: str
    enriched_query: str = ""
    intake_answers: dict = field(default_factory=dict)
    asked_questions: list = field(default_factory=list)
    answer_history: list = field(default_factory=list)
    current_retrieved_cases: list = field(default_factory=list)
    current_consensus: dict = field(default_factory=dict)
    turn_count: int = 0
    max_turns: int = 7
    stop_reason: str = ""