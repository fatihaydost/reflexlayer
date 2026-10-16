from dataclasses import dataclass


@dataclass(frozen=True)
class S1Prediction:
    choice: str
    probs: dict[str, float]
    p_max: float
    cost: float
    latency_ms: float


@dataclass(frozen=True)
class LLMResult:
    choice: str
    cost: float
    latency_ms: float


@dataclass(frozen=True)
class Decision:
    choice: str
    p: float | None
    tier: str  # "rule" | "s1" | "llm"
    cost: float
    latency_ms: float


@dataclass
class Policy:
    decision_type: str
    mode: str = "shadow"  # "shadow" | "active"
    threshold: float | None = None
    alpha: float = 0.02  # error budget: max disagreement rate with the LLM
    delta: float = 0.05  # 1 - confidence of the budget guarantee
    sample_rate: float = 0.05  # share of trusted calls still sent to the LLM for audit
    destructive: bool = False
    note: str = ""
