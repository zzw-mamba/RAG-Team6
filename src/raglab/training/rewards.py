from __future__ import annotations

from dataclasses import dataclass

from raglab.evaluation import exact_match, token_f1


@dataclass(frozen=True)
class RewardWeights:
    final_em: float = 1.0
    em_gain: float = 0.5
    f1_gain: float = 0.1
    revision_cost: float = 0.05
    invalid_penalty: float = 1.0


def outcome_reward(
    initial: str,
    final: str,
    gold: list[str],
    revised: bool,
    valid: bool = True,
    weights: RewardWeights | None = None,
) -> dict[str, float]:
    """One-step critic action reward, not a claim of full-trajectory RL training."""
    if not gold:
        raise ValueError("Reward computation requires training gold answers")
    weights = weights or RewardWeights()
    before, after = exact_match(initial, gold), exact_match(final, gold)
    pieces = {
        "final_em": weights.final_em * after,
        "em_gain": weights.em_gain * (after - before),
        "f1_gain": weights.f1_gain * (token_f1(final, gold) - token_f1(initial, gold)),
        "revision_cost": -weights.revision_cost * int(revised),
        "invalid_penalty": -weights.invalid_penalty * int(not valid),
    }
    return {**pieces, "total": sum(pieces.values())}
