def f1_score(
    precision: float,
    recall: float,
) -> float:
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)
