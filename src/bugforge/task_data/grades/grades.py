"""Grading helpers: letter grades, weighted averages, curving and ranking."""


def letter(score):
    """Letter grade for a score in [0, 100]: A >= 90, B >= 80, C >= 70, D >= 60."""
    if score < 0 or score > 100:
        raise ValueError("score must be between 0 and 100")
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def weighted_average(scores, weights):
    """Weighted mean rounded to 2 decimals."""
    if len(scores) != len(weights) or len(scores) == 0:
        raise ValueError("scores and weights must be non-empty and the same length")
    total_weight = 0
    total = 0
    for s, w in zip(scores, weights):
        if w < 0:
            raise ValueError("weights must be non-negative")
        total += s * w
        total_weight += w
    if total_weight == 0:
        raise ValueError("weights must not all be zero")
    return round(total / total_weight, 2)


def curve(scores, target_max=100):
    """Shift every score so the best becomes target_max, clamped to [0, 100]."""
    if not scores:
        return []
    shift = target_max - max(scores)
    result = []
    for s in scores:
        result.append(min(100, max(0, s + shift)))
    return result


def rank(scores):
    """Competition ranking: highest is 1, ties share a rank, the next rank skips."""
    ranks = []
    for s in scores:
        higher = 0
        for other in scores:
            if other > s:
                higher += 1
        ranks.append(higher + 1)
    return ranks


def passed_count(scores, cutoff=60):
    """How many scores are at or above the cutoff."""
    count = 0
    for s in scores:
        if s >= cutoff:
            count += 1
    return count
