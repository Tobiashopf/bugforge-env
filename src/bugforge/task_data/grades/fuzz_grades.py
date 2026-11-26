"""Differential fuzz test against the reference (see fuzz_inventory.py)."""
import os
import random

import grades
import ref_grades

SEED = int(os.environ.get("BUGFORGE_FUZZ_SEED", "0"))


def outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__)


def test_random_inputs_match_reference():
    rng = random.Random(SEED)
    edges = [-1, 0, 59, 60, 69, 70, 79, 80, 89, 90, 100, 101]
    for _ in range(400):
        s = rng.choice(edges + [rng.randint(-5, 105), rng.uniform(-1, 101)])
        assert outcome(grades.letter, s) == outcome(ref_grades.letter, s)
        n = rng.randint(0, 5)
        scores = [rng.randint(0, 100) for _ in range(n)]
        weights = [rng.choice([0, 0, 1, 2, 3, -1]) for _ in range(n + rng.choice([0, 0, 0, 1]))]
        assert outcome(grades.weighted_average, scores, weights) == outcome(
            ref_grades.weighted_average, scores, weights)
        target = rng.choice([0, 50, 90, 100, 130])
        assert outcome(grades.curve, scores, target) == outcome(ref_grades.curve, scores, target)
        assert outcome(grades.curve, scores) == outcome(ref_grades.curve, scores)
        dup = scores + [rng.choice(scores)] if scores else scores
        assert outcome(grades.rank, dup) == outcome(ref_grades.rank, dup)
        cutoff = rng.choice([0, 50, 60, 70, 100])
        assert outcome(grades.passed_count, dup, cutoff) == outcome(
            ref_grades.passed_count, dup, cutoff)
        assert outcome(grades.passed_count, dup) == outcome(ref_grades.passed_count, dup)
