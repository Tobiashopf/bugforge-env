"""Differential fuzz test against the reference (see fuzz_inventory.py)."""
import os
import random

import intervals
import ref_intervals

SEED = int(os.environ.get("BUGFORGE_FUZZ_SEED", "0"))


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__)


def random_intervals(rng):
    out = []
    for _ in range(rng.randint(0, 6)):
        a, b = rng.randint(0, 15), rng.randint(0, 15)
        if rng.random() < 0.9:
            a, b = min(a, b), max(a, b)
        out.append((a, b))
    return out


def test_random_inputs_match_reference():
    rng = random.Random(SEED)
    for _ in range(400):
        ivs = random_intervals(rng)
        point = rng.randint(-1, 16)
        lo, hi = rng.randint(-1, 16), rng.randint(-1, 16)
        if rng.random() < 0.9:
            lo, hi = min(lo, hi), max(lo, hi)
        assert outcome(intervals.merge, list(ivs)) == outcome(ref_intervals.merge, list(ivs))
        assert outcome(intervals.total_covered, list(ivs)) == outcome(
            ref_intervals.total_covered, list(ivs))
        assert outcome(intervals.contains, list(ivs), point) == outcome(
            ref_intervals.contains, list(ivs), point)
        assert outcome(intervals.gaps, list(ivs), lo, hi) == outcome(
            ref_intervals.gaps, list(ivs), lo, hi)
