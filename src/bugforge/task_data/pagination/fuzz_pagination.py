"""Differential fuzz test against the reference (see fuzz_inventory.py)."""
import os
import random

import pagination
import ref_pagination

SEED = int(os.environ.get("BUGFORGE_FUZZ_SEED", "0"))


def outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__)


def test_random_inputs_match_reference():
    rng = random.Random(SEED)
    for _ in range(500):
        total, per = rng.randint(-1, 30), rng.randint(-1, 7)
        assert outcome(pagination.page_count, total, per) == outcome(
            ref_pagination.page_count, total, per)
        items = list(range(rng.randint(0, 25)))
        page, per = rng.randint(-1, 8), rng.randint(1, 7)
        assert outcome(pagination.page_slice, items, page, per) == outcome(
            ref_pagination.page_slice, items, page, per)
        tp = rng.randint(0, 12)
        cur = rng.randint(-1, 13)
        width = rng.choice([1, 2, 3, 4, 5, 7])
        assert outcome(pagination.page_window, cur, tp, width) == outcome(
            ref_pagination.page_window, cur, tp, width)
        assert outcome(pagination.page_window, cur, tp) == outcome(
            ref_pagination.page_window, cur, tp)
        assert pagination.has_next(cur, tp) == ref_pagination.has_next(cur, tp)
        assert pagination.has_previous(cur) == ref_pagination.has_previous(cur)
