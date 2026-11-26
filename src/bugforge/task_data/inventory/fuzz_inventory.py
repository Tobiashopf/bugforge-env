"""Differential fuzz test: the candidate must behave like the reference.

Hidden from the agent. The verifier places the original module next to it as
`ref_inventory.py` and sets BUGFORGE_FUZZ_SEED.
"""
import os
import random

import inventory
import ref_inventory

SEED = int(os.environ.get("BUGFORGE_FUZZ_SEED", "0"))


def outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - the exception type is the outcome
        return ("err", type(exc).__name__)


def test_random_operation_sequences_match_reference():
    rng = random.Random(SEED)
    skus = ["A", "B", "C", "D"]
    prices = [0.0, 0.5, 2.5, 10.0, 25.0]
    for _ in range(150):
        cand, ref = inventory.Inventory(), ref_inventory.Inventory()
        for _ in range(12):
            op = rng.choice(["add", "add", "remove", "query"])
            sku = rng.choice(skus)
            if op == "add":
                args = (sku, rng.randint(-1, 6), rng.choice(prices + [-1.0]),
                        rng.choice(["x", "X", "y"]))
                assert outcome(cand.add, *args) == outcome(ref.add, *args)
            elif op == "remove":
                args = (sku, rng.randint(-1, 7))
                assert outcome(cand.remove, *args) == outcome(ref.remove, *args)
            for s in skus:
                assert cand.quantity(s) == ref.quantity(s)
        threshold = rng.choice([0.0, 10.0, 50.0, 100.0, ref.total_value(1e18)])
        rate = rng.choice([0.0, 0.1, 0.5])
        assert cand.total_value(threshold, rate) == ref.total_value(threshold, rate)
        assert cand.total_value() == ref.total_value()
        limit = rng.randint(0, 8)
        assert cand.low_stock(limit) == ref.low_stock(limit)
        assert cand.low_stock() == ref.low_stock()
        cat = rng.choice(["x", "X", "y", "z"])
        assert cand.by_category(cat) == ref.by_category(cat)
