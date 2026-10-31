"""Differential fuzz test against the reference (see fuzz_inventory.py)."""
import os
import random

import ref_textstats
import textstats

SEED = int(os.environ.get("BUGFORGE_FUZZ_SEED", "0"))
ALPHABET = "aabbcdeXYZ19 ,.-!\t"


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__)


def test_random_text_matches_reference():
    rng = random.Random(SEED)
    for _ in range(400):
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 30)))
        for name in ["tokenize", "word_counts", "average_word_length", "longest_word"]:
            assert outcome(getattr(textstats, name), text) == outcome(
                getattr(ref_textstats, name), text), name
        k = rng.randint(-1, 5)
        assert outcome(textstats.top_k, text, k) == outcome(ref_textstats.top_k, text, k)
        assert outcome(textstats.top_k, text) == outcome(ref_textstats.top_k, text)
