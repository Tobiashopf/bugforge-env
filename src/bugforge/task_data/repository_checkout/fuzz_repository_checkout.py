import os
import random

from repository_checkout import checkout
from ref_repository_checkout import checkout as reference


def outcome(fn, lines, stock, member):
    try:
        return ('ok', fn(lines, stock, member))
    except Exception as exc:
        return ('error', type(exc).__name__)


def test_seeded_checkout_matches_reference():
    rng = random.Random(int(os.environ.get('BUGFORGE_FUZZ_SEED', '0')))
    for _ in range(48):
        stock = {sku: rng.randrange(0, 9) for sku in ('a', 'b', 'c')}
        lines = [(rng.choice(('a', 'b', 'c')), rng.randrange(-1, 1500),
                  rng.randrange(1, 5)) for _ in range(rng.randrange(0, 5))]
        member = rng.choice((True, False))
        assert outcome(checkout, lines, dict(stock), member) == outcome(
            reference, lines, dict(stock), member)
