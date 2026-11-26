import os
import random

from repository_ledger import replay
from ref_repository_ledger import replay as reference


def outcome(fn, opening, transactions):
    try:
        return ('ok', fn(dict(opening), list(transactions)))
    except Exception as exc:
        return ('error', type(exc).__name__)


def test_seeded_ledger_matches_reference():
    rng = random.Random(int(os.environ.get('BUGFORGE_FUZZ_SEED', '0')))
    for i in range(64):
        opening = {a: rng.randrange(1, 10000) for a in ('a', 'b', 'c')}
        src, dst = rng.sample(list(opening), 2)
        amount = rng.randint(1, opening[src])
        tx = ('first', src, dst, amount)
        transactions = [tx]
        if i % 4 == 0:
            transactions += [tx]
        elif i % 4 == 1:
            transactions += [('first', src, dst, amount + 1)]
        elif i % 4 == 2:
            transactions += [('second', src, dst, opening[src] + 1)]
        else:
            transactions += [('second', dst, src, amount)]
        got = outcome(replay, opening, transactions)
        assert got == outcome(reference, opening, transactions)
        if got[0] == 'ok':
            assert sum(got[1]['balances'].values()) == sum(opening.values())
            assert all(v >= 0 for v in got[1]['balances'].values())
