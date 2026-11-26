import os
import random
from repository_reconciliation import reconcile
from ref_repository_reconciliation import reconcile as reference


def test_seeded_reconciliation():
    rng = random.Random(int(os.environ.get('BUGFORGE_FUZZ_SEED', '0')))
    for _ in range(64):
        invoices = [[str(i), rng.randrange(1, 2000)] for i in range(rng.randrange(6))]
        totals = dict(invoices)
        payments = [[str(i), str(rng.randrange(7)), rng.randrange(1, 3000)]
                    for i in range(rng.randrange(1, 12))]
        unique = {key: (invoice, cents) for key, invoice, cents in payments}
        result = reconcile(invoices, payments)
        assert result == reference(invoices, payments)
        assert reconcile(invoices, payments + [payments[0]]) == result
        assert reconcile(list(reversed(invoices)), payments) == result
        assert reconcile(invoices, list(reversed(payments)))['outstanding'] == result['outstanding']
        assert set(result['payments']) == set(unique)
        for key, (invoice, cents) in unique.items():
            allocation = result['payments'][key]
            assert allocation['invoice'] == invoice
            assert 0 <= allocation['applied'] <= cents
            assert allocation['applied'] + allocation['unapplied'] == cents
            if invoice not in totals:
                assert allocation['applied'] == 0
        for invoice, total in totals.items():
            applied = sum(p['applied'] for p in result['payments'].values() if p['invoice'] == invoice)
            assert 0 <= result['outstanding'][invoice] <= total
            assert applied + result['outstanding'][invoice] == total
