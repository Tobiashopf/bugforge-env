import pytest
from repository_reconciliation import reconcile


def test_positive_payment():
    with pytest.raises(ValueError):
        reconcile([['a', 100]], [['p', 'a', 0]])


def test_conflict():
    with pytest.raises(ValueError):
        reconcile([['a', 100]], [['p', 'a', 20], ['p', 'a', 30]])


def test_overpayment():
    result = reconcile([['a', 100]], [['p', 'a', 150]])
    assert result['payments']['p']['applied'] == 100


def test_unapplied():
    result = reconcile([], [['p', 'missing', 75]])
    assert result['payments']['p']['unapplied'] == 75


def test_outstanding():
    assert reconcile([['a', 100]], [['p', 'a', 100]])['outstanding'] == {'a': 0}


def test_partial():
    assert reconcile([['a', 100]], [['p', 'a', 30], ['q', 'a', 50]]) == {
        'outstanding': {'a': 20}, 'payments': {
            'p': {'invoice': 'a', 'applied': 30, 'unapplied': 0},
            'q': {'invoice': 'a', 'applied': 50, 'unapplied': 0}}}


def test_duplicate():
    p = ['p', 'a', 30]
    assert reconcile([['a', 100]], [p, p]) == reconcile([['a', 100]], [p])


def test_multiple_invoices():
    result = reconcile([['a', 100], ['b', 90]], [['p', 'b', 30], ['q', 'a', 10]])
    assert result['outstanding'] == {'a': 90, 'b': 60}


def test_excess_after_settlement():
    result = reconcile([['a', 100]], [['p', 'a', 100], ['q', 'a', 20]])
    assert result['payments']['q'] == {'invoice': 'a', 'applied': 0, 'unapplied': 20}


def test_invalid_inputs():
    for invoices, payments in [([['a', True]], []), ([['a', 0]], []),
                               ([['a', 2], ['a', 2]], []), ([], [['p', '', 2]]),
                               ([], [['p', 'a', True]]), ([], [['p', 'a', -1]]),
                               ([], [['p', 'a']]), ({}, []), ([], {})]:
        with pytest.raises(ValueError):
            reconcile(invoices, payments)


def test_empty():
    assert reconcile([], []) == {'outstanding': {}, 'payments': {}}


def test_integer_precision():
    amount = 10**18 + 1
    assert reconcile([['a', amount]], [['p', 'a', amount - 1]])['outstanding'] == {'a': 1}
