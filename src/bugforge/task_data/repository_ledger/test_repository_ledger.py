import pytest
from repository_ledger import replay


def test_positive_amount():
    with pytest.raises(ValueError):
        replay({'a': 100, 'b': 0}, [('x', 'a', 'b', 0)])


def test_exact_funds():
    assert replay({'a': 100, 'b': 0}, [('x', 'a', 'b', 100)]) == {
        'balances': {'a': 0, 'b': 100}, 'applied': ['x']}


def test_credit_destination():
    assert replay({'a': 100, 'b': 10}, [('x', 'a', 'b', 20)]) == {
        'balances': {'a': 80, 'b': 30}, 'applied': ['x']}


def test_duplicate_id():
    tx = ('x', 'a', 'b', 20)
    result = replay({'a': 100, 'b': 0}, [tx, tx])
    assert result['balances']['a'] == 80 and result['applied'] == ['x']


def test_conflicting_id():
    with pytest.raises(ValueError):
        replay({'a': 100, 'b': 0}, [('x', 'a', 'b', 20), ('x', 'a', 'b', 21)])


def test_insufficient_funds():
    with pytest.raises(ValueError):
        replay({'a': 10, 'b': 0}, [('x', 'a', 'b', 11)])


def test_invalid_accounts():
    for source, target in [('a', 'a'), ('a', 'c'), ('c', 'b')]:
        with pytest.raises(ValueError):
            replay({'a': 100, 'b': 0}, [('x', source, target, 1)])


def test_amount_type():
    for amount in (True, 1.5, -1):
        with pytest.raises(ValueError):
            replay({'a': 100, 'b': 0}, [('x', 'a', 'b', amount)])


def test_invalid_opening_and_id():
    with pytest.raises(ValueError):
        replay({'a': -1, 'b': 0}, [])
    with pytest.raises(ValueError):
        replay({'a': 100, 'b': 0}, [('', 'a', 'b', 1)])


def test_order_and_conservation():
    result = replay({'a': 100, 'b': 0, 'c': 0}, [
        ('x', 'a', 'b', 60), ('y', 'b', 'c', 30), ('z', 'c', 'a', 10)])
    assert result == {'balances': {'a': 50, 'b': 30, 'c': 20}, 'applied': ['x', 'y', 'z']}
    assert sum(result['balances'].values()) == 100


def test_empty_and_distinct_ids():
    assert replay({'a': 100, 'b': 0}, []) == {'balances': {'a': 100, 'b': 0}, 'applied': []}
    result = replay({'a': 100, 'b': 0}, [('x', 'a', 'b', 5), ('y', 'a', 'b', 5)])
    assert result == {'balances': {'a': 90, 'b': 10}, 'applied': ['x', 'y']}


def test_retry_after_spend():
    tx = ('x', 'a', 'b', 60)
    result = replay({'a': 100, 'b': 0}, [tx, ('y', 'a', 'b', 30), tx])
    assert result == {'balances': {'a': 10, 'b': 90}, 'applied': ['x', 'y']}
