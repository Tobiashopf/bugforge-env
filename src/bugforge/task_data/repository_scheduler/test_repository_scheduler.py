import pytest
from repository_scheduler import schedule


def job(key, priority=0, deps=None, failures=0):
    return {'id': key, 'priority': priority, 'deps': [] if deps is None else deps,
            'failures': failures}


def test_priority_type():
    with pytest.raises(ValueError):
        schedule([job('a', True)], 2)


def test_conflict():
    with pytest.raises(ValueError):
        schedule([job('a'), job('a', 1)], 2)


def test_tie_break():
    assert schedule([job('b'), job('a')], 2)['events'] == [
        ['a', 1, 'succeeded'], ['b', 1, 'succeeded']]


def test_retry_limit():
    assert schedule([job('a', failures=5)], 2) == {
        'states': {'a': 'failed'}, 'attempts': {'a': 2},
        'events': [['a', 1, 'pending'], ['a', 2, 'failed']]}


def test_dependency_gate():
    assert schedule([job('b', 9, ['a']), job('a')], 2)['events'] == [
        ['a', 1, 'succeeded'], ['b', 1, 'succeeded']]


def test_priority_order():
    assert schedule([job('a', -1), job('b', 2)], 1)['events'][0][0] == 'b'


def test_recover():
    assert schedule([job('a', failures=1)], 2)['events'] == [
        ['a', 1, 'pending'], ['a', 2, 'succeeded']]


def test_duplicate():
    assert schedule([job('a'), job('a')], 1)['attempts'] == {'a': 1}


def test_failed_dependency():
    result = schedule([job('a', failures=3), job('b', deps=['a'])], 1)
    assert result['states'] == {'a': 'failed', 'b': 'blocked'}
    assert result['attempts'] == {'a': 1, 'b': 0}


def test_cycle():
    result = schedule([job('a', deps=['b']), job('b', deps=['a']), job('c')], 2)
    assert result['states'] == {'a': 'blocked', 'b': 'blocked', 'c': 'succeeded'}
    assert result['events'] == [['c', 1, 'succeeded']]


def test_invalid_inputs():
    for jobs, limit in [([job('a', deps=['unknown'])], 1), ([job('a', failures=-1)], 2),
                        ([job('')], 1), ([job('a')], True), ([job('a')], 0),
                        ([job('a', deps=['a', 'a'])], 1)]:
        with pytest.raises(ValueError):
            schedule(jobs, limit)


def test_empty():
    assert schedule([], 1) == {'states': {}, 'attempts': {}, 'events': []}
