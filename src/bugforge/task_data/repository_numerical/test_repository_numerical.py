import pytest
from repository_numerical import fit, transform, run


def test_mean():
    assert fit([2, 4, None])['mean'] == pytest.approx(3)


def test_population_scale():
    assert fit([-2, 2])['scale'] == pytest.approx(2)


def test_missing_imputation():
    assert transform([None], {'mean': 5, 'scale': 2}) == [0.0]


def test_standardization():
    assert transform([7], {'mean': 3, 'scale': 2}) == pytest.approx([2])


def test_no_inference_leakage():
    result = run([-1, 1], [9])
    assert result['inference'] == pytest.approx(transform([9], result['state']))


def test_constant_training():
    result = run([5, 5, None], [5, 7])
    assert result['state'] == {'mean': 5, 'scale': 1}
    assert result['inference'] == pytest.approx([0, 2])


def test_missing_training_error():
    for values in ([], [None, None]):
        with pytest.raises(ValueError):
            fit(values)


def test_invalid_observation():
    for value in (True, '3', 10**13):
        with pytest.raises(ValueError):
            fit([1, value])
        with pytest.raises(ValueError):
            transform([value], {'mean': 0, 'scale': 1})


def test_invalid_scale():
    for scale in (0, -1, True):
        with pytest.raises(ValueError):
            transform([1], {'mean': 0, 'scale': scale})


def test_permutation():
    a = fit([3, None, 8, 1])
    b = fit([1, 8, 3, None])
    assert a == pytest.approx(b)
    state = {'mean': 4, 'scale': 2}
    assert transform([2, 6, None], state) == pytest.approx([-1, 1, 0])
    assert transform([None, 6, 2], state) == pytest.approx([0, 1, -1])


def test_empty_inference():
    assert run([1, 3], [])['inference'] == []


def test_training_inference_state():
    result = run([1, 3, None], [4, None])
    assert result['state'] == pytest.approx({'mean': 2, 'scale': 1})
    assert result['training'] == pytest.approx([-1, 1, 0])
    assert result['inference'] == pytest.approx([2, 0])
