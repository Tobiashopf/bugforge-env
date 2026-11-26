import pytest
from repository_checkout import checkout


def test_price_quantity():
    assert checkout([('x', 200, 3)], {'x': 10})['total'] == 600


def test_member_threshold():
    assert checkout([('x', 500, 2)], {'x': 4}, True)['total'] == 900


def test_exact_stock():
    assert checkout([('x', 200, 3)], {'x': 3})['remaining'] == {'x': 0}


def test_duplicate_sku():
    with pytest.raises(ValueError):
        checkout([('x', 100, 2), ('x', 100, 2)], {'x': 3})


def test_no_member_discount():
    assert checkout([('x', 500, 2)], {'x': 3})['total'] == 1000


def test_rounding_cents():
    assert checkout([('x', 1009, 1)], {'x': 3}, True)['total'] == 909


def test_reject_invalid_line():
    with pytest.raises(ValueError):
        checkout([('x', -1, 2)], {'x': 3})


def test_reject_insufficient_second_item():
    stock = {'x': 3, 'y': 1}
    with pytest.raises(ValueError):
        checkout([('x', 2, 1), ('y', 2, 2)], stock)


def test_multi_item():
    assert checkout([('x', 200, 3), ('y', 100, 2)], {'x': 5, 'y': 2}) == {
        'subtotal': 800, 'discount': 0, 'total': 800, 'remaining': {'x': 2, 'y': 0}}


def test_empty():
    assert checkout([], {'x': 2}, True)['total'] == 0
