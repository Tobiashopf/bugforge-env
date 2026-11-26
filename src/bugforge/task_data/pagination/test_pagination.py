import pytest
from pagination import has_next, has_previous, page_count, page_slice, page_window


def test_page_count_exact_multiple():
    assert page_count(20, 10) == 2


def test_page_count_rounds_up():
    assert page_count(21, 10) == 3


def test_page_count_one_item():
    assert page_count(1, 10) == 1


def test_page_count_zero_items():
    assert page_count(0, 10) == 0


def test_page_count_per_page_one():
    assert page_count(7, 1) == 7


def test_page_count_per_page_zero_raises():
    with pytest.raises(ValueError):
        page_count(5, 0)


def test_page_count_negative_items_raises():
    with pytest.raises(ValueError):
        page_count(-1, 5)


def test_page_slice_first_page():
    assert page_slice(list(range(10)), 1, 4) == [0, 1, 2, 3]


def test_page_slice_middle_page():
    assert page_slice(list(range(10)), 2, 4) == [4, 5, 6, 7]


def test_page_slice_partial_last_page():
    assert page_slice(list(range(10)), 3, 4) == [8, 9]


def test_page_slice_past_the_end_is_empty():
    assert page_slice(list(range(10)), 4, 4) == []


def test_page_slice_page_zero_raises():
    with pytest.raises(ValueError):
        page_slice([1, 2], 0, 1)


def test_page_window_centered():
    assert page_window(10, 20) == [8, 9, 10, 11, 12]


def test_page_window_at_start():
    assert page_window(1, 20) == [1, 2, 3, 4, 5]


def test_page_window_near_start():
    assert page_window(2, 20) == [1, 2, 3, 4, 5]


def test_page_window_at_end():
    assert page_window(20, 20) == [16, 17, 18, 19, 20]


def test_page_window_fewer_pages_than_width():
    assert page_window(2, 3) == [1, 2, 3]


def test_page_window_custom_width():
    assert page_window(5, 9, width=3) == [4, 5, 6]


def test_page_window_current_zero_raises():
    with pytest.raises(ValueError):
        page_window(0, 5)


def test_page_window_current_past_total_raises():
    with pytest.raises(ValueError):
        page_window(6, 5)


def test_page_window_no_pages_raises():
    with pytest.raises(ValueError):
        page_window(1, 0)


def test_has_next():
    assert has_next(4, 5) is True
    assert has_next(5, 5) is False


def test_has_previous():
    assert has_previous(2) is True
    assert has_previous(1) is False
