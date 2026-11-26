import pytest
from grades import curve, letter, passed_count, rank, weighted_average


def test_letter_a_at_boundary():
    assert letter(90) == "A"


def test_letter_b_just_below_a():
    assert letter(89.99) == "B"


def test_letter_b_at_boundary():
    assert letter(80) == "B"


def test_letter_c_at_boundary():
    assert letter(70) == "C"


def test_letter_c_just_below_b():
    assert letter(79) == "C"


def test_letter_d_at_boundary():
    assert letter(60) == "D"


def test_letter_f_just_below_d():
    assert letter(59.5) == "F"


def test_letter_accepts_extremes():
    assert letter(0) == "F"
    assert letter(100) == "A"


def test_letter_above_range_raises():
    with pytest.raises(ValueError):
        letter(100.5)


def test_letter_below_range_raises():
    with pytest.raises(ValueError):
        letter(-1)


def test_weighted_average():
    assert weighted_average([80, 90], [1, 3]) == 87.5


def test_weighted_average_rounds():
    assert weighted_average([1, 2, 2], [1, 1, 1]) == 1.67


def test_weighted_average_zero_weight_ignored():
    assert weighted_average([50, 100], [0, 2]) == 100.0


def test_weighted_average_length_mismatch_raises():
    with pytest.raises(ValueError):
        weighted_average([1, 2], [1])


def test_weighted_average_empty_raises():
    with pytest.raises(ValueError):
        weighted_average([], [])


def test_weighted_average_negative_weight_raises():
    with pytest.raises(ValueError):
        weighted_average([1, 2], [1, -1])


def test_weighted_average_all_zero_weights_raises():
    with pytest.raises(ValueError):
        weighted_average([1, 2], [0, 0])


def test_curve_shifts_to_target():
    assert curve([70, 85, 60]) == [85, 100, 75]


def test_curve_custom_target():
    assert curve([50, 40], target_max=90) == [90, 80]


def test_curve_clamps_at_zero():
    assert curve([100, 5], target_max=90) == [90, 0]


def test_curve_clamps_at_hundred():
    assert curve([80, 70], target_max=120) == [100, 100]


def test_curve_empty():
    assert curve([]) == []


def test_rank_distinct():
    assert rank([70, 90, 80]) == [3, 1, 2]


def test_rank_ties_share_and_skip():
    assert rank([90, 80, 90, 70]) == [1, 3, 1, 4]


def test_passed_count_cutoff_inclusive():
    assert passed_count([59, 60, 61]) == 2


def test_passed_count_custom_cutoff():
    assert passed_count([50, 70, 90], cutoff=70) == 2
