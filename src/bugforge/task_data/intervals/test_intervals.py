import pytest
from intervals import contains, gaps, merge, total_covered


def test_merge_disjoint_keeps_both_sorted():
    assert merge([(5, 7), (1, 3)]) == [(1, 3), (5, 7)]


def test_merge_overlapping():
    assert merge([(1, 4), (2, 6)]) == [(1, 6)]


def test_merge_touching_intervals_join():
    assert merge([(1, 3), (3, 5)]) == [(1, 5)]


def test_merge_gap_of_one_stays_apart():
    assert merge([(1, 3), (4, 5)]) == [(1, 3), (4, 5)]


def test_merge_contained_interval_absorbed():
    assert merge([(1, 10), (2, 3)]) == [(1, 10)]


def test_merge_drops_empty_intervals():
    assert merge([(2, 2), (4, 6)]) == [(4, 6)]


def test_merge_chain():
    assert merge([(1, 2), (2, 3), (3, 4), (6, 8)]) == [(1, 4), (6, 8)]


def test_merge_empty_input():
    assert merge([]) == []


def test_merge_invalid_interval_raises():
    with pytest.raises(ValueError):
        merge([(3, 1)])


def test_total_covered_counts_union_once():
    assert total_covered([(1, 4), (2, 6)]) == 5


def test_total_covered_disjoint():
    assert total_covered([(0, 2), (10, 13)]) == 5


def test_total_covered_empty():
    assert total_covered([]) == 0


def test_contains_start_is_inclusive():
    assert contains([(2, 5)], 2) is True


def test_contains_end_is_exclusive():
    assert contains([(2, 5)], 5) is False


def test_contains_point_before_start():
    assert contains([(2, 5)], 1) is False


def test_contains_last_point():
    assert contains([(2, 5)], 4) is True


def test_contains_checks_every_interval():
    assert contains([(0, 1), (10, 20)], 15) is True


def test_contains_invalid_interval_raises():
    with pytest.raises(ValueError):
        contains([(5, 2)], 3)


def test_gaps_between_intervals():
    assert gaps([(2, 4), (6, 8)], 0, 10) == [(0, 2), (4, 6), (8, 10)]


def test_gaps_fully_covered():
    assert gaps([(0, 10)], 2, 8) == []


def test_gaps_no_intervals():
    assert gaps([], 3, 7) == [(3, 7)]


def test_gaps_ignores_intervals_outside_range():
    assert gaps([(0, 2), (20, 30)], 5, 10) == [(5, 10)]


def test_gaps_interval_ending_at_lo_is_ignored():
    assert gaps([(0, 5)], 5, 8) == [(5, 8)]


def test_gaps_interval_starting_at_hi_is_ignored():
    assert gaps([(8, 12)], 5, 8) == [(5, 8)]


def test_gaps_partial_overlap_at_edges():
    assert gaps([(0, 6), (9, 15)], 5, 10) == [(6, 9)]


def test_gaps_empty_range():
    assert gaps([(1, 2)], 4, 4) == []


def test_gaps_invalid_range_raises():
    with pytest.raises(ValueError):
        gaps([], 5, 4)
