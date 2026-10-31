import pytest
from textstats import average_word_length, longest_word, tokenize, top_k, word_counts


def test_tokenize_lowercases():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_splits_on_punctuation():
    assert tokenize("a,b;c") == ["a", "b", "c"]


def test_tokenize_keeps_digits():
    assert tokenize("route 66") == ["route", "66"]


def test_tokenize_last_word_without_trailing_space():
    assert tokenize("end") == ["end"]


def test_tokenize_multiple_separators():
    assert tokenize("  a -- b  ") == ["a", "b"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_word_counts():
    assert word_counts("a b a A") == {"a": 3, "b": 1}


def test_word_counts_single_word():
    assert word_counts("x") == {"x": 1}


def test_top_k_orders_by_count():
    assert top_k("b b b a a c") == [("b", 3), ("a", 2), ("c", 1)]


def test_top_k_breaks_ties_alphabetically():
    assert top_k("z y x z y x w", k=2) == [("x", 2), ("y", 2)]


def test_top_k_default_is_three():
    assert len(top_k("a b c d e")) == 3


def test_top_k_one_is_allowed():
    assert top_k("q q r", k=1) == [("q", 2)]


def test_top_k_zero_raises():
    with pytest.raises(ValueError):
        top_k("a", k=0)


def test_top_k_more_than_available():
    assert top_k("a b", k=5) == [("a", 1), ("b", 1)]


def test_average_word_length():
    assert average_word_length("ab abcd") == 3.0


def test_average_word_length_rounds():
    assert average_word_length("a bb bb") == 1.67


def test_average_word_length_single_word():
    assert average_word_length("hello") == 5.0


def test_average_word_length_no_words():
    assert average_word_length("!!") == 0.0


def test_longest_word():
    assert longest_word("a abc ab") == "abc"


def test_longest_word_first_wins_tie():
    assert longest_word("cat dog") == "cat"


def test_longest_word_single_letter():
    assert longest_word("x") == "x"


def test_longest_word_no_words():
    assert longest_word("...") == ""
