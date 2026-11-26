"""Word statistics over plain text."""


def tokenize(text):
    """Lowercase words made of letters and digits; everything else separates."""
    words = []
    current = ""
    for ch in text:
        if ch.isalnum():
            current += ch.lower()
        elif current:
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return words


def word_counts(text):
    """Occurrences of each word."""
    counts = {}
    for word in tokenize(text):
        counts[word] = counts.get(word, 0) + 1
    return counts


def top_k(text, k=3):
    """The k most frequent words as (word, count), ties broken alphabetically."""
    if k < 1:
        raise ValueError("k must be at least 1")
    pairs = sorted(word_counts(text).items())
    pairs = sorted(pairs, key=lambda p: p[1], reverse=True)
    return pairs[:k]


def average_word_length(text):
    """Mean word length rounded to 2 decimals; 0.0 when there are no words."""
    words = tokenize(text)
    if len(words) == 0:
        return 0.0
    total = 0
    for word in words:
        total += len(word)
    return round(total / len(words), 2)


def longest_word(text):
    """The longest word; the first one wins a tie. Empty string if no words."""
    best = ""
    for word in tokenize(text):
        if len(word) > len(best):
            best = word
    return best
