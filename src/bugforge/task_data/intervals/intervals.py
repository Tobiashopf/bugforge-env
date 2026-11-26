"""Half-open integer intervals [start, end): merging, coverage and gaps."""


def _check(intervals):
    for start, end in intervals:
        if start > end:
            raise ValueError("interval start must not exceed end")


def merge(intervals):
    """Merge overlapping or touching intervals; empty intervals are dropped."""
    _check(intervals)
    result = []
    for start, end in sorted(intervals):
        if start == end:
            continue
        if result and start <= result[-1][1]:
            last_start, last_end = result[-1]
            result[-1] = (last_start, max(last_end, end))
        else:
            result.append((start, end))
    return result


def total_covered(intervals):
    """Number of integer points covered by the union of the intervals."""
    total = 0
    for start, end in merge(intervals):
        total += end - start
    return total


def contains(intervals, point):
    """Whether any interval contains the point (start inclusive, end exclusive)."""
    _check(intervals)
    for start, end in intervals:
        if start <= point < end:
            return True
    return False


def gaps(intervals, lo, hi):
    """Uncovered sub-intervals of [lo, hi), in order."""
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    result = []
    cursor = lo
    for start, end in merge(intervals):
        if end <= lo or start >= hi:
            continue
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < hi:
        result.append((cursor, hi))
    return result
