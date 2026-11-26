"""Pagination: page counts, page slices and the page-number window of a pager."""


def page_count(total_items, per_page):
    """Pages needed for total_items; zero items need zero pages."""
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    if total_items < 0:
        raise ValueError("total_items must be non-negative")
    return (total_items + per_page - 1) // per_page


def page_slice(items, page, per_page):
    """Items on a 1-based page; a page past the end is empty."""
    if page < 1:
        raise ValueError("page numbers start at 1")
    if page > page_count(len(items), per_page):
        return []
    start = (page - 1) * per_page
    return items[start:start + per_page]


def page_window(current, total_pages, width=5):
    """Up to `width` consecutive page numbers around `current`, kept inside [1, total]."""
    if total_pages < 1:
        raise ValueError("total_pages must be at least 1")
    if current < 1 or current > total_pages:
        raise ValueError("current page out of range")
    half = width // 2
    start = max(1, current - half)
    end = min(total_pages, start + width - 1)
    start = max(1, end - width + 1)
    return list(range(start, end + 1))


def has_next(page, total_pages):
    return page < total_pages


def has_previous(page):
    return page > 1
