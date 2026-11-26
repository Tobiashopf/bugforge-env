"""Deterministic priority selection and retry state transitions."""


def choose(ready, jobs):
    return min(ready, key=lambda key: (-jobs[key]['priority'], key))


def attempt(job, count, max_attempts):
    if count > job['failures']:
        return 'succeeded'
    if count >= max_attempts:
        return 'failed'
    return 'pending'
