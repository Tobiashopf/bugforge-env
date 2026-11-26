"""Validate a finite job specification and deduplicate identical definitions."""


def collect(jobs, max_attempts):
    if type(jobs) is not list or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('invalid scheduler input')
    unique = {}
    for job in jobs:
        if type(job) is not dict or set(job) != {'id', 'priority', 'deps', 'failures'}:
            raise ValueError('invalid job fields')
        key = job['id']
        if type(key) is not str or not key.strip():
            raise ValueError('invalid id')
        if type(job['priority']) is not int:
            raise ValueError('invalid priority')
        if type(job['failures']) is not int or job['failures'] < 0:
            raise ValueError('invalid failure count')
        deps = job['deps']
        if type(deps) is not list or any(type(dep) is not str or not dep.strip() for dep in deps):
            raise ValueError('invalid dependencies')
        if len(set(deps)) != len(deps):
            raise ValueError('duplicate dependency')
        if key in unique:
            if unique[key] != job:
                raise ValueError('conflicting job id')
            continue
        unique[key] = job
    for job in unique.values():
        if any(dep not in unique for dep in job['deps']):
            raise ValueError('unknown dependency')
    return unique
