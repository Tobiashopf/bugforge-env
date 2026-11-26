"""Simulate serial job scheduling without wall-clock or external side effects."""
from scheduler_rules import collect
from scheduler_state import choose, attempt


def schedule(jobs, max_attempts):
    specs = collect(jobs, max_attempts)
    states = {key: 'pending' for key in specs}
    attempts = {key: 0 for key in specs}
    events = []
    while True:
        ready = [key for key in specs if states[key] == 'pending'
                 and all(states[dep] == 'succeeded' for dep in specs[key]['deps'])]
        if not ready:
            break
        key = choose(ready, specs)
        attempts[key] += 1
        state = attempt(specs[key], attempts[key], max_attempts)
        states[key] = state
        events.append([key, attempts[key], state])
    for key in states:
        if states[key] == 'pending':
            states[key] = 'blocked'
    return {'states': states, 'attempts': attempts, 'events': events}
