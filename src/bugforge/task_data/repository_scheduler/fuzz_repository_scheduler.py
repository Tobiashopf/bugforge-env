import os
import random
from repository_scheduler import schedule
from ref_repository_scheduler import schedule as reference


def test_seeded_schedule():
    rng = random.Random(int(os.environ.get('BUGFORGE_FUZZ_SEED', '0')))
    for _ in range(64):
        jobs = []
        for i in range(rng.randrange(1, 9)):
            jobs.append({'id': str(i), 'priority': rng.randrange(-2, 3),
                         'deps': [j['id'] for j in jobs if rng.random() < 0.3],
                         'failures': rng.randrange(4)})
        limit = rng.randrange(1, 4)
        rng.shuffle(jobs)
        result = schedule(jobs, limit)
        assert result == reference(jobs, limit)
        assert schedule(list(reversed(jobs)), limit) == result
        states = {}
        attempts = {j['id']: 0 for j in jobs}
        specs = {j['id']: j for j in jobs}
        for key, count, state in result['events']:
            assert states.get(key, 'pending') == 'pending'
            assert all(states.get(dep) == 'succeeded' for dep in specs[key]['deps'])
            attempts[key] += 1
            assert count == attempts[key] <= limit
            expected = ('succeeded' if count > specs[key]['failures'] else
                        'failed' if count == limit else 'pending')
            assert state == expected
            states[key] = state
        assert result['attempts'] == attempts
        assert result['states'] == {key: states.get(key, 'blocked') for key in specs}
