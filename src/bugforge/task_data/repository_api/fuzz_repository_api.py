import os
import random
from repository_api import convert
from ref_repository_api import convert as reference


def test_seeded_contracts():
    rng = random.Random(int(os.environ.get('BUGFORGE_FUZZ_SEED', '0')))
    for i in range(64):
        record = {'id': f'u{rng.randrange(100000)}', 'name': f'Name {i}'}
        if rng.choice([True, False]):
            record['email'] = rng.choice([None, f'user{i}@example.org'])
        if rng.choice([True, False]):
            record['active'] = rng.choice([True, False])
        canonical = {'id': record['id'], 'name': record['name'],
                     'email': record.get('email'), 'active': record.get('active', True)}
        modern = {'id': record['id'], 'profile': {'name': record['name'], 'email': canonical['email']},
                  'enabled': canonical['active']}
        for source, data in [(1, record), (2, modern)]:
            for target, expected in [(1, canonical), (2, modern)]:
                assert convert(data, source, target) == expected == reference(data, source, target)
        assert convert(convert(record, 1, 2), 2, 1) == canonical
