import os
import random
import math
import pytest
from repository_numerical import run
from ref_repository_numerical import run as reference


def test_seeded_numerical_matches_reference():
    rng = random.Random(int(os.environ.get('BUGFORGE_FUZZ_SEED', '0')))
    for i in range(64):
        training = [rng.randint(-1000, 1000) / 8 for _ in range(rng.randint(2, 12))]
        if i % 4 == 0:
            training = [training[0]] * len(training)
        training.append(None)
        inference = [rng.randint(-4000, 4000) / 8, None, rng.randint(-4000, 4000) / 8]
        result = run(training, inference)
        expected = reference(training, inference)
        assert result['state'] == pytest.approx(expected['state'], rel=1e-10, abs=1e-10)
        for key in ('training', 'inference'):
            assert result[key] == pytest.approx(expected[key], rel=1e-10, abs=1e-10)
        shuffled = list(training)
        rng.shuffle(shuffled)
        permuted = run(shuffled, inference)
        assert permuted['state'] == pytest.approx(result['state'], rel=1e-10, abs=1e-10)
        assert permuted['inference'] == pytest.approx(result['inference'], rel=1e-10, abs=1e-10)
        assert result['inference'][1] == pytest.approx(0, abs=1e-10)
        assert math.fsum(result['training']) == pytest.approx(0, abs=1e-9)
