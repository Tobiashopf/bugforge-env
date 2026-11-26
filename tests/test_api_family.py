"""Integration and regression checks for the synthetic versioned API."""
from itertools import combinations

import pytest

from bugforge import base, verifier
from bugforge.agents import OracleAgent, run_episode
from bugforge.env import BugForgeEnv
from bugforge.generator import generate
from bugforge.repository import API_DEFECTS
from bugforge.runner import DockerRunner

FAMILY = 'repository-api'
BASE = 'repository_api'


def test_api_shape_and_repeatability():
    a = generate(17, FAMILY, check=False)
    b = generate(17, FAMILY, check=False)
    assert a.workspace == b.workspace and a.bugs == b.bugs
    assert len(a.module_files) == 3 and len(a.bugs) == 2
    assert len({bug.file for bug in a.bugs}) == 2
    assert len(a.visible_tests[BASE]) == 5
    assert {bug.site for bug in a.bugs} <= set(a.visible_tests[BASE])
    assert not any(f.startswith(('fuzz_', 'ref_')) for f in a.workspace)
    assert BASE not in base.available()


@pytest.mark.parametrize('defect', API_DEFECTS)
def test_api_single_defect(defect):
    b = base.load(BASE)
    filename, old, new, witness = defect
    candidate = b.sources
    assert candidate[filename].count(old) == 1
    candidate[filename] = candidate[filename].replace(old, new, 1)
    result = verifier.verify(candidate, [b], isolated=True)
    assert result.valid and not result.solved, result
    assert any(name.endswith('::' + witness) for name in result.failed), result


PAIRS = [(a, b) for a, b in combinations(API_DEFECTS, 2) if a[0] != b[0]]


@pytest.mark.parametrize('pair', PAIRS)
def test_api_pairs(pair):
    b = base.load(BASE)
    candidate = b.sources
    for filename, old, new, _ in pair:
        candidate[filename] = candidate[filename].replace(old, new, 1)
    result = verifier.verify(candidate, [b], isolated=True)
    assert result.valid and not result.solved, result
    assert all(any(name.endswith('::' + defect[3]) for name in result.failed)
               for defect in pair), result


def test_api_reference_and_imports():
    b = base.load(BASE)
    source = verifier._reference_source(b.source, frozenset(f[:-3] for f in b.sources))
    assert 'from ref_api_schema import validate_v1 as check_v1' in source
    assert 'from ref_api_migration import upgrade as to_v2' in source
    result = verifier.verify(b.sources, [b], isolated=True, fuzz_seed=23)
    assert result.solved and result.total == 13, result


def test_api_oracle_and_noop():
    task = generate(0, FAMILY)
    env = BugForgeEnv(FAMILY, sandbox='local')
    try:
        assert run_episode(env, OracleAgent(env), 0, task=task)['reward'] == 1.0
        env.reset(0, task=task)
        _, reward, *_ = env.step({'tool': 'submit', 'args': {}})
        assert reward == 0.0
    finally:
        env.close()


@pytest.mark.skipif(not DockerRunner.available(), reason='Docker image unavailable')
def test_docker_api_reference_and_bugged():
    task = generate(0, FAMILY, check=False)
    b = task.load_bases()
    assert verifier.verify(task.solution, b, runner=DockerRunner()).solved
    result = verifier.verify({f: task.workspace[f] for f in task.module_files}, b,
                             runner=DockerRunner())
    assert result.valid and not result.solved
