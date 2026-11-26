"""Integration and regression checks for the synthetic numerical pipeline."""
from itertools import combinations

import pytest

from bugforge import base, verifier
from bugforge.agents import OracleAgent, run_episode
from bugforge.env import BugForgeEnv
from bugforge.generator import generate
from bugforge.repository import NUMERICAL_DEFECTS
from bugforge.runner import DockerRunner

FAMILY = 'repository-numerical'
BASE = 'repository_numerical'


def test_numerical_shape_and_repeatability():
    a = generate(17, FAMILY, check=False)
    b = generate(17, FAMILY, check=False)
    assert a.workspace == b.workspace and a.bugs == b.bugs
    assert len(a.module_files) == 3 and len(a.bugs) == 2
    assert len({bug.file for bug in a.bugs}) == 2
    assert len(a.visible_tests[BASE]) == 5
    assert {bug.site for bug in a.bugs} <= set(a.visible_tests[BASE])
    assert not any(f.startswith(('fuzz_', 'ref_')) for f in a.workspace)
    assert BASE not in base.available()


@pytest.mark.parametrize('defect', NUMERICAL_DEFECTS)
def test_numerical_single_defect(defect):
    b = base.load(BASE)
    filename, old, new, witness = defect
    candidate = b.sources
    assert candidate[filename].count(old) == 1
    candidate[filename] = candidate[filename].replace(old, new, 1)
    result = verifier.verify(candidate, [b], isolated=True)
    assert result.valid and not result.solved, result
    assert any(name.endswith('::' + witness) for name in result.failed), result


PAIRS = [(a, b) for a, b in combinations(NUMERICAL_DEFECTS, 2) if a[0] != b[0]]


@pytest.mark.parametrize('pair', PAIRS)
def test_numerical_pairs(pair):
    b = base.load(BASE)
    candidate = b.sources
    for filename, old, new, _ in pair:
        candidate[filename] = candidate[filename].replace(old, new, 1)
    result = verifier.verify(candidate, [b], isolated=True)
    assert result.valid and not result.solved, result
    assert all(any(name.endswith('::' + defect[3]) for name in result.failed)
               for defect in pair), result


def test_numerical_reference_and_imports():
    b = base.load(BASE)
    source = verifier._reference_source(b.source, frozenset(f[:-3] for f in b.sources))
    assert 'from ref_numerical_stats import fit as fit_stats' in source
    assert 'from ref_numerical_transform import transform as apply_transform' in source
    result = verifier.verify(b.sources, [b], isolated=True, fuzz_seed=23)
    assert result.solved and result.total == 13, result


def test_numerical_oracle_and_noop():
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
def test_docker_numerical_reference_and_bugged():
    task = generate(0, FAMILY, check=False)
    b = task.load_bases()
    assert verifier.verify(task.solution, b, runner=DockerRunner()).solved
    result = verifier.verify({f: task.workspace[f] for f in task.module_files}, b,
                             runner=DockerRunner())
    assert result.valid and not result.solved
