"""Regression tests for the opt-in multi-module checkout family."""
from itertools import combinations

import pytest

from bugforge import base, policy, verifier
from bugforge.agents import OracleAgent, run_episode
from bugforge.env import BugForgeEnv
from bugforge.generator import generate
from bugforge.repository import DEFECTS
from bugforge.runner import DockerRunner

FAMILY = 'repository-checkout'


def test_legacy_pool_is_unchanged():
    assert base.available() == ['grades', 'intervals', 'inventory', 'pagination', 'textstats']


def test_shape_and_repeatability():
    a, b = generate(11, FAMILY, check=False), generate(11, FAMILY, check=False)
    assert a.workspace == b.workspace and a.bugs == b.bugs
    assert len(a.module_files) == 3 and len(a.bugs) == 2
    assert len({bug.file for bug in a.bugs}) == 2
    assert len(a.visible_tests['repository_checkout']) == 4
    assert not any(f.startswith(('fuzz_', 'ref_')) for f in a.workspace)


@pytest.mark.parametrize('defect', DEFECTS)
def test_single_defect_detected_by_public_api(defect):
    b = base.load('repository_checkout')
    filename, old, new, witness = defect
    candidate = b.sources
    assert candidate[filename].count(old) == 1
    candidate[filename] = candidate[filename].replace(old, new)
    result = verifier.verify(candidate, [b], isolated=True)
    assert result.valid and not result.solved
    assert any(name.endswith('::' + witness) for name in result.failed), result


def test_all_distinct_file_pairs():
    b = base.load('repository_checkout')
    pairs = [(a, c) for a, c in combinations(DEFECTS, 2) if a[0] != c[0]]
    assert len(pairs) == 5
    for pair in pairs:
        candidate = b.sources
        for filename, old, new, _ in pair:
            candidate[filename] = candidate[filename].replace(old, new, 1)
        result = verifier.verify(candidate, [b], isolated=True)
        assert result.valid and not result.solved, (pair, result)


def test_reference_imports_do_not_bind_candidate_dependencies():
    b = base.load('repository_checkout')
    source = verifier._reference_source(b.source, frozenset(f[:-3] for f in b.sources))
    assert 'import ref_repo_pricing as repo_pricing' in source
    assert 'from ref_repo_stock import reserve' in source
    candidate = b.sources
    candidate['repo_pricing.py'] = candidate['repo_pricing.py'].replace(
        'unit_cents * quantity', '0')
    for isolated in (False, True):
        result = verifier.verify(candidate, [b], isolated=isolated)
        assert result.valid and any('test_seeded_checkout_matches_reference' in f
                                    for f in result.failed), result


def test_project_imports_and_missing_dependency():
    b = base.load('repository_checkout')
    allowed = frozenset(f[:-3] for f in b.sources)
    assert policy.check(b.source, project_modules=allowed) == []
    assert policy.check('import ref_repo_stock', project_modules=allowed)
    assert policy.check('import os', project_modules=allowed)
    candidate = b.sources
    del candidate['repo_stock.py']
    result = verifier.verify(candidate, [b], isolated=True)
    assert not result.solved and result.pass_fraction == 0


def test_oracle_and_noop_episode():
    task = generate(0, FAMILY)
    env = BugForgeEnv(FAMILY, sandbox='local')
    try:
        assert run_episode(env, OracleAgent(env), 0, task=task)['reward'] == 1.0
        env.reset(0, task=task)
        _, reward, *_ = env.step({'tool': 'submit', 'args': {}})
        assert reward == 0.0
    finally:
        env.close()


def test_split_reference():
    b = base.load('repository_checkout')
    result = verifier.verify(b.sources, [b], isolated=True)
    assert result.solved and result.total == 11, result


@pytest.mark.skipif(not DockerRunner.available(), reason='Docker image unavailable')
def test_docker_repository_reference_and_bugged():
    task = generate(0, FAMILY, check=False)
    b = task.load_bases()
    assert verifier.verify(task.solution, b, runner=DockerRunner()).solved
    result = verifier.verify({f: task.workspace[f] for f in task.module_files}, b,
                             runner=DockerRunner())
    assert result.valid and not result.solved
