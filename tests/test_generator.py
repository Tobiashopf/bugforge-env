import pytest

from bugforge import base
from bugforge.generator import DIFFICULTIES, generate
from bugforge.index import load_index


@pytest.mark.parametrize("name", base.available())
def test_kill_matrix_is_fresh(name):
    load_index(name)  # raises if a base task changed without rebuilding the index


def test_same_seed_same_task():
    a, b = generate(5, "medium"), generate(5, "medium")
    assert a.workspace == b.workspace and [x.site for x in a.bugs] == [x.site for x in b.bugs]


def test_different_seeds_differ():
    tasks = {tuple(sorted(generate(s, "easy").workspace.items())) for s in range(6)}
    assert len(tasks) > 1


@pytest.mark.parametrize("difficulty", list(DIFFICULTIES))
def test_task_shape_matches_difficulty(difficulty):
    d = DIFFICULTIES[difficulty]
    task = generate(3, difficulty)
    assert len(task.bases) == d.n_files
    assert len(task.bugs) == d.n_bugs
    assert len({b.file for b in task.bugs}) == min(d.n_bugs, d.n_files)
    for name in task.bases:
        assert f"{name}.py" in task.workspace and f"test_{name}.py" in task.workspace
        assert f"fuzz_{name}.py" not in task.workspace
        assert set(task.visible_tests[name]) <= set(base.load(name).test_names())
    assert "README.md" in task.workspace


def test_bugged_modules_differ_from_solution():
    task = generate(2, "hard")
    bugged = {b.file for b in task.bugs}
    for f in task.module_files:
        assert (task.workspace[f] != task.solution[f]) == (f in bugged)


def test_unknown_difficulty_raises():
    with pytest.raises(ValueError):
        generate(0, "impossible")


def test_operator_knob_restricts_bug_types():
    from bugforge.generator import custom
    diff = custom(3, 2, 0.4, operators=("arith",))
    assert diff.name.endswith("-arith")
    for seed in range(3):
        assert {b.operator for b in generate(seed, diff).bugs} == {"arith"}


def test_unknown_operator_raises():
    from bugforge.generator import custom
    with pytest.raises(ValueError):
        custom(1, 1, 0.6, operators=("teleport",))


def test_generation_validation_uses_episode_fuzz_seed(monkeypatch):
    from types import SimpleNamespace

    from bugforge import generator

    calls = []
    def verify(candidates, bases, *, fuzz_seed):
        calls.append(fuzz_seed)
        return SimpleNamespace(valid=True, solved=len(calls) % 2 == 0)
    monkeypatch.setattr(generator, 'verify', verify)
    monkeypatch.setattr(generator, '_visible_fails', lambda task: True)
    assert generator.generate(37, 'easy').seed == 37
    assert calls == [37, 37]


def test_invalid_baseline_is_not_a_valid_task(monkeypatch):
    from types import SimpleNamespace

    from bugforge import generator

    monkeypatch.setattr(generator, 'verify', lambda *a, **kw:
                        SimpleNamespace(valid=False, solved=False))
    with pytest.raises(RuntimeError, match='no valid task'):
        generator.generate(37, 'easy', max_attempts=2)
