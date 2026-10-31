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
