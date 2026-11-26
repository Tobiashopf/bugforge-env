import shutil
from pathlib import Path

from bugforge import base
from bugforge.index import load_index


def test_bundled_task_resources_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("BUGFORGE_TASKS", raising=False)
    monkeypatch.chdir(tmp_path)
    root = base.tasks_root()
    assert root == Path(base.__file__).resolve().parent / "task_data"
    names = base.available()
    assert names
    for name in names:
        assert base.load(name).test_names()
        assert load_index(name)["name"] == name


def test_external_task_directory_override(tmp_path, monkeypatch):
    monkeypatch.delenv("BUGFORGE_TASKS", raising=False)
    name = base.available()[0]
    shutil.copytree(base.tasks_root() / name, tmp_path / name)
    monkeypatch.setenv("BUGFORGE_TASKS", str(tmp_path))
    assert base.tasks_root() == tmp_path
    assert base.available() == [name]
    assert load_index(name)["name"] == name
