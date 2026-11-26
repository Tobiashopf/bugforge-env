import pytest

from bugforge import base, verifier
from bugforge.rpc_worker import pack, unpack
from bugforge.runner import DockerRunner


@pytest.mark.parametrize("name", base.available())
def test_split_reference_passes(name):
    b = base.load(name)
    result = verifier.verify({b.module_file: b.source}, [b], isolated=True)
    assert result.solved, result


def test_split_wrong_output_fails():
    b = base.load("grades")
    candidate = b.source.replace("return 'A'", "return 'B'")
    assert candidate != b.source
    result = verifier.verify({b.module_file: candidate}, [b], isolated=True)
    assert not result.solved and result.failed


def test_dead_worker_fails_closed(monkeypatch):
    b = base.load("grades")
    monkeypatch.setattr(verifier.policy, "check", lambda _: [])
    result = verifier.verify({b.module_file: 'raise RuntimeError("dead")\n' + b.source},
                             [b], isolated=True)
    assert not result.solved and result.pass_fraction == 0 and not result.report_valid


def test_codec_keeps_types_and_rejects_unknown_values():
    value = {"a": [(1, False), None], 3: 4.0}
    assert unpack(pack(value)) == value
    assert type(unpack(pack((1, 2)))) is tuple
    with pytest.raises(ValueError):
        unpack({"t": "pickle", "v": []})
    with pytest.raises(ValueError):
        pack(object())


def test_worker_mount_is_read_only_and_candidate_only(tmp_path):
    directory = tmp_path / "candidate"
    command = DockerRunner(image="img").worker_command(directory, "grades", "worker")
    assert f"{directory.resolve()}:/work:ro" in command
    assert "--network" in command and "--read-only" in command
    assert "65534:65534" in command and "-i" in command
    assert all("trusted" not in item for item in command)
    assert command[-4:] == ["python", "-u", "_bf_worker.py", "grades"]


@pytest.mark.skipif(not DockerRunner.available(), reason="Docker image unavailable")
def test_docker_split_reference():
    b = base.load("inventory")
    result = verifier.verify({b.module_file: b.source}, [b], runner=DockerRunner())
    assert result.solved, result


def test_trusted_staging_never_mounted(monkeypatch):
    from bugforge.runner import Result

    b = base.load("grades")
    mounted = []

    class InspectRunner(DockerRunner):
        def worker_command(self, directory, module, name):
            mounted.append(directory)
            return super().worker_command(directory, module, name)

        def remove_worker(self, name):
            pass

    def inspect_run(directory, files, timeout, env, runner):
        assert directory != mounted[0]
        assert {p.name for p in mounted[0].iterdir()} == {"grades.py", "_bf_worker.py"}
        assert (directory / "ref_grades.py").read_text() == b.source
        assert (directory / "fuzz_grades.py").read_text() == b.fuzz
        assert (directory / "grades.py").read_text() != b.source
        report = directory / "report.xml"
        return Result(2, "", ""), report

    monkeypatch.setattr(verifier, "run_pytest", inspect_run)
    result = verifier.verify({b.module_file: b.source}, [b], runner=InspectRunner())
    assert not result.solved and result.pass_fraction == 0


@pytest.mark.skipif(not DockerRunner.available(), reason="Docker image unavailable")
def test_docker_worker_cannot_see_private_files(monkeypatch):
    b = base.load("grades")
    # Controlled regression fixture, not a real submission. Check the mount layout
    # with the static policy disabled to validate the independent boundary.
    monkeypatch.setattr(verifier.policy, "check", lambda _: [])
    probe = ('from pathlib import Path\n'
             'assert set(p.name for p in Path("/work").iterdir()) == '
             '{"grades.py", "_bf_worker.py"}\n')
    result = verifier.verify({b.module_file: probe + b.source}, [b], runner=DockerRunner())
    assert result.solved, result
