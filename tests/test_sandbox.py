import pytest

from bugforge.agents import OracleAgent, exploits, run_episode
from bugforge.env import BugForgeEnv
from bugforge.runner import DockerRunner, get_runner
from bugforge.sandbox import ToolError, Workspace

FILES = {
    "calc.py": "def add(a, b):\n    return a - b\n",
    "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n",
}


def test_guarded_local_mode_refuses_unsafe_module_code():
    with Workspace(FILES) as ws:
        ws.write_file("calc.py", "import os\n\ndef add(a, b):\n    return a + b\n")
        run = ws.run_tests()
    assert not run.all_passed and "tests not run" in run.output


def test_guarded_local_mode_ignores_edited_tests():
    with Workspace(FILES) as ws:
        ws.write_file("test_calc.py", "def test_ok():\n    assert True\n")
        run = ws.run_tests()
    assert run.failed == 1 and "ignored" in run.output


def test_guarded_local_mode_runs_a_real_fix():
    with Workspace(FILES) as ws:
        ws.replace("calc.py", "a - b", "a + b")
        assert ws.run_tests().all_passed


def test_unknown_sandbox_raises():
    with pytest.raises(ValueError):
        get_runner("vm")


def test_docker_command_is_locked_down(tmp_path):
    cmd = DockerRunner(image="img").command(tmp_path, ["-q"], {"BUGFORGE_FUZZ_SEED": "3",
                                                               "SECRET": "x"}, "n1")
    joined = " ".join(cmd)
    for flag in ["--network none", "--read-only", "--cap-drop ALL", "--pids-limit",
                 "--memory", "no-new-privileges", ":/work"]:
        assert flag in joined
    assert "BUGFORGE_FUZZ_SEED=3" in joined and "SECRET" not in joined
    assert cmd[-4:] == ["python", "-m", "pytest", "-q"]


docker = pytest.mark.skipif(not DockerRunner.available(),
                            reason="Docker or the bugforge-sandbox image is not available")


@docker
def test_docker_oracle_solves():
    env = BugForgeEnv("medium", sandbox="docker")
    assert run_episode(env, OracleAgent(env), 1)["reward"] == 1.0


@docker
@pytest.mark.parametrize("exploit", exploits.ALL, ids=lambda e: e.name)
def test_docker_exploits_score_zero(exploit):
    env = BugForgeEnv("easy", sandbox="docker")
    assert run_episode(env, exploit(), 2)["reward"] == 0.0


@docker
def test_docker_blocks_network_for_agent_code():
    env = BugForgeEnv("easy", sandbox="docker")
    env.reset(0)
    env.step({"tool": "write_file", "args": {"path": "test_net.py", "content": (
        "import socket\n\ndef test_net():\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=3)\n")}})
    obs, *_ = env.step({"tool": "run_tests", "args": {}})
    env.close()
    assert "test_net" in obs and "failed" in obs


def test_host_tools_reject_workspace_links(tmp_path):
    outside = tmp_path / 'outside.txt'
    outside.write_text('untouched', encoding='utf-8')
    with Workspace(FILES) as ws:
        link = ws.root / 'link.py'
        try:
            link.symlink_to(outside)
            (ws.root / 'linked_dir').symlink_to(tmp_path, target_is_directory=True)
        except OSError:
            pytest.skip('symlink creation unavailable')
        for name in ('link.py', 'linked_dir/outside.txt'):
            with pytest.raises(ToolError):
                ws.read_file(name)
            with pytest.raises(ToolError):
                ws.write_file(name, 'changed')
        assert ws.snapshot(['link.py']) == {}
        assert 'link.py' not in ws.list_files()
        assert outside.read_text(encoding='utf-8') == 'untouched'


def test_host_tools_reject_large_candidate_file():
    with Workspace(FILES) as ws:
        (ws.root / 'calc.py').write_text('x' * 200001, encoding='utf-8')
        with pytest.raises(ToolError, match='too large'):
            ws.read_file('calc.py')
        assert ws.snapshot(['calc.py']) == {}
