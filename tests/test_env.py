import pytest

from bugforge.agents import OracleAgent, RandomAgent, exploits, run_episode
from bugforge.env import BugForgeEnv
from bugforge.generator import DIFFICULTIES


@pytest.fixture
def env():
    e = BugForgeEnv("easy", max_steps=5)
    yield e
    e.close()


def test_reset_lists_files(env):
    obs, info = env.reset(0)
    for name in info["bases"]:
        assert f"{name}.py" in obs


def test_path_traversal_is_refused(env):
    env.reset(0)
    for path in ["../evil.py", "/etc/passwd", "C:\\x.py", "a/../../b.py"]:
        obs, *_ = env.step({"tool": "write_file", "args": {"path": path, "content": "x"}})
        assert obs.startswith("error")


def test_unknown_tool_and_bad_args_are_errors_not_crashes(env):
    env.reset(0)
    assert env.step({"tool": "rm_rf", "args": {}})[0].startswith("error")
    assert env.step({"tool": "read_file", "args": {}})[0].startswith("error")


def test_run_tests_reports_failures_on_the_bugged_task(env):
    env.reset(0)
    obs, *_ = env.step({"tool": "run_tests", "args": {}})
    assert "failed" in obs


def test_step_budget_ends_and_grades(env):
    env.reset(0)
    out = None
    for _ in range(5):
        out = env.step({"tool": "list_files", "args": {}})
    obs, reward, terminated, truncated, _ = out
    assert truncated and not terminated and reward == 0.0 and "Episode over" in obs


def test_step_after_end_raises(env):
    env.reset(0)
    env.step({"tool": "submit", "args": {}})
    with pytest.raises(RuntimeError):
        env.step({"tool": "list_files", "args": {}})


def test_tool_schemas_cover_the_tools():
    names = {t["name"] for t in BugForgeEnv.tool_schemas()}
    assert names == {"list_files", "read_file", "write_file", "replace_in_file", "run_tests",
                     "submit"}


@pytest.mark.parametrize("difficulty", list(DIFFICULTIES))
@pytest.mark.parametrize("seed", [0, 1])
def test_oracle_always_solves(difficulty, seed):
    env = BugForgeEnv(difficulty)
    assert run_episode(env, OracleAgent(env), seed)["reward"] == 1.0


def test_random_agent_scores_zero():
    env = BugForgeEnv("easy")
    assert all(run_episode(env, RandomAgent(s), s)["reward"] == 0.0 for s in range(3))


@pytest.mark.parametrize("difficulty", list(DIFFICULTIES))
@pytest.mark.parametrize("exploit", exploits.ALL, ids=lambda e: e.name)
def test_exploits_score_zero(exploit, difficulty):
    env = BugForgeEnv(difficulty)
    result = run_episode(env, exploit(), seed=4)
    assert result["reward"] == 0.0, result


def test_partial_credit_rewards_progress():
    env = BugForgeEnv("medium", partial_credit=True)
    env.reset(1)
    task = env.task
    bug_file = task.bugs[0].file
    env.step({"tool": "write_file", "args": {"path": bug_file, "content": task.solution[bug_file]}})
    _, reward, *_ = env.step({"tool": "submit", "args": {}})
    assert 0.0 < reward < 1.0
