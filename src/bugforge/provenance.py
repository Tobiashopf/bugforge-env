"""Experiment identity; fingerprints describe configuration, not model determinism."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from .base import tasks_root
from .runner import DockerRunner


def fingerprint(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def tree_hash(root: Path, suffixes: set[str]) -> str:
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in suffixes
             and '__pycache__' not in p.parts}
    return fingerprint(files)


def config(diff) -> dict:
    c = asdict(diff)
    del c['name']
    # Normalize tuples before saving to JSON and comparing loaded manifests.
    return json.loads(json.dumps(c))


def manifest(client, agent, sandbox: str, max_steps: int, model_revision=None) -> dict:
    # Only explicit, non-secret fields. Never serialize client.__dict__ or os.environ.
    inference = {k: getattr(client, k) for k in ('temperature', 'num_ctx', 'max_tokens', 'timeout')
                 if hasattr(client, k)}
    endpoint = getattr(client, 'host', None) or getattr(client, 'url', None)
    if endpoint:
        u = urlsplit(endpoint)
        # Hash the origin only; no userinfo, query or URL path is recorded.
        inference['origin_hash'] = fingerprint([u.scheme, u.hostname, u.port])
    runner = {'mode': sandbox}
    if sandbox == 'docker':
        docker = DockerRunner()
        result = subprocess.run([docker.docker, 'image', 'inspect', '--format', '{{.Id}}',
                                 docker.image], capture_output=True, text=True, timeout=20,
                                check=True)
        image_id = result.stdout.strip()
        if not image_id.startswith('sha256:') or len(image_id) != 71:
            raise RuntimeError('Docker image ID unavailable')
        runner.update(image_id=image_id, memory=docker.memory, cpus=docker.cpus, pids=docker.pids)
    return {
        'schema': 1,
        'provider': client.provider, 'model': getattr(client, 'model', '?'),
        'client_class': f'{type(client).__module__}.{type(client).__qualname__}',
        'model_revision_claimed': model_revision, 'inference': inference,
        'agent': {'max_steps': max_steps, 'max_nudges': agent.max_nudges,
                  'parse_text_calls': agent.parse_text_calls,
                  'keep_recent_tool_turns': agent.keep_recent_tool_turns,
                  'clip_old_outputs': agent.clip_old_outputs,
                  'system_sha256': fingerprint(agent.system)},
        'runner': runner,
        'runtime': {'python': sys.version, 'platform': platform.platform(),
                    'pytest': version('pytest'), 'python_hash_seed': os.getenv('PYTHONHASHSEED'),
                    'bugforge': '0.1.0'},
        'source_sha256': tree_hash(Path(__file__).parent, {'.py'}),
        'tasks_sha256': tree_hash(tasks_root(), {'.py', '.json'}),
        'generation_fuzz_seed': 'task_seed',
    }
