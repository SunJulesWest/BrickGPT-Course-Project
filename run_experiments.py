"""Batch the official BrickGPT API; mock mode needs only Python's standard library."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Callable
import uuid


SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str


def load_prompts(path: Path) -> list[Prompt]:
    """Read UTF-8 JSONL: each nonblank line has a unique id and a text field."""
    prompts = []
    seen = set()
    with path.open(encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt = Prompt(row['id'], row['text'])
                _validate_prompt(prompt)
                if prompt.id in seen:
                    raise ValueError(f'duplicate prompt id: {prompt.id}')
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
            seen.add(prompt.id)
            prompts.append(prompt)
    if not prompts:
        raise ValueError('At least one prompt is required.')
    return prompts


def _validate_prompt(prompt: Prompt) -> None:
    for field in (prompt.id, prompt.text):
        if not isinstance(field, str) or not field.strip():
            raise ValueError('Prompt id and text must be nonempty strings.')


def sampling_strategy(config: dict) -> dict:
    """Describe the upstream policy, not an unobserved per-step temperature trace."""
    adaptive = config['max_brick_rejections'] > 0 and config['temperature_increase'] > 0
    return {
        'initial_temperature': config['temperature'],
        'temperature_mode': 'adaptive_on_already_rejected' if adaptive else 'fixed',
        'temperature_increase': config['temperature_increase'],
        'max_temperature': config['max_temperature'],
        'temperature_reset': 'initial_temperature_at_each_brick',
        'temperature_update': 'min(max_temperature, temperature + temperature_increase)',
        'top_k_effective': None if config['use_logit_masking'] else config['top_k'],
        'top_p_effective': None if config['use_logit_masking'] else config['top_p'],
        'actual_temperature_trace': None,
        'trace_status': 'not_instrumented',
    }


class OfficialBackend:
    name = 'official'

    def __init__(self, overrides: dict):
        # Keep every heavyweight import behind explicit real execution.
        from brickgpt.models import BrickGPT, BrickGPTConfig
        from transformers import set_seed
        import torch

        self.config = BrickGPTConfig(**overrides)
        self.model = BrickGPT(self.config)
        self.set_seed = set_seed
        self.torch = torch
        self.model_revision = getattr(self.model.llm.model.config, '_commit_hash', None)

    def configuration(self, initial_temperature: float) -> dict:
        return dict(asdict(self.config), temperature=initial_temperature)

    def generate(self, prompt: Prompt, initial_temperature: float, seed: int) -> dict:
        self.model.temperature = initial_temperature
        self.set_seed(seed)
        return self.model(prompt.text)

    def synchronize(self) -> None:
        if str(self.model.device).startswith('cuda'):
            self.torch.cuda.synchronize(self.model.device)
        elif str(self.model.device) == 'mps':
            self.torch.mps.synchronize()


class MockBricks:
    """Synthetic artifacts for exercising I/O; these are not model predictions."""

    def __init__(self, seed: int):
        self.x = random.Random(seed).randrange(5)

    def __len__(self):
        return 1

    def to_txt(self):
        return f'1x1 ({self.x},0,0)\n'

    def to_ldr(self):
        return f'1 115 {self.x * 20 + 10} 0 10 0 0 1 0 1 0 -1 0 0 3005.dat\n0 STEP\n'

    def to_json(self):
        return {'1': {'brick_id': 10, 'x': self.x, 'y': 0, 'z': 0, 'ori': 0}}


class MockBackend:
    name = 'mock'
    model_revision = None

    def __init__(self, overrides: dict, fail_prompt_ids=()):
        # Only the policy fields are simulated. Real mode resolves the full
        # configuration from the official dataclass, without copying its defaults.
        self.config = {
            'temperature_increase': 0.01, 'max_temperature': 2.0,
            'max_brick_rejections': 500, 'max_regenerations': 100,
            'use_logit_masking': True, 'use_gurobi': True,
            'top_k': 20, 'top_p': 1.0,
        } | overrides
        self.fail_prompt_ids = set(fail_prompt_ids)

    def configuration(self, initial_temperature: float) -> dict:
        return dict(self.config, temperature=initial_temperature)

    def generate(self, prompt: Prompt, initial_temperature: float, seed: int) -> dict:
        if prompt.id in self.fail_prompt_ids:
            raise RuntimeError(f'Injected mock failure for prompt {prompt.id}')
        return {'bricks': MockBricks(seed), 'rejection_reasons': Counter(), 'n_regenerations': 0}

    def synchronize(self) -> None:
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_metadata() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ['git', '-C', str(ROOT), *args], text=True, stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    revision = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain')
    return {'commit': revision, 'dirty': None if status is None else bool(status)}


def _error(exc: BaseException) -> dict:
    return {'type': type(exc).__name__, 'message': str(exc),
            'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def _append_record(stream, record: dict) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()
    # Persist the started event before invoking a model or external renderer.
    os.fsync(stream.fileno())


def load_latest_records(path: Path) -> list[dict]:
    """Read one latest event per run_id, including unfinished running attempts.

    Keep legacy rows unchanged: absent generation_status means unknown, not failed.
    Invalid JSON/IDs raise rather than silently dropping samples from statistics.
    """
    latest = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                run_id = record['run_id']
                if not isinstance(run_id, str) or not run_id.strip():
                    raise ValueError('run_id must be a nonempty string.')
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
            latest[run_id] = record
    return list(latest.values())


def _render(ldr_path: Path, png_path: Path) -> None:
    from brickgpt.render_bricks import render_bricks
    render_bricks(str(ldr_path), str(png_path))


def run_experiments(
    prompts: list[Prompt], temperatures: list[float], seeds: list[int], output_dir: Path,
    *, config: dict | None = None, backend_factory: Callable = OfficialBackend,
    backend_name: str = 'official', render: bool = False, renderer: Callable = _render,
) -> dict:
    """Run a Cartesian sweep; append started/terminal events keyed by run_id.

    One model is loaded for the batch. Generation/export failures do not stop later
    jobs. Initialization failure is recorded for every job without repeated loading.
    Ctrl-C records the interrupted attempt and leaves remaining manifest jobs pending.
    """
    config = dict(config or {})
    if 'temperature' in config:
        raise ValueError('Use temperatures, not config.temperature, for the sweep.')
    if not prompts or len({p.id for p in prompts}) != len(prompts):
        raise ValueError('Prompts must be nonempty and have unique ids.')
    for prompt in prompts:
        _validate_prompt(prompt)
    if not temperatures or any(isinstance(t, bool) or not isinstance(t, (int, float))
                               or not math.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError('Initial temperatures must be finite positive numbers.')
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Seeds must be integers in [0, 2**32).')
    if len(set(temperatures)) != len(temperatures) or len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate temperatures or seeds would repeat jobs.')
    # Reject unserializable/non-finite requested parameters before creating outputs.
    json.dumps(config, allow_nan=False)
    batch_id = uuid.uuid4().hex
    batch_dir = Path(output_dir).resolve() / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    jobs = [
        {'run_id': f'{batch_id}-{i:06d}', 'prompt_id': prompt.id, 'prompt': prompt.text,
         'initial_temperature': float(temperature), 'seed': seed}
        for i, (prompt, temperature, seed) in enumerate(
            itertools.product(prompts, temperatures, seeds), 1)
    ]
    provenance = _git_metadata()
    manifest = {'schema_version': SCHEMA_VERSION, 'batch_id': batch_id, 'created_at': _now(),
                'backend': backend_name, 'requested_config': config,
                'render_requested': render, 'source': provenance, 'jobs': jobs}
    (batch_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    backend = None
    initialization_error = None
    counts = Counter()
    generation_counts = Counter()
    with (batch_dir / 'records.jsonl').open('x', encoding='utf-8') as stream:
        for job in jobs:
            artifact_dir = batch_dir / job['run_id']
            record = {
                'schema_version': SCHEMA_VERSION, 'batch_id': batch_id, **job,
                'backend': backend_name, 'source': provenance,
                'requested_parameters': dict(config, temperature=job['initial_temperature']),
                'effective_config': None, 'sampling_strategy': None, 'model_revision': None,
                'status': 'running', 'stage': 'initialization', 'started_at': _now(),
                'generation_status': 'not_started',
                'finished_at': None, 'generation_seconds': None, 'total_seconds': None,
                'render_requested': render,
                'artifacts': {'txt': None, 'ldr': None, 'bricks_json': None, 'png': None},
                'statistics': None, 'error': None,
            }
            _append_record(stream, record)
            started = time.perf_counter()
            try:
                if initialization_error is not None:
                    record['error'] = initialization_error
                    record['status'] = 'failed'
                    continue
                if backend is None:
                    try:
                        backend = backend_factory(config)
                    except Exception as exc:
                        initialization_error = _error(exc)
                        raise
                record['effective_config'] = backend.configuration(job['initial_temperature'])
                record['sampling_strategy'] = sampling_strategy(record['effective_config'])
                record['model_revision'] = backend.model_revision
                record['stage'] = 'generation'
                record['generation_status'] = 'running'
                backend.synchronize()
                generation_start = time.perf_counter()
                try:
                    output = backend.generate(Prompt(job['prompt_id'], job['prompt']),
                                              job['initial_temperature'], job['seed'])
                    backend.synchronize()
                finally:
                    record['generation_seconds'] = time.perf_counter() - generation_start
                bricks = output['bricks']
                record['generation_status'] = 'succeeded'
                record['stage'] = 'export'
                record['statistics'] = {
                    'n_bricks': len(bricks),
                    'rejection_reasons': dict(output['rejection_reasons']),
                    'n_regenerations': output['n_regenerations'],
                }
                artifact_dir.mkdir()
                # Save TXT first: an invalid structure may fail LDraw/JSON export.
                for key, name, serialize in (
                    ('txt', 'structure.txt', bricks.to_txt),
                    ('ldr', 'structure.ldr', bricks.to_ldr),
                    ('bricks_json', 'bricks.json', lambda: json.dumps(bricks.to_json(), ensure_ascii=False)),
                ):
                    path = artifact_dir / name
                    path.write_text(serialize(), encoding='utf-8')
                    record['artifacts'][key] = path.relative_to(batch_dir).as_posix()
                if render:
                    record['stage'] = 'render'
                    png_path = artifact_dir / 'structure.png'
                    renderer(artifact_dir / 'structure.ldr', png_path)
                    if not png_path.is_file():
                        raise RuntimeError('Renderer returned without creating the PNG.')
                    record['artifacts']['png'] = png_path.relative_to(batch_dir).as_posix()
                record['status'] = 'succeeded'
                record['stage'] = 'complete'
            except KeyboardInterrupt as exc:
                record['status'] = 'interrupted'
                if record['generation_status'] == 'running':
                    record['generation_status'] = 'interrupted'
                record['error'] = _error(exc)
            except Exception as exc:
                record['status'] = 'failed'
                if record['generation_status'] == 'running':
                    record['generation_status'] = 'failed'
                record['error'] = _error(exc)
            finally:
                record['finished_at'] = _now()
                record['total_seconds'] = time.perf_counter() - started
                _append_record(stream, record)
                counts[record['status']] += 1
                generation_counts[record['generation_status']] += 1
            if record['status'] == 'interrupted':
                break
    return {'batch_dir': str(batch_dir), 'records': str(batch_dir / 'records.jsonl'),
            'planned': len(jobs), 'pending': len(jobs) - sum(counts.values()),
            'generation_counts': dict(generation_counts), **counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, required=True, help='JSONL with unique id/text pairs')
    parser.add_argument('--temperatures', type=float, nargs='+', required=True, help='Initial temperatures')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/runs'))
    parser.add_argument('--config', type=Path, help='JSON object of official BrickGPTConfig overrides')
    parser.add_argument('--mock', action='store_true', help='Synthetic smoke test; never loads a model')
    parser.add_argument('--mock-fail-prompt-id', action='append', default=[], help='Inject a mock generation failure')
    parser.add_argument('--render', action='store_true', help='Also render PNGs (requires Blender/LDraw)')
    args = parser.parse_args(argv)
    if args.mock_fail_prompt_id and not args.mock:
        parser.error('--mock-fail-prompt-id requires --mock')
    if args.mock and args.render:
        parser.error('--mock does not simulate Blender rendering; omit --render')
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError('--config must contain a JSON object.')
        prompts = load_prompts(args.prompts)
        if set(args.mock_fail_prompt_id) - {p.id for p in prompts}:
            raise ValueError('Unknown --mock-fail-prompt-id.')
        factory = (lambda cfg: MockBackend(cfg, args.mock_fail_prompt_id)) if args.mock else OfficialBackend
        summary = run_experiments(prompts, args.temperatures, args.seeds, args.output_dir,
                                  config=config, backend_factory=factory,
                                  backend_name='mock' if args.mock else 'official', render=args.render)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False))
    return 130 if summary.get('interrupted') else (1 if summary.get('failed') else 0)


if __name__ == '__main__':
    sys.exit(main())
