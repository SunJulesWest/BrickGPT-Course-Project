"""Run with python -S -m unittest discover -s tests -p test_run_experiments.py -v."""

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Literal
import unittest
from unittest.mock import patch

import run_experiments as runner


def events(summary):
    return [json.loads(line) for line in Path(summary['records']).read_text(encoding='utf-8').splitlines()]


def terminal(summary):
    return [row for row in events(summary) if row['status'] != 'running']


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = [runner.Prompt('chair', 'A chair.'), runner.Prompt('table', '一张桌子。')]

    def run_batch(self, **kwargs):
        options = dict(prompts=self.prompts, temperatures=[0.4, 0.6], seeds=[1, 42],
                       output_dir=self.root, backend_factory=runner.MockBackend, backend_name='mock')
        options.update(kwargs)
        return runner.run_experiments(**options)

    def test_cartesian_sweep_and_artifacts_can_be_joined_by_run_id(self):
        summary = self.run_batch()
        rows = terminal(summary)
        self.assertEqual((summary['planned'], summary['succeeded'], summary['pending']), (8, 8, 0))
        self.assertEqual(len(events(summary)), 16)
        self.assertEqual(len({r['run_id'] for r in rows}), 8)
        self.assertEqual({(r['prompt_id'], r['initial_temperature'], r['seed']) for r in rows},
                         {(p.id, t, s) for p in self.prompts for t in [0.4, 0.6] for s in [1, 42]})
        for row in rows:
            self.assertEqual(row['backend'], 'mock')
            self.assertIsNone(row['error'])
            self.assertGreaterEqual(row['total_seconds'], row['generation_seconds'])
            self.assertIsNotNone(row['finished_at'])
            for key in ('txt', 'ldr', 'bricks_json'):
                relative = row['artifacts'][key]
                self.assertFalse(Path(relative).is_absolute())
                self.assertNotIn('\\', relative)
                self.assertTrue((Path(summary['batch_dir']) / relative).is_file())
            self.assertIsNone(row['artifacts']['png'])
        self.assertEqual(json.loads((Path(summary['batch_dir']) / 'manifest.json').read_text(encoding='utf-8'))['jobs'][0]['prompt_id'], 'chair')

    def test_failure_is_retained_and_later_jobs_continue(self):
        summary = self.run_batch(backend_factory=lambda cfg: runner.MockBackend(cfg, ['chair']))
        rows = terminal(summary)
        self.assertEqual((summary['failed'], summary['succeeded']), (4, 4))
        failed = rows[0]
        self.assertEqual(failed['stage'], 'generation')
        self.assertEqual(failed['error']['type'], 'RuntimeError')
        self.assertIn('Injected mock failure', failed['error']['traceback'])
        self.assertIsNotNone(failed['generation_seconds'])
        self.assertEqual(rows[-1]['status'], 'succeeded')

    def test_initializer_fails_once_but_all_jobs_are_recorded(self):
        calls = []
        def unavailable(config):
            calls.append(config)
            raise ImportError('model dependencies unavailable')
        summary = self.run_batch(backend_factory=unavailable)
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary['failed'], 8)
        for row in terminal(summary):
            self.assertEqual(row['stage'], 'initialization')
            self.assertEqual(row['error']['type'], 'ImportError')
            self.assertIsNone(row['generation_seconds'])
            self.assertIsNone(row['effective_config'])

    def test_export_failure_keeps_txt_and_continues(self):
        class BrokenBricks(runner.MockBricks):
            def to_ldr(self):
                raise ValueError('unsupported part')
        class Backend(runner.MockBackend):
            def generate(self, prompt, temperature, seed):
                result = super().generate(prompt, temperature, seed)
                if prompt.id == 'chair':
                    result['bricks'] = BrokenBricks(seed)
                return result
        summary = self.run_batch(backend_factory=Backend, temperatures=[0.6], seeds=[1])
        failed, succeeded = terminal(summary)
        self.assertEqual(failed['stage'], 'export')
        self.assertIsNotNone(failed['artifacts']['txt'])
        self.assertIsNone(failed['artifacts']['ldr'])
        self.assertTrue((Path(summary['batch_dir']) / failed['artifacts']['txt']).exists())
        self.assertEqual(succeeded['status'], 'succeeded')

    def test_render_failure_keeps_generated_structure(self):
        def broken_renderer(*args):
            raise RuntimeError('Blender unavailable')
        summary = self.run_batch(prompts=self.prompts[:1], temperatures=[0.6], seeds=[1],
                                 render=True, renderer=broken_renderer)
        row = terminal(summary)[0]
        self.assertEqual((row['status'], row['stage']), ('failed', 'render'))
        self.assertIsNotNone(row['artifacts']['bricks_json'])
        self.assertIsNone(row['artifacts']['png'])

    def test_renderer_must_produce_an_image(self):
        summary = self.run_batch(prompts=self.prompts[:1], temperatures=[0.6], seeds=[1],
                                 render=True, renderer=lambda *args: None)
        self.assertIn('without creating', terminal(summary)[0]['error']['message'])

    def test_interrupt_is_persisted_and_unattempted_jobs_remain_in_manifest(self):
        class Interrupted(runner.MockBackend):
            def generate(self, *args):
                raise KeyboardInterrupt()
        summary = self.run_batch(backend_factory=Interrupted)
        self.assertEqual((summary['interrupted'], summary['pending']), (1, 7))
        self.assertEqual(terminal(summary)[0]['error']['type'], 'KeyboardInterrupt')

    def test_started_record_is_flushed_before_generation(self):
        root = self.root
        class Observed(runner.MockBackend):
            def generate(self, *args):
                journal = next(root.glob('*/records.jsonl'))
                row = json.loads(journal.read_text(encoding='utf-8').splitlines()[-1])
                if row['status'] != 'running':
                    raise AssertionError('Missing started record')
                return super().generate(*args)
        summary = self.run_batch(backend_factory=Observed)
        self.assertEqual(summary['succeeded'], 8)

    def test_repeated_batches_never_overwrite_prior_outputs(self):
        first = self.run_batch()
        original = Path(first['records']).read_bytes()
        second = self.run_batch()
        self.assertNotEqual(first['batch_dir'], second['batch_dir'])
        self.assertEqual(Path(first['records']).read_bytes(), original)

    def test_sampling_policy_does_not_claim_an_observed_temperature_trace(self):
        summary = self.run_batch(config={'temperature_increase': 0.05, 'max_temperature': 1.5})
        row = terminal(summary)[0]
        policy = row['sampling_strategy']
        self.assertEqual(policy['initial_temperature'], 0.4)
        self.assertEqual(policy['temperature_mode'], 'adaptive_on_already_rejected')
        self.assertEqual(policy['temperature_increase'], 0.05)
        self.assertEqual(policy['max_temperature'], 1.5)
        self.assertIsNone(policy['top_k_effective'])
        self.assertIsNone(policy['actual_temperature_trace'])
        self.assertEqual(policy['trace_status'], 'not_instrumented')
        config = row['effective_config'] | {'max_brick_rejections': 0, 'use_logit_masking': False}
        policy = runner.sampling_strategy(config)
        self.assertEqual(policy['temperature_mode'], 'fixed')
        self.assertEqual(policy['top_k_effective'], 20)

    def test_bad_sweeps_are_rejected_before_creating_outputs(self):
        invalid = [dict(temperatures=[0]), dict(temperatures=[float('nan')]),
                   dict(temperatures=[float('inf')]), dict(temperatures=[True]),
                   dict(seeds=[-1]), dict(seeds=[2**32]), dict(seeds=[True]),
                   dict(seeds=[1, 1]), dict(temperatures=[0.6, 0.6]),
                   dict(config={'temperature': 0.5}), dict(config={'x': float('nan')}),
                   dict(prompts=[runner.Prompt('same', 'a'), runner.Prompt('same', 'b')])]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_batch(**kwargs)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_prompt_loading_handles_unicode_bom_and_bad_input(self):
        path = self.root / 'prompts.jsonl'
        path.write_text('\ufeff{"id":"桌子","text":"一张桌子。"}\n\n', encoding='utf-8')
        self.assertEqual(runner.load_prompts(path), [runner.Prompt('桌子', '一张桌子。')])
        for text in ('{}', '[]', '{', '{"id":"a","text":" "}',
                     '{"id":"a","text":"x"}\n{"id":"a","text":"y"}'):
            path.write_text(text, encoding='utf-8')
            with self.subTest(text=text), self.assertRaises(ValueError):
                runner.load_prompts(path)

    def test_official_adapter_reuses_model_sets_seed_and_resolves_upstream_defaults(self):
        # Compile only the upstream config class; never execute its torch imports.
        source = ast.parse((runner.ROOT / 'src/brickgpt/models/brickgpt.py').read_text(encoding='utf-8'))
        config_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'BrickGPTConfig')
        scope = {'dataclass': dataclass, 'field': field, 'Literal': Literal}
        exec(compile(ast.Module(body=[config_class], type_ignores=[]), '<upstream-config>', 'exec'), scope)
        configs, calls, seed_calls = [], [], []
        class FakeModel:
            def __init__(self, config):
                configs.append(config)
                self.device = 'cpu'
                self.llm = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash='mock-revision')))
            def __call__(self, text):
                calls.append((text, self.temperature))
                return runner.MockBackend({}).generate(runner.Prompt('id', text), self.temperature, 1)
        modules = {
            'brickgpt.models': types.SimpleNamespace(BrickGPT=FakeModel, BrickGPTConfig=scope['BrickGPTConfig']),
            'transformers': types.SimpleNamespace(set_seed=seed_calls.append),
            'torch': types.SimpleNamespace(),
        }
        with patch.dict(sys.modules, modules):
            summary = self.run_batch(backend_factory=runner.OfficialBackend, backend_name='official')
        self.assertEqual(summary['succeeded'], 8)
        self.assertEqual(len(configs), 1)
        self.assertEqual(seed_calls, [1, 42] * 4)
        self.assertEqual([temp for _, temp in calls], [0.4, 0.4, 0.6, 0.6] * 2)
        self.assertEqual(terminal(summary)[0]['model_revision'], 'mock-revision')
        config = terminal(summary)[0]['effective_config']
        self.assertEqual(config['model_name_or_path'], 'AvaLovelace/BrickGPT')
        for name in runner.MockBackend({}).config:
            self.assertEqual(config[name], runner.MockBackend({}).config[name])

    def test_cli_mock_is_dependency_free_and_reports_failure_exit_code(self):
        # -S excludes site-packages, so accidental heavy imports fail this test.
        command = [sys.executable, '-S', str(runner.ROOT / 'run_experiments.py'),
                   '--prompts', str(runner.ROOT / 'examples/experiment_prompts.jsonl'),
                   '--temperatures', '0.4', '0.6', '--seeds', '1', '42',
                   '--output-dir', str(self.root), '--mock']
        for extra, expected in (([], 0), (['--mock-fail-prompt-id', 'chair-01'], 1)):
            result = subprocess.run(command + extra, cwd=self.root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, expected, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(len(terminal(summary)), 8)
            if expected:
                self.assertEqual(summary['failed'], 4)


if __name__ == '__main__':
    unittest.main()
