"""Failure boundaries and result interpretation for the home-PC runner."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ripple import doctor, workflow
from ripple.common import dump_json


class WorkflowTests(unittest.TestCase):
    def test_gpu_failure_is_recorded_and_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def fail_gpu(*args):
                if 'ripple.doctor' in args:
                    self.assertIn('--gpu', args)
                    raise subprocess.CalledProcessError(1, args)
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'run_step', side_effect=fail_gpu) as step:
                with patch('ripple.train.train_models') as train:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(subprocess.CalledProcessError):
                            workflow.execute(device='GPU')
                    train.assert_not_called()
                self.assertEqual(step.call_count, 2)
            status = json.loads(next((root / 'reports/workflows').glob('*.json')).read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['current_step'], 'environment')
            self.assertEqual(status['completed_steps'], ['dependency_check'])

    def test_offline_missing_data_fails_before_training_without_fetch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def missing_raw(*args):
                self.assertNotIn('ripple.fetch', args)
                if 'ripple.prepare' in args:
                    raise subprocess.CalledProcessError(1, args)
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'run_step', side_effect=missing_raw):
                with patch('ripple.train.train_models') as train:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(subprocess.CalledProcessError):
                            workflow.execute(device='CPU', offline=True)
                    train.assert_not_called()
            status = json.loads(next((root / 'reports/workflows').glob('*.json')).read_text())
            self.assertEqual(status['current_step'], 'prepare')
            self.assertEqual(status['status'], 'failed')

    def test_summary_keeps_validation_choice_even_when_test_favors_another_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dump_json(root / 'run.json', {'purpose': 'research_training', 'device': 'CPU',
                                         'selected_by_validation_rmse': 'ridge'})
            rows = []
            for split in ['validation', 'test_temporal', 'test_unseen_plants']:
                for model, score in [('mean_delta', 2.), ('ridge', 1. if split == 'validation' else 3.)]:
                    rows.append(dict(split=split, model=model, rmse_c=score))
            pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
            workflow.write_summary(root)
            report = (root / 'SUMMARY.md').read_text()
            self.assertIn('Selected using validation only: ridge', report)
            self.assertIn('| test_unseen_plants | 3.0000 | 2.0000 | False |', report)

    def test_failed_doctor_replaces_old_success_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dump_json(root / 'reports/environment_check.json', {'status': 'passed'})
            with patch.object(doctor, 'ROOT', root), patch('sys.argv', ['doctor']), \
                    patch.object(doctor.importlib.metadata, 'version', side_effect=RuntimeError('missing package')):
                with self.assertRaisesRegex(RuntimeError, 'missing package'):
                    doctor.main()
            report = json.loads((root / 'reports/environment_check.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error'], 'missing package')


if __name__ == '__main__':
    unittest.main()
