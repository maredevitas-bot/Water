"""Run the public-data premodel and retain success/failure evidence per attempt."""
import argparse
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys

from .common import ROOT, dump_json, sha256, timestamp


def run_step(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def code_version():
    if not (ROOT / '.git').exists() or not shutil.which('git'):
        return {'commit': None, 'dirty': None, 'note': 'ZIP copy or Git unavailable'}
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, timeout=15).strip()
    try:
        return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain'))}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None, 'note': 'Git inspection unavailable'}


def write_summary(run):
    import pandas as pd
    meta = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    scores = pd.read_csv(run / 'metrics.csv')
    name = meta['selected_by_validation_rmse']
    lines = [
        '# Public-data premodel result', '',
        f"Purpose: {meta['purpose']}", f"Selected using validation only: {name}",
        f"CatBoost device: {meta['device']} (other models: CPU)", '',
        '| Split | Selected RMSE (C) | Mean baseline RMSE (C) | Beats baseline |',
        '|---|---:|---:|---|',
    ]
    for split in ['validation', 'test_temporal', 'test_unseen_plants']:
        part = scores[scores.split == split].set_index('model')
        selected = float(part.loc[name, 'rmse_c'])
        baseline = float(part.loc['mean_delta', 'rmse_c'])
        lines.append(f'| {split} | {selected:.4f} | {baseline:.4f} | {selected < baseline} |')
    lines += ['',
        'Monthly conditional estimates of reported cooling-system discharge temperature.',
        'This does not establish real-time outfall compliance, causal cooling effects, or transfer to Korean plants.',
        'If the held-out plant score does not beat the baseline, new-plant generalization is not established.',
        'The example predictions reuse actual public-data inputs; they are not additional independent validation.',
        'Treatment ratio, auxiliary energy, water consumption and curtailment optimization require separate evidence.',
        'Smoke runs verify software only and must not be reported as research results.', '']
    (run / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')


def execute(device='GPU', offline=False, smoke=False):
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    status_path = ROOT / 'reports' / 'workflows' / f'{attempt}.json'
    status = {'started_at_utc': timestamp(), 'status': 'running', 'device': device,
              'offline': offline, 'smoke': smoke, 'code': code_version(), 'completed_steps': []}
    def step(label, action):
        status['current_step'] = label
        dump_json(status_path, status)
        print(f'\n=== {label} ===', flush=True)
        result = action()
        status['completed_steps'].append(label)
        dump_json(status_path, status)
        return result
    try:
        step('dependency_check', lambda: run_step('-m', 'pip', 'check'))
        step('environment', lambda: run_step('-m', 'ripple.doctor', *(['--gpu'] if device == 'GPU' else [])))
        step('software_tests', lambda: run_step('-m', 'unittest', 'discover', '-s', 'tests', '-v'))
        if not offline:
            step('fetch', lambda: run_step('-m', 'ripple.fetch'))
        # prepare verifies the raw manifest even in offline mode; missing/stale data fails closed.
        step('prepare', lambda: run_step('-m', 'ripple.prepare'))
        from .train import train_models
        run = step('train', lambda: train_models(device, smoke))
        status['run_directory'] = str(run.relative_to(ROOT))
        step('example_prediction', lambda: run_step(
            '-m', 'ripple.predict', '--run', str(run), '--input', str(ROOT / 'example_inputs.csv'),
            '--output', str(run / 'example_predictions.csv'), *(['--allow-smoke'] if smoke else [])))
        step('summary', lambda: write_summary(run))
        # Preserve evidence alongside this exact model run, even after subsequent preparation.
        evidence = run / 'provenance'
        evidence.mkdir()
        for source in [ROOT / 'config.json', ROOT / 'requirements-lock.txt',
                       ROOT / 'data/raw/manifest.json', ROOT / 'reports/quality_report.json',
                       ROOT / 'reports/environment_check.json']:
            shutil.copy2(source, evidence / source.name)
        status['data_sha256'] = sha256(ROOT / 'data/processed/model_data.csv')
        status['status'] = 'completed'
        status['finished_at_utc'] = timestamp()
        dump_json(status_path, status)
        dump_json(evidence / 'workflow.json', status)
        print(f'\nCompleted. Read {run / "SUMMARY.md"}', flush=True)
        return run
    except BaseException as exc:
        status.update(status='failed', finished_at_utc=timestamp(),
                      error_type=type(exc).__name__, error=str(exc))
        dump_json(status_path, status)
        print(f'Workflow stopped at {status.get("current_step")}. Details: {status_path}', file=sys.stderr)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    ap.add_argument('--offline', action='store_true', help='Use previously downloaded, hash-verified raw files')
    ap.add_argument('--smoke', action='store_true', help='Software check only, not research training')
    args = ap.parse_args()
    execute(args.device, args.offline, args.smoke)


if __name__ == '__main__':
    main()
