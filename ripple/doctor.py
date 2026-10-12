import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

from .common import ROOT, dump_json, timestamp, read_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    args = ap.parse_args()
    report = {'checked_at_utc': timestamp(), 'python': sys.version, 'platform': platform.platform(), 'packages': {}}
    for name in ['numpy','pandas','openpyxl','scikit-learn','catboost']:
        report['packages'][name] = importlib.metadata.version(name)
    if args.gpu:
        from catboost import CatBoostRegressor
        from catboost.utils import get_gpu_device_count
        report['gpu_count'] = get_gpu_device_count()
        if report['gpu_count'] < 1:
            raise RuntimeError('No usable NVIDIA GPU. Install/update the driver and rerun. No automatic CPU fallback.')
        if shutil.which('nvidia-smi'):
            proc = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'], capture_output=True, text=True, timeout=15)
            report['nvidia_smi'] = proc.stdout.strip()
        # Synthetic engineering check only. Never used as research data.
        import numpy as np
        rng = np.random.default_rng(42)
        x = rng.normal(size=(128, 3))
        CatBoostRegressor(iterations=2, depth=2, task_type='GPU', devices=read_config()['gpu_devices'],
                         verbose=False, allow_writing_files=False).fit(x, x[:, 0] + x[:, 1])
        report['gpu_fit_check'] = 'passed_on_synthetic_data'
    dump_json(ROOT / 'reports' / 'environment_check.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
