"""Train on past years, select on validation only, report two held-out tests."""
import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import ROOT, dump_json, read_config, sha256
from .features import CATEGORICAL, NUMERIC, TARGET


def features(frame):
    x = frame[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        x[c] = x[c].fillna('unknown').astype(str)
    return x


def validate_splits(df):
    splits = {k: df[df.split == k].copy() for k in ['train', 'validation', 'test_temporal', 'test_unseen_plants']}
    if any(v.empty for v in splits.values()):
        raise ValueError('Empty train/validation/test partition')
    if df.duplicated(['plant_id', 'cooling_id', 'year', 'month']).any():
        raise ValueError('Duplicate sample keys')
    if not (splits['train'].year.max() < splits['validation'].year.min()
            and splits['validation'].year.max() < splits['test_temporal'].year.min()
            and splits['validation'].year.max() < splits['test_unseen_plants'].year.min()):
        raise ValueError('Temporal split overlaps')
    known = set(splits['train'].plant_id)
    if any(set(splits[k].plant_id) - known for k in ['validation', 'test_temporal']):
        raise ValueError('Unknown plants in known-plant evaluation')
    held = set(splits['test_unseen_plants'].plant_id)
    for k in ['train', 'validation', 'test_temporal']:
        if held & set(splits[k].plant_id):
            raise ValueError('Held-out plants leaked into model development')
    if not np.isfinite(df.loc[df.split != 'unused', NUMERIC + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite training input or label')
    return splits


def make_pipeline(estimator):
    return Pipeline([
        ('preprocess', ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                  ('scale', StandardScaler())]), NUMERIC),
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ])),
        ('regression', estimator),
    ])


def make_cpu_models(cfg, smoke=False):
    return {
        'ridge': make_pipeline(Ridge(alpha=1.0)),
        'random_forest': make_pipeline(RandomForestRegressor(
            n_estimators=20 if smoke else cfg['random_forest']['n_estimators'],
            max_depth=cfg['random_forest']['max_depth'],
            min_samples_leaf=cfg['random_forest']['min_samples_leaf'],
            max_features=cfg['random_forest']['max_features'],
            n_jobs=cfg['cpu_jobs'], random_state=cfg['seed'])),
        'gradient_boosting': make_pipeline(GradientBoostingRegressor(
            n_estimators=20 if smoke else cfg['gradient_boosting']['n_estimators'],
            learning_rate=cfg['gradient_boosting']['learning_rate'],
            max_depth=cfg['gradient_boosting']['max_depth'],
            min_samples_leaf=cfg['gradient_boosting']['min_samples_leaf'],
            random_state=cfg['seed'])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['CPU', 'GPU'], default='GPU',
                    help='CatBoost device only; sklearn RF/GB and Ridge always run on CPU')
    ap.add_argument('--smoke', action='store_true', help='20 trees; pipeline verification only, never a research result')
    args = ap.parse_args()
    cfg = read_config()
    data_path = ROOT / 'data' / 'processed' / 'model_data.csv'
    report = json.loads((ROOT / 'reports' / 'quality_report.json').read_text(encoding='utf-8'))
    if report['model_data_sha256'] != sha256(data_path):
        raise ValueError('Dataset hash changed; re-run prepare before training')
    if cfg != report['config']:
        raise ValueError('Config changed since prepare; re-run prepare so splits/config match')
    df = pd.read_csv(data_path, dtype={'cooling_id': str})
    parts = validate_splits(df)
    train, val = parts['train'], parts['validation']
    run_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + ('_smoke' if args.smoke else '_full')
    out = ROOT / 'runs' / run_name
    out.mkdir(parents=True, exist_ok=False)
    cpu_models = make_cpu_models(cfg, args.smoke)
    for name, estimator in cpu_models.items():
        print(f'Training {name} on CPU...', flush=True)
        estimator.fit(features(train), train[TARGET])
        joblib.dump(estimator, out / f'{name}.joblib')
    print(f'Training CatBoost gradient boosting on {args.device}...', flush=True)
    model = CatBoostRegressor(
        iterations=20 if args.smoke else cfg['iterations'], depth=cfg['depth'],
        learning_rate=cfg['learning_rate'], loss_function='RMSE', eval_metric='RMSE',
        random_seed=cfg['seed'], task_type=args.device,
        **({'devices': cfg['gpu_devices'], 'gpu_ram_part': 0.7} if args.device == 'GPU' else {'thread_count': 4}),
        allow_writing_files=False, verbose=100,
    )
    model.fit(features(train), train[TARGET], cat_features=CATEGORICAL,
              eval_set=(features(val), val[TARGET]),
              early_stopping_rounds=cfg['early_stopping_rounds'], use_best_model=True)
    model.save_model(str(out / 'catboost.cbm'))
    constant_delta = float(train[TARGET].mean())
    metrics, predictions = [], []
    for split, frame in parts.items():
        predictors = {
            'mean_delta': np.full(len(frame), constant_delta),
            **{name: estimator.predict(features(frame)) for name, estimator in cpu_models.items()},
            'catboost': model.predict(features(frame)),
        }
        for name, pred in predictors.items():
            y = frame[TARGET].to_numpy()
            error = pred - y
            per_plant = pd.DataFrame({'plant': frame.plant_id.to_numpy(), 'absolute_error': np.abs(error)})
            metrics.append({
                'split': split, 'model': name, 'n': len(frame),
                'mae_c': float(mean_absolute_error(y, pred)),
                'rmse_c': float(np.sqrt(mean_squared_error(y, pred))),
                'r2_delta': float(r2_score(y, pred)),
                'mean_bias_c': float(error.mean()),
                'macro_plant_mae_c': float(per_plant.groupby('plant').absolute_error.mean().mean())
            })
            p = frame[['plant_id', 'cooling_id', 'date', 'cooling_type', 'intake_temp_c', 'discharge_temp_c']].copy()
            p['split'], p['model'] = split, name
            p['predicted_delta_c'] = pred
            p['predicted_discharge_c'] = frame.intake_temp_c.to_numpy() + pred
            predictions.append(p)
    scores = pd.DataFrame(metrics)
    scores.to_csv(out / 'metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(predictions).to_csv(out / 'predictions.csv', index=False, encoding='utf-8-sig')
    selected = scores.loc[scores.split == 'validation'].sort_values('rmse_c').iloc[0]['model']
    pd.DataFrame({'feature': NUMERIC + CATEGORICAL, 'importance': model.feature_importances_}).to_csv(
        out / 'feature_importance.csv', index=False, encoding='utf-8-sig')
    dump_json(out / 'run.json', {
        'purpose': 'smoke_test_only' if args.smoke else 'research_training',
        'config': cfg, 'device': args.device, 'python': platform.python_version(),
        'model_devices': {**{name: 'CPU' for name in cpu_models}, 'catboost': args.device},
        'versions': {x: importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','catboost','openpyxl']},
        'data_sha256': sha256(data_path), 'features': NUMERIC + CATEGORICAL, 'target': TARGET,
        'selected_by_validation_rmse': selected, 'mean_delta_c': constant_delta,
        'best_iteration': model.get_best_iteration(),
        'input_training_ranges': {x: [float(train[x].min()), float(train[x].max())] for x in NUMERIC},
        'input_categories': {x: sorted(train[x].unique().tolist()) for x in CATEGORICAL},
        'notes': ['GPU floating-point reductions may be nondeterministic.',
                  'Contemporaneous monthly inputs; no prospective or real-time forecast claim.',
                  'Holdouts are never used for early stopping or model selection.',
                  'Temperature error is not a regulatory violation probability.']
    })
    print(scores.to_string(index=False))
    print(f'Run saved: {out}')


if __name__ == '__main__':
    main()
