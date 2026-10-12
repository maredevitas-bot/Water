import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor

from .features import CATEGORICAL, NUMERIC
from .train import features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--allow-smoke', action='store_true')
    args = ap.parse_args()
    meta = json.loads((args.run / 'run.json').read_text(encoding='utf-8'))
    if meta['purpose'] == 'smoke_test_only' and not args.allow_smoke:
        raise ValueError('Smoke models are not research models. Train a full run first.')
    df = pd.read_csv(args.input)
    if set(NUMERIC + CATEGORICAL) - set(df.columns):
        raise ValueError(f'Required columns: {NUMERIC + CATEGORICAL}')
    import numpy as np
    if not np.isfinite(df[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError('Numeric input has missing/non-finite values')
    if args.output.resolve() == args.input.resolve() or args.output.exists():
        raise ValueError('Choose a new output path; existing files are never overwritten')
    name = meta['selected_by_validation_rmse']
    if name == 'catboost':
        model = CatBoostRegressor()
        model.load_model(str(args.run / 'catboost.cbm'))
        delta = model.predict(features(df))
    elif name in {'ridge', 'random_forest', 'gradient_boosting'}:
        delta = joblib.load(args.run / f'{name}.joblib').predict(features(df))
    elif name == 'mean_delta':
        delta = np.full(len(df), meta['mean_delta_c'])
    else:
        raise ValueError(f'Unknown saved model: {name}')
    df['outside_training_domain'] = False
    for c, (lo, hi) in meta['input_training_ranges'].items():
        df.loc[~df[c].between(lo, hi), 'outside_training_domain'] = True
    for c in CATEGORICAL:
        df.loc[~df[c].isin(meta['input_categories'][c]), 'outside_training_domain'] = True
    df['predicted_delta_c'] = delta
    df['predicted_discharge_c'] = df.intake_temp_c + delta
    df['model'] = name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'Saved {len(df)} predictions. Outside training domain: {int(df.outside_training_domain.sum())}')


if __name__ == '__main__':
    main()
