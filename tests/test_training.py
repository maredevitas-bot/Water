"""Offline end-to-end engineering tests; synthetic fixtures are not research data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ripple import predict, train
from ripple.common import dump_json, read_config, sha256
from ripple.features import CATEGORICAL, NUMERIC, TARGET
from ripple.prepare import assign_splits


class TrainingTests(unittest.TestCase):
    def test_all_models_train_save_reload_and_predict(self):
        cfg = read_config()
        cfg['cpu_jobs'] = 1
        rng = np.random.default_rng(17)
        rows = []
        for plant in range(10):
            for year in range(2021, 2025):
                for month in [1, 2, 3]:
                    row = {c: float(rng.uniform(1, 10)) for c in NUMERIC}
                    row.update(plant_id=plant, cooling_id='a', year=year, month=month,
                               date=f'{year}-{month:02d}-01', cooling_type='once through',
                               state='AA' if plant % 2 else 'BB')
                    row[TARGET] = row['intake_temp_c'] * .2 + rng.normal(0, .1)
                    row['discharge_temp_c'] = row['intake_temp_c'] + row[TARGET]
                    rows.append(row)
        df, _ = assign_splits(pd.DataFrame(rows), cfg)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data/processed/model_data.csv'
            data.parent.mkdir(parents=True)
            df.to_csv(data, index=False)
            dump_json(root / 'reports/quality_report.json',
                      {'config': cfg, 'model_data_sha256': sha256(data)})
            with patch.object(train, 'ROOT', root), patch.object(train, 'read_config', return_value=cfg):
                with patch('sys.argv', ['train', '--device', 'CPU', '--smoke']), contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                run = next((root / 'runs').iterdir())
                metrics = pd.read_csv(run / 'metrics.csv')
                expected = {'mean_delta', 'ridge', 'random_forest', 'gradient_boosting', 'catboost'}
                self.assertEqual(set(metrics.model), expected)
                self.assertEqual(len(metrics), 20)
                self.assertTrue(np.isfinite(metrics.rmse_c).all())
                meta = json.loads((run / 'run.json').read_text())
                best = metrics[metrics.split == 'validation'].sort_values('rmse_c').iloc[0].model
                self.assertEqual(meta['selected_by_validation_rmse'], best)
                inputs = root / 'input.csv'
                sample = df[df.split == 'test_temporal'].head(3).copy()
                sample[NUMERIC + CATEGORICAL].to_csv(inputs, index=False)
                saved = pd.read_csv(run / 'predictions.csv')
                for name in expected:
                    meta['selected_by_validation_rmse'] = name
                    dump_json(run / 'run.json', meta)
                    output = root / f'{name}.csv'
                    argv = ['predict', '--run', str(run), '--input', str(inputs), '--output', str(output)]
                    with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'Smoke models'):
                        predict.main()
                    with patch('sys.argv', argv + ['--allow-smoke']), contextlib.redirect_stdout(io.StringIO()):
                        predict.main()
                    actual = pd.read_csv(output)
                    original = saved[(saved.split == 'test_temporal') & (saved.model == name)].head(3)
                    np.testing.assert_allclose(actual.predicted_delta_c, original.predicted_delta_c)
                    np.testing.assert_allclose(actual.predicted_discharge_c,
                                               actual.intake_temp_c + actual.predicted_delta_c)
                    with patch('sys.argv', argv + ['--allow-smoke']), self.assertRaisesRegex(ValueError, 'new output'):
                        predict.main()
                # The same CLI must reject stale dataset provenance before starting a run.
                data.write_text(data.read_text() + '\n')
                with patch('sys.argv', ['train', '--device', 'CPU', '--smoke']):
                    with self.assertRaisesRegex(ValueError, 'hash changed'):
                        train.main()

    def test_preprocessing_does_not_fit_on_prediction_categories(self):
        cfg = read_config()
        cfg['cpu_jobs'] = 1
        x = pd.DataFrame({c: [1., 2., 3., 4.] for c in NUMERIC})
        for c in CATEGORICAL:
            x[c] = 'known'
        for model in train.make_cpu_models(cfg, smoke=True).values():
            model.fit(x, [1., 2., 3., 4.])
            query = x.iloc[:1].copy()
            query['state'] = 'unseen'
            self.assertTrue(np.isfinite(model.predict(query)).all())
            encoder = model['preprocess'].named_transformers_['category']
            self.assertNotIn('unseen', encoder.categories_[1])


if __name__ == '__main__':
    unittest.main()
