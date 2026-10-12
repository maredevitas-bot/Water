import unittest

import numpy as np
import pandas as pd

from ripple.features import NUMERIC, CATEGORICAL
from ripple.prepare import assign_splits, enrich, fahrenheit_to_celsius, quality_flags
from ripple.train import validate_splits


class PipelineTests(unittest.TestCase):
    def test_temperature_units_and_missing(self):
        self.assertAlmostEqual(fahrenheit_to_celsius(68), 20)
        self.assertTrue(np.isnan(fahrenheit_to_celsius('.')))

    def test_volume_uses_calendar_not_operating_hours(self):
        x = pd.DataFrame([dict(year=2024,month=2,plant_nuclear_generation_mwh=696000,
                              withdrawal_million_gal=1, discharge_temp_c=30,intake_temp_c=20,
                              cooling_type='Once through without cooling pond(s) or canal(s)')])
        out = enrich(x).iloc[0]
        self.assertEqual(out.calendar_hours, 696)
        self.assertAlmostEqual(out.withdrawal_m3_s, 3785.411784 / (696*3600))
        self.assertEqual(out.plant_nuclear_mean_mw, 1000)

    def test_no_target_leakage(self):
        self.assertFalse(set(NUMERIC + CATEGORICAL) & {'discharge_temp_c','delta_temp_c','heat_load_mw'})

    def test_temporal_and_plant_holdout(self):
        df = pd.DataFrame([dict(plant_id=p,year=y,month=1,cooling_id='a',delta_temp_c=3,
                               **{c:1 for c in NUMERIC}) for p in range(20) for y in range(2021,2025)])
        cfg=dict(seed=42,heldout_plant_fraction=.2,train_end_year=2022,validation_year=2023,test_year=2024)
        out, held = assign_splits(df,cfg)
        parts=validate_splits(out)
        self.assertEqual(len(held),4)
        self.assertFalse(set(parts['train'].plant_id) & set(parts['test_unseen_plants'].plant_id))
        self.assertEqual(set(parts['test_temporal'].year),{2024})
        bad=out.copy()
        bad.loc[bad.split=='validation','year']=2021
        with self.assertRaises(ValueError): validate_splits(bad)

    def test_missing_and_zero_are_not_imputed_to_labels(self):
        x=pd.DataFrame([dict(intake_temp_c=np.nan,discharge_temp_c=20,withdrawal_m3_s=0,
                             plant_nuclear_mean_mw=100,hours_in_service=20,calendar_hours=744,
                             cooling_status='OP',delta_temp_c=np.nan,once_through=True)])
        reasons=quality_flags(x,'once_through').iloc[0]
        self.assertIn('missing_intake_temperature',reasons)
        self.assertIn('missing_or_nonpositive_withdrawal',reasons)


if __name__=='__main__': unittest.main()
