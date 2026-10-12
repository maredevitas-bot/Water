"""One record per nuclear-site / cooling-system / month, with explicit QA."""
import calendar
import hashlib
import io
import json
import zipfile

import numpy as np
import openpyxl
import pandas as pd

from .common import ROOT, dump_json, normalize, read_config, sha256, timestamp

KEY = ['plant_id', 'cooling_id', 'year', 'month']


def table(z, member, sheet, header_marker):
    # Never extract archive paths to disk.
    wb = openpyxl.load_workbook(io.BytesIO(z.read(member)), read_only=True, data_only=True)
    try:
        ws = wb[sheet]
        headers = None
        for row_number, row in enumerate(ws.iter_rows(values_only=True), 1):
            if headers is None:
                candidate = [normalize(x) for x in row]
                if header_marker in candidate:
                    headers = candidate
                elif row_number >= 15:
                    raise ValueError(f'Header not found: {member}/{sheet}')
                continue
            if any(v is not None for v in row):
                record = dict(zip(headers, row))
                record['_source_row'] = row_number
                yield record
    finally:
        wb.close()


def member_matching(z, part):
    names = [x for x in z.namelist() if part in x and x.endswith('.xlsx')]
    if len(names) != 1:
        raise ValueError(f'Ambiguous workbook: {part}: {names}')
    return names[0]


def numeric(value):
    value = normalize(value)
    if value in {'', '.', 'NA', 'N/A', '--'}:
        return np.nan
    try:
        result = float(value.replace(',', ''))
    except ValueError as e:
        raise ValueError(f'Unrecognized numeric value {value!r}') from e
    return result if np.isfinite(result) else np.nan


def fahrenheit_to_celsius(value):
    return (numeric(value) - 32) * 5 / 9


def prefix_value(row, prefix):
    keys = [k for k in row if k.startswith(prefix)]
    if len(keys) != 1:
        raise ValueError(f'Expected one column starting with {prefix!r}: {keys}')
    return row[keys[0]]


def read_year(year):
    path = ROOT / 'data' / 'raw' / f'f923_{year}.zip'
    with zipfile.ZipFile(path) as z:
        gen_member = member_matching(z, 'Schedules_2_3_4_5')
        nuc = []
        seen = set()
        for r in table(z, gen_member, 'Page 1 Generation and Fuel Data', 'Nuclear Unit Id'):
            if normalize(r['Reported Fuel Type Code']) != 'NUC':
                continue
            if int(numeric(r['YEAR'])) != year:
                raise ValueError('Generation year does not match ZIP year')
            unit_key = (numeric(r['Plant Id']), normalize(r['Nuclear Unit Id']), normalize(r['Reported Prime Mover']))
            if unit_key in seen:
                raise ValueError(f'Duplicate nuclear unit record: {unit_key}')
            seen.add(unit_key)
            for month in range(1, 13):
                nuc.append({
                    'plant_id': int(numeric(r['Plant Id'])), 'year': year, 'month': month,
                    'plant_name': normalize(r['Plant Name']), 'state': normalize(r['Plant State']),
                    'unit_mwh': numeric(r[f'Netgen {calendar.month_name[month]}']),
                })
        units = pd.DataFrame(nuc)
        if units.empty:
            raise ValueError(f'No NUC rows: {year}')
        grouped = units.groupby(['plant_id', 'year', 'month'], sort=True)
        # If any reactor has missing generation, do not silently sum a partial plant total.
        gen = grouped.agg(plant_nuclear_generation_mwh=('unit_mwh', lambda x: x.sum(min_count=len(x))),
                          plant_name=('plant_name', 'first'), state=('state', 'first')).reset_index()
        ids = set(gen.plant_id)
        cool_member = member_matching(z, 'Schedule_8_')
        rows = []
        for r in table(z, cool_member, '8D Cooling System Information', 'Type of Cooling System'):
            if numeric(r['Plant ID']) not in ids:
                continue
            if int(numeric(r['Year'])) != year:
                raise ValueError('Cooling year does not match ZIP year')
            rows.append({
                'plant_id': int(numeric(r['Plant ID'])), 'year': year,
                'month': int(numeric(r['Month'])), 'cooling_id': normalize(r['Cooling System ID']),
                'cooling_type': normalize(r['Type of Cooling System']),
                'cooling_status': normalize(r['Cooling System Status']),
                'hours_in_service': numeric(r['Hours in Service']),
                'intake_temp_c': fahrenheit_to_celsius(prefix_value(r, 'Intake Average Temperature')),
                'discharge_temp_c': fahrenheit_to_celsius(prefix_value(r, 'Discharge Average Temperature')),
                'withdrawal_million_gal': numeric(prefix_value(r, 'Withdrawal Volume')),
                'discharge_million_gal': numeric(prefix_value(r, 'Discharge Volume')),
                'consumption_million_gal': numeric(prefix_value(r, 'Consumption Volume')),
                'flow_measurement_method': normalize(r['Method for Flow Rates']),
                'temperature_measurement_method': normalize(r['Method for Temperatures']),
                'source_zip': path.name, 'source_member': cool_member,
                'source_row': r['_source_row'], 'generation_source_member': gen_member,
            })
    frame = pd.DataFrame(rows)
    if frame.duplicated(KEY).any():
        raise ValueError(f'Duplicate cooling keys in {year}; must inspect before joining')
    frame = frame.merge(gen, on=['plant_id', 'year', 'month'], how='left', validate='many_to_one')
    return frame, {'year': year, 'nuclear_plants_in_generation': len(ids),
                   'nuclear_unit_rows': len(seen), 'cooling_rows_at_nuclear_sites': len(frame),
                   'raw_sha256': sha256(path)}


def enrich(df):
    df = df.copy()
    dates = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df['date'] = dates.dt.strftime('%Y-%m-%d')
    df['calendar_hours'] = dates.dt.days_in_month * 24
    df['plant_nuclear_mean_mw'] = df.plant_nuclear_generation_mwh / df.calendar_hours
    # Calendar-average withdrawal, NOT internal circulation flow or active-hour flow.
    df['withdrawal_m3_s'] = df.withdrawal_million_gal * 3785.411784 / (df.calendar_hours * 3600)
    df['delta_temp_c'] = df.discharge_temp_c - df.intake_temp_c
    df['month_sin'] = np.sin(2 * np.pi * df.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.month / 12)
    df['once_through'] = df.cooling_type.str.lower().str.startswith('once through')
    df['log1p_withdrawal_m3_s'] = np.log1p(df.withdrawal_m3_s.clip(lower=0))
    return df


def quality_flags(df, scope):
    flags = pd.Series('', index=df.index)
    def flag(mask, reason):
        flags.loc[mask] += reason + ';'
    flag(df.intake_temp_c.isna(), 'missing_intake_temperature')
    flag(df.discharge_temp_c.isna(), 'missing_discharge_temperature')
    flag(df.withdrawal_m3_s.isna() | (df.withdrawal_m3_s <= 0), 'missing_or_nonpositive_withdrawal')
    flag(df.plant_nuclear_mean_mw.isna() | (df.plant_nuclear_mean_mw <= 0), 'missing_or_nonpositive_generation')
    flag(df.hours_in_service.isna() | (df.hours_in_service <= 0) | (df.hours_in_service > df.calendar_hours + 1), 'invalid_operating_hours')
    flag(df.cooling_status != 'OP', 'not_operable')
    # Wide research plausibility bounds, not environmental/legal limits. Keep excluded rows for review.
    flag(df.intake_temp_c.notna() & ~df.intake_temp_c.between(-3, 50), 'intake_outside_QA_range')
    flag(df.discharge_temp_c.notna() & ~df.discharge_temp_c.between(-3, 70), 'discharge_outside_QA_range')
    flag(df.delta_temp_c.notna() & ~df.delta_temp_c.between(-5, 40), 'delta_outside_QA_range')
    if scope == 'once_through':
        flag(~df.once_through, 'outside_once_through_scope')
    elif scope != 'all':
        raise ValueError('cooling_scope must be once_through or all')
    return flags.str.rstrip(';')


def assign_splits(df, cfg):
    if not cfg['train_end_year'] < cfg['validation_year'] < cfg['test_year']:
        raise ValueError('Require train_end_year < validation_year < test_year')
    if not 0 < cfg['heldout_plant_fraction'] < 1:
        raise ValueError('heldout_plant_fraction must be between zero and one')
    out = df.copy()
    plants = sorted(out.plant_id.unique(), key=lambda p: hashlib.sha256(f'{cfg["seed"]}:{p}'.encode()).hexdigest())
    if len(plants) < 5:
        raise ValueError('At least five eligible plants required for plant holdout')
    held = set(plants[:max(1, round(len(plants) * cfg['heldout_plant_fraction']))])
    known = ~out.plant_id.isin(held)
    out['split'] = 'unused'
    out.loc[known & (out.year <= cfg['train_end_year']), 'split'] = 'train'
    out.loc[known & (out.year == cfg['validation_year']), 'split'] = 'validation'
    out.loc[known & (out.year == cfg['test_year']), 'split'] = 'test_temporal'
    out.loc[~known & (out.year == cfg['test_year']), 'split'] = 'test_unseen_plants'
    # Require every temporal validation/test site to have actual training records.
    trained = set(out.loc[out.split == 'train', 'plant_id'])
    out.loc[known & ~out.plant_id.isin(trained) & (out.split != 'unused'), 'split'] = 'unused'
    for split in ['train', 'validation', 'test_temporal', 'test_unseen_plants']:
        if not (out.split == split).any():
            raise ValueError(f'Empty split: {split}; inspect quality_report.json/config')
    return out, sorted(int(x) for x in held)


def main():
    cfg = read_config()
    raw = ROOT / 'data' / 'raw'
    manifest = {x['file']: x for x in json.loads((raw / 'manifest.json').read_text(encoding='utf-8'))}
    frames, inventory = [], []
    for year in cfg['years']:
        name = f'f923_{year}.zip'
        if sha256(raw / name) != manifest[name]['sha256']:
            raise ValueError(f'Raw hash mismatch: {name}')
        print(f'Preparing {year}...', flush=True)
        frame, stats = read_year(year)
        frames.append(frame)
        inventory.append(stats)
    all_rows = enrich(pd.concat(frames, ignore_index=True))
    if all_rows.duplicated(KEY).any():
        raise ValueError('Duplicate cooling-month keys across files')
    all_rows['exclusion_reasons'] = quality_flags(all_rows, cfg['cooling_scope'])
    outdir = ROOT / 'data' / 'processed'
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows.sort_values(KEY).to_csv(outdir / 'nuclear_cooling_all.csv', index=False, encoding='utf-8-sig')
    eligible = all_rows.loc[all_rows.exclusion_reasons == ''].copy()
    eligible, held = assign_splits(eligible, cfg)
    eligible = eligible.sort_values(KEY)
    eligible.to_csv(outdir / 'model_data.csv', index=False, encoding='utf-8-sig')
    # A real-data input example for post-training prediction, no target columns.
    from .features import NUMERIC, CATEGORICAL
    eligible.loc[eligible.split == 'test_temporal', NUMERIC + CATEGORICAL].head(5).to_csv(
        ROOT / 'example_inputs.csv', index=False, encoding='utf-8-sig')
    report = {
        'created_at_utc': timestamp(), 'config': cfg, 'annual_inventory': inventory,
        'all_nuclear_site_cooling_rows': len(all_rows), 'eligible_rows': len(eligible),
        'eligible_plants': int(eligible.plant_id.nunique()),
        'eligible_cooling_systems': int(eligible[['plant_id', 'cooling_id']].drop_duplicates().shape[0]),
        'split_rows': {k: int(v) for k, v in eligible.split.value_counts().items()},
        'heldout_plants': held, 'cooling_type_rows_before_QA': {k: int(v) for k, v in all_rows.cooling_type.value_counts().items()},
        'missing_fraction_before_QA': {k: float(all_rows[k].isna().mean()) for k in ['intake_temp_c', 'discharge_temp_c', 'withdrawal_m3_s', 'plant_nuclear_mean_mw']},
        'exclusion_reason_counts': {k: int(v) for k, v in all_rows.exclusion_reasons.str.split(';').explode().loc[lambda s:s != ''].value_counts().items()},
        'model_data_sha256': sha256(outdir / 'model_data.csv'),
        'limitations': ['Monthly conditional prediction, not an ahead-of-time forecast.',
                       'Plant nuclear generation is site context, never allocated to individual cooling systems.',
                       'No measured bypass ratio, cooling electricity, causal derating labels or legal limits.',
                       'Temperatures and flows include source-estimated records; measurement methods are retained.',
                       'No universal heat-load target is derived from monthly means and different stream boundaries.']
    }
    dump_json(ROOT / 'reports' / 'quality_report.json', report)
    print(json.dumps({k:report[k] for k in ['eligible_rows','eligible_plants','split_rows']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
