# Explicit allowlist prevents discharge temperature, delta-T and heat proxies from leaking into X.
NUMERIC = ['intake_temp_c', 'log1p_withdrawal_m3_s', 'plant_nuclear_mean_mw',
           'hours_in_service', 'month_sin', 'month_cos']
CATEGORICAL = ['cooling_type', 'state']
TARGET = 'delta_temp_c'
