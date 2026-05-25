#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_analyzer.py

Cleans NASA Small Body Database export (latest_fulldb.csv):
  - Drops columns irrelevant to trajectory design and asteroid mining value.
  - Drops cometary objects (keeps only asteroid-class bodies).
  - Writes the cleaned dataset to dataset_cleaned.csv.

Kept column groups
------------------
Identifiers      : id, spkid, full_name, pdes, name
Classification   : class, neo, pha
Orbital elements : e, a, q, i, om, w, ma
Epoch            : epoch_mjd
Derived orbital  : ad, n, tp, per
Accessibility    : moid, moid_jup, t_jup
Orbit uncertainty: sigma_e, sigma_a, sigma_q, sigma_i, sigma_om,
                   sigma_w, sigma_ma, sigma_ad, sigma_n, sigma_tp, sigma_per
Orbit quality    : n_obs_used, condition_code, rms
Mining / physical: H, H_sigma, diameter, diameter_sigma, albedo,
                   spec_B, spec_T, BV, UB
"""

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("Loading latest_fulldb.csv ...")
df = pd.read_csv('latest_fulldb.csv', low_memory=False)
print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Drop comet rows
#    Comet classes: COM, JFc, JFC, HTC, ETc, CTc, PAR, HYP
#    Asteroid classes kept: MBA, OMB, IMB, MCA, APO, AMO, ATE, TJN,
#                           TNO, CEN, AST, IEO, HYA
# ---------------------------------------------------------------------------
COMET_CLASSES = {'COM', 'JFc', 'JFC', 'HTC', 'ETc', 'CTc', 'PAR', 'HYP'}
n_before = len(df)
df = df[~df['class'].isin(COMET_CLASSES)].copy()
n_comets_dropped = n_before - len(df)
print(f"  Dropped {n_comets_dropped:,} comet rows → {len(df):,} asteroid rows remain")

# ---------------------------------------------------------------------------
# 3. Select columns to keep
# ---------------------------------------------------------------------------
KEEP_COLUMNS = [
    # --- Identifiers ---
    'id', 'spkid', 'full_name', 'pdes', 'name',

    # --- Classification ---
    'class', 'neo', 'pha',

    # --- Core Keplerian orbital elements (at epoch) ---
    'e',    # eccentricity
    'a',    # semi-major axis (au)
    'q',    # perihelion distance (au)
    'i',    # inclination (deg)
    'om',   # longitude of ascending node (deg)
    'w',    # argument of perihelion (deg)
    'ma',   # mean anomaly at epoch (deg)

    # --- Epoch ---
    'epoch_mjd',   # Modified Julian Date of osculating elements

    # --- Derived orbital quantities ---
    'ad',    # aphelion distance (au)
    'n',     # mean motion (deg/day)
    'tp',    # time of perihelion passage (JD)
    'per',   # orbital period (days)

    # --- Earth-accessibility metrics ---
    'moid',      # minimum orbit intersection distance with Earth (au)
    'moid_jup',  # MOID with Jupiter (au)
    't_jup',     # Tisserand parameter w.r.t. Jupiter

    # --- Orbit uncertainty (1-sigma) ---
    'sigma_e', 'sigma_a', 'sigma_q', 'sigma_i',
    'sigma_om', 'sigma_w', 'sigma_ma',
    'sigma_ad', 'sigma_n', 'sigma_tp', 'sigma_per',

    # --- Orbit solution quality ---
    'n_obs_used',      # number of observations used
    'condition_code',  # MPC orbit condition code (0=best)
    'rms',             # RMS of fit (arcsec)

    # --- Physical / mining-value properties ---
    'H',               # absolute magnitude (proxy for size)
    'H_sigma',         # uncertainty in H
    'diameter',        # effective diameter (km)
    'diameter_sigma',  # uncertainty in diameter (km)
    'albedo',          # geometric albedo (reflects surface composition)
    'spec_B',          # spectral type – Bus-DeMeo taxonomy
    'spec_T',          # spectral type – Tholen taxonomy
    'BV',              # B-V color index (composition indicator)
    'UB',              # U-B color index (composition indicator)
]

# Only keep columns that actually exist in the dataframe
KEEP_COLUMNS = [c for c in KEEP_COLUMNS if c in df.columns]
df = df[KEEP_COLUMNS]
print(f"  Kept {len(KEEP_COLUMNS)} columns (dropped {75 - len(KEEP_COLUMNS)} columns)")

# ---------------------------------------------------------------------------
# 4. Drop physically invalid orbits (a < 0 or e > 1)
# ---------------------------------------------------------------------------
n_before = len(df)
df = df[(df['a'] >= 0) & (df['e'] <= 1)].copy()
n_invalid = n_before - len(df)
print(f"  Dropped {n_invalid:,} rows with a < 0 or e > 1 → {len(df):,} rows remain")

# ---------------------------------------------------------------------------
# 5. Write cleaned dataset
# ---------------------------------------------------------------------------
output_path = 'dataset_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\nWrote cleaned dataset → {output_path}")
print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 5. Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)

print(f"\nTotal asteroids: {len(df):,}")

print("\n--- Class breakdown ---")
print(df['class'].value_counts().to_string())

print("\n--- NEO / PHA counts ---")
print(f"  Near-Earth Objects (neo=Y): {(df['neo']=='Y').sum():,}")
print(f"  Potentially Hazardous (pha=Y): {(df['pha']=='Y').sum():,}")

print("\n--- Physical data coverage ---")
for col in ['H', 'diameter', 'albedo', 'spec_B', 'spec_T', 'BV', 'UB']:
    n_valid = df[col].notna().sum()
    print(f"  {col:20s}: {n_valid:>8,}  ({100*n_valid/len(df):.1f}% coverage)")

print("\n--- Orbital element coverage ---")
for col in ['e', 'a', 'q', 'i', 'om', 'w', 'ma', 'moid']:
    n_valid = df[col].notna().sum()
    print(f"  {col:20s}: {n_valid:>8,}  ({100*n_valid/len(df):.1f}% coverage)")

print("\n--- Key orbital element ranges (asteroids with data) ---")
for col, unit in [('a','au'), ('e',''), ('i','deg'), ('moid','au'), ('per','days')]:
    s = df[col].dropna()
    if len(s):
        print(f"  {col:6s} ({unit}): min={s.min():.4g}, median={s.median():.4g}, "
              f"max={s.max():.4g}")

print("\n--- Spectral type distribution (Bus-DeMeo, spec_B) ---")
print(df['spec_B'].value_counts().head(20).to_string())

print("\n--- Condition code distribution ---")
print(df['condition_code'].value_counts().sort_index().to_string())

print("\nDone.")
