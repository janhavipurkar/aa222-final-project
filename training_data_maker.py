#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_data_maker.py

Builds a 1000-asteroid training set for the GP surrogate.

Pipeline:
    1. Load classified_asteroids.csv (NEOs only, valid composition + orbit).
    2. Stratified sampling by composition class (C, M, S) with equal allocation.
    3. Within each stratum, Latin Hypercube sampling on (a, e, i),
       snapped to nearest real asteroid.
    4. For each sampled asteroid, run optimizer.net_cost(...).
    5. Save complete table to training_set.csv with all columns needed
       for Pareto-front analysis (time vs profit) and GP training.

Output columns include:
    orbital   : a, e, i, q, moid, diameter
    composition: comp_class, value_per_kg, total_mass_kg
    trajectory: dv_out_kms, cargo_LLO_kg, dry_mass_kg, prop_mass_kg
    timing    : tof_out_days, loiter_days, tof_ret_days, mission_time_days
    economics : mission_cost_usd, revenue_LLO_usd, net_cost_usd, direct_cost_usd
    feasibility: feasible (bool), verdict

NOTE: expect ~1-2 s per evaluation in pykep. 1000 asteroids ≈ 20-30 min.
"""

import time
import numpy as np
import pandas as pd
from scipy.stats        import qmc
from sklearn.neighbors  import NearestNeighbors

from optimizer import net_cost                # uses orbital_calculation under the hood


# ============================================================================
# CONFIG
# ============================================================================
SEED                 = 42
N_PER_CLASS          = 1666          # 333+333+334 = 1000
CLASSES              = ['C', 'M', 'S']
USE_NEOS_ONLY        = False         # mining targets ↔ near-Earth
ASTEROID_MASS_RETURN = 500_000.0    # kg wet at asteroid departure
INPUT_CATALOG        = "classified_asteroids.csv"
OUT_SAMPLE_CSV       = "training_sample_5000_nep.csv"
OUT_TRAINING_CSV     = "training_set_5000_nep.csv"

# ============================================================================
# 1. Load + filter catalog
# ============================================================================
print(f"Loading {INPUT_CATALOG} ...")
df = pd.read_csv(INPUT_CATALOG, low_memory=False)
print(f"  {len(df):,} classified asteroids")

df = df.dropna(subset=['a', 'e', 'i', 'comp_class', 'dataset_index'])
if USE_NEOS_ONLY:
    df = df[df['neo'] == 'Y']
    print(f"  {len(df):,} are classified NEOs")
print(f"  Class breakdown: {dict(df['comp_class'].value_counts())}")


# ============================================================================
# 2-3. Stratified Latin Hypercube sample
# ============================================================================
print("\nBuilding stratified Latin Hypercube sample ...")
all_samples = []

for cls in CLASSES:
    pool = df[df['comp_class'] == cls].reset_index(drop=True)
    if len(pool) == 0:
        print(f"  [{cls}] no asteroids in pool, skipping")
        continue

    # Need the LHS dims to be populated — drop pool members missing om/w
    pool = pool.dropna(subset=['a', 'e', 'i', 'om', 'w']).reset_index(drop=True)
    if len(pool) == 0:
        print(f"  [{cls}] no asteroids with full (a,e,i,om,w), skipping")
        continue
    n_take = min(N_PER_CLASS, len(pool))

    # Per-stratum [1, 99]-percentile range — keeps LHS coverage tight to
    # where this composition class actually lives in (a, e, i, Ω, ω).
    # For angles (om, w), this collapses to roughly [0°, 360°] since they
    # span the full range uniformly; we use percentiles for consistency.
    aL, aH = pool['a' ].quantile([0.01, 0.99])
    eL, eH = pool['e' ].quantile([0.01, 0.99])
    iL, iH = pool['i' ].quantile([0.01, 0.99])
    oL, oH = pool['om'].quantile([0.01, 0.99])     # Ω, RAAN
    wL, wH = pool['w' ].quantile([0.01, 0.99])     # ω, arg of perihelion

    pool_norm = np.column_stack([
        (pool['a' ] - aL) / (aH - aL + 1e-12),
        (pool['e' ] - eL) / (eH - eL + 1e-12),
        (pool['i' ] - iL) / (iH - iL + 1e-12),
        (pool['om'] - oL) / (oH - oL + 1e-12),
        (pool['w' ] - wL) / (wH - wL + 1e-12),
    ])

    # LHS in normalized [0,1]^5
    sampler  = qmc.LatinHypercube(d=5, seed=SEED + ord(cls))
    lhs_norm = sampler.random(n_take)

    # Snap to nearest real asteroid — WITHOUT REPLACEMENT.
    # Reason: when real asteroids cluster in (a,e,i) space, multiple LHS
    # targets in sparse regions all snap to the same nearest asteroid and
    # collapse to one sample. Without-replacement guarantees n_take unique
    # picks. Greedy K-nearest works well for this scale.
    K = min(64, len(pool_norm))             # candidates per LHS target
    nn = NearestNeighbors(n_neighbors=K).fit(pool_norm)
    _, nn_idx = nn.kneighbors(lhs_norm)     # (n_take, K) sorted by distance

    picked        = set()
    picked_in_pool = []
    spills        = 0
    for i in range(n_take):
        for cand in nn_idx[i]:
            if cand not in picked:
                picked.add(int(cand))
                picked_in_pool.append(int(cand))
                break
        else:
            # All K nearest are taken — fall back to any unpicked asteroid
            remaining = list(set(range(len(pool_norm))) - picked)
            cand = remaining[0]
            picked.add(cand)
            picked_in_pool.append(cand)
            spills += 1

    chosen = pool.iloc[picked_in_pool].copy()
    chosen['stratum_class'] = cls
    all_samples.append(chosen)

    print(f"  [{cls}] pool={len(pool):,}, LHS picked {n_take}, "
          f"unique={chosen['dataset_index'].nunique()}"
          + (f"   ({spills} fallback spills)" if spills else ""))

samples = pd.concat(all_samples, ignore_index=True)
samples = samples.drop_duplicates('dataset_index').reset_index(drop=True)
samples.to_csv(OUT_SAMPLE_CSV, index=False)
print(f"\nSaved sample → {OUT_SAMPLE_CSV} "
      f"({len(samples)} unique asteroids: "
      f"{dict(samples['comp_class'].value_counts())})")


# ============================================================================
# 4. Run net_cost on each sampled asteroid
# ============================================================================
print(f"\nEvaluating net_cost for {len(samples)} asteroids "
      f"(asteroid_mass={ASTEROID_MASS_RETURN:,.0f} kg) ...")
t_start = time.time()

records = []
for k, row in samples.iterrows():
    idx = int(row['dataset_index'])
    try:
        rec = net_cost(row, asteroid_mass=ASTEROID_MASS_RETURN)
    except Exception as ex:
        rec = {
            'dataset_index' : idx,
            'comp_class'    : row['comp_class'],
            'feasible'      : False,
            'error'         : f"{type(ex).__name__}: {ex}",
        }
    records.append(rec)

    if (k + 1) % 20 == 0 or k + 1 == len(samples):
        elapsed = time.time() - t_start
        eta     = elapsed / (k + 1) * (len(samples) - k - 1)
        n_feasible = sum(r.get('feasible', False) for r in records)
        print(f"  [{k+1:>4d}/{len(samples)}] "
              f"elapsed {elapsed:>6.0f} s   eta {eta:>6.0f} s   "
              f"feasible so far: {n_feasible}")

print(f"\nDone in {time.time() - t_start:.0f} s")


# ============================================================================
# 5. Merge and save
# ============================================================================
results_df = pd.DataFrame(records)

# Verdict column
def _verdict(r):
    if not r.get('feasible', False):
        if pd.isna(r.get('dv_out_kms', np.nan)):
            return 'outbound_infeasible'
        return 'return_or_LLO_infeasible'
    if r['mission_cost_usd'] < r['direct_cost_usd']:
        return 'mining_cheaper_than_direct'
    return 'direct_cheaper_than_mining'

results_df['verdict'] = results_df.apply(_verdict, axis=1)

# Merge with catalog so the training set is self-contained
catalog_cols   = ['dataset_index', 'full_name', 'pdes', 'neo',
                  'a', 'e', 'i', 'q', 'om', 'w',         # orbital elements
                  'moid', 'diameter',
                  'E_M_useful_kg', 'sigma_M_kg', 'value_usd']
catalog_subset = samples[[c for c in catalog_cols if c in samples.columns]]
final_df = catalog_subset.merge(results_df, on='dataset_index',
                                how='inner', suffixes=('', '_eval'))

# ============================================================================
# INFEASIBILITY → PHYSICALLY MEANINGFUL PENALTY for GP training
# ----------------------------------------------------------------------------
#   outbound_infeasible       — never launched, no cost paid. DROP these
#                                (they're pure noise for the GP).
#   return_or_LLO_infeasible  — launched but stranded / nothing returned.
#                                We paid mission_cost and got zero revenue
#                                ⇒ net_cost = mission_cost (full loss).
#                                These rows are KEPT — they teach the GP
#                                where the feasibility boundary is.
#   feasible                  — net_cost = mission_cost − revenue   (unchanged)
#
# After this step:
#   - net_cost_usd is non-NaN for every retained row
#   - profit_usd = -net_cost_usd  is a single continuous training target
# ============================================================================
print(f"\nPre-filter rows: {len(final_df)}  "
      f"(verdict breakdown: {dict(final_df['verdict'].value_counts())})")

# Drop unlaunched
final_df = final_df[final_df['verdict'] != 'outbound_infeasible'].copy()

# For stranded missions: net_cost = mission_cost, revenue = 0, direct_cost = 0
#   "direct_cost = 0" interpretation: the do-nothing alternative is to not
#   launch the equivalent mission at all, since stranded means we got 0 LLO
#   cargo. That gives direct_minus_mining = -mission_cost (mining lost us
#   the full cost) — a finite, physically-meaningful penalty signal for the GP.
stranded = final_df['verdict'] == 'return_or_LLO_infeasible'
final_df.loc[stranded, 'revenue_LLO_usd']     = 0.0
final_df.loc[stranded, 'net_cost_usd']        = final_df.loc[stranded, 'mission_cost_usd']
final_df.loc[stranded, 'direct_cost_usd']     = 0.0
# mission_time_days for stranded = ToF_out only (we never returned)
final_df.loc[stranded, 'mission_time_days']   = final_df.loc[stranded, 'tof_out_days']

# Sanity — every retained row must have net_cost_usd
n_bad = int(final_df['net_cost_usd'].isna().sum())
if n_bad > 0:
    print(f"  WARNING: {n_bad} rows still have NaN net_cost — investigate")

# Profit column for convenience
final_df['profit_usd'] = -final_df['net_cost_usd']

# ----------------------------------------------------------------------------
# GP TRAINING TARGETS
# ----------------------------------------------------------------------------
#   direct_minus_mining_usd  — direct-launch cost minus mining net cost.
#                               POSITIVE = asteroid mining saves money vs
#                               just launching the cargo from LEO.
#                               This is the "value-added by mining" signal.
#
#   total_time_years         — full mission duration in YEARS
#                               (outbound + loiter + return), or just
#                               outbound for stranded missions.
# ----------------------------------------------------------------------------
final_df['direct_minus_mining_usd'] = (
    final_df['direct_cost_usd'] - final_df['net_cost_usd']
)
final_df['total_time_years'] = final_df['mission_time_days'] / 365.25

final_df.to_csv(OUT_TRAINING_CSV, index=False)
print(f"\nSaved training set → {OUT_TRAINING_CSV}  "
      f"({len(final_df)} rows, {len(final_df.columns)} columns)")


# ============================================================================
# Summary
# ============================================================================
print(f"\n--- Training set summary ---")
print(f"  Total          : {len(final_df):,}")
print(f"  Feasible       : {int(final_df['feasible'].sum()):,}")
print(f"\n  Verdict counts:")
print(final_df['verdict'].value_counts().to_string())

feas = final_df[final_df['feasible']].copy()
if not feas.empty:
    print(f"\n  Mission time (days) — feasible only:")
    print(f"    p10={feas['mission_time_days'].quantile(.10):>6.0f}  "
          f"median={feas['mission_time_days'].median():>6.0f}  "
          f"p90={feas['mission_time_days'].quantile(.90):>6.0f}")
    print(f"  Net cost (USD) — feasible only:")
    print(f"    p10={feas['net_cost_usd'].quantile(.10):>15,.0f}  "
          f"median={feas['net_cost_usd'].median():>15,.0f}  "
          f"p90={feas['net_cost_usd'].quantile(.90):>15,.0f}")
    n_profit = int((feas['net_cost_usd'] < 0).sum())
    print(f"  Profitable     : {n_profit:>4d} / {len(feas):>4d} "
          f"({100*n_profit/len(feas):.1f} %)")

# ----------------------------------------------------------------------------
# GP-target distribution (across ALL retained rows — feasible + stranded)
# ----------------------------------------------------------------------------
print(f"\n--- GP training targets (all retained rows: "
      f"{len(final_df):,}) ---")
print(f"  direct_minus_mining_usd  (positive = mining saves money):")
print(f"    p10={final_df['direct_minus_mining_usd'].quantile(.10):>15,.0f}  "
      f"median={final_df['direct_minus_mining_usd'].median():>15,.0f}  "
      f"p90={final_df['direct_minus_mining_usd'].quantile(.90):>15,.0f}")
n_mining_wins = int((final_df['direct_minus_mining_usd'] > 0).sum())
print(f"    Mining beats direct: {n_mining_wins:>5,} / {len(final_df):>5,}  "
      f"({100*n_mining_wins/len(final_df):5.1f} %)")
print(f"  total_time_years:")
print(f"    p10={final_df['total_time_years'].quantile(.10):>6.2f}  "
      f"median={final_df['total_time_years'].median():>6.2f}  "
      f"p90={final_df['total_time_years'].quantile(.90):>6.2f}")

print(f"\n  → Pareto-front recipe:")
print(f"        x = total_time_years            (minimize)")
print(f"        y = direct_minus_mining_usd     (maximize)")
print(f"     Dominating set = asteroids where neither (smaller time AND larger")
print(f"     savings) exists among any other asteroid.")


# ============================================================================
# PARETO-FRONT PLOT  →  saved as PNG
# ============================================================================
import matplotlib
matplotlib.use("Agg")          # headless-safe; we only savefig, never show
import matplotlib.pyplot as plt

OUT_PARETO_PNG = "pareto_time_vs_savings.png"

def _pareto_indices(df, x_col, y_col):
    """Return df indices on the Pareto front (min x, max y)."""
    pts = df[[x_col, y_col]].dropna().sort_values(x_col)
    best_y     = -np.inf
    keep_index = []
    for idx, (t, p) in zip(pts.index, pts.itertuples(index=False)):
        if p > best_y:
            keep_index.append(idx)
            best_y = p
    return keep_index

front_idx = _pareto_indices(final_df, 'total_time_years',
                                       'direct_minus_mining_usd')
front     = final_df.loc[front_idx]

fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0e0e16')
ax.set_facecolor('#0e0e16')

# Background scatter — colour by composition class
cmap = {'C': '#66ccff', 'M': '#dddddd', 'S': '#ffaa33'}
for cls, c in cmap.items():
    sub = final_df[final_df['comp_class'] == cls]
    if not sub.empty:
        ax.scatter(sub['total_time_years'], sub['direct_minus_mining_usd'],
                   c=c, s=10, alpha=0.45, label=f'{cls}-type (n={len(sub):,})',
                   linewidths=0)

# Pareto front — bigger, red, line-connected
front_sorted = front.sort_values('total_time_years')
ax.plot(front_sorted['total_time_years'],
        front_sorted['direct_minus_mining_usd'],
        color='#ff5533', linewidth=1.5, alpha=0.6, zorder=4)
ax.scatter(front_sorted['total_time_years'],
           front_sorted['direct_minus_mining_usd'],
           c='#ff5533', s=45, edgecolor='white', linewidth=0.6, zorder=5,
           label=f'Pareto front (n={len(front)})')

# Zero-savings line
ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4)

ax.set_xlabel('Total mission time (years)        — minimize',
              color='#cccccc')
ax.set_ylabel('Direct cost − mining cost ($)     — maximize',
              color='#cccccc')
ax.tick_params(colors='#aaaaaa')
for s in ax.spines.values():
    s.set_edgecolor('#333344')
ax.grid(color='#1e1e2e', linewidth=0.4, linestyle='--', alpha=0.6)
ax.legend(facecolor='#0d0d1a', edgecolor='#333344', labelcolor='#dddddd',
          framealpha=0.4, loc='best', fontsize=9)
ax.set_title("Pareto front: mission time vs mining-vs-direct savings",
             color='#eeeeff', fontsize=11)

plt.tight_layout()
plt.savefig(OUT_PARETO_PNG, dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved Pareto plot → {OUT_PARETO_PNG}")
