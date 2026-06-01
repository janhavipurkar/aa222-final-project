#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer.py

Single-asteroid mining-mission evaluation + economics comparison.

Loads classified_asteroids.csv (produced by composition_value.py), picks one
asteroid, runs the trajectory + capture analysis, and compares:

    A) Asteroid mining mission cost vs.
    B) Direct LEO→LLO launch cost for the same payload.

If (A) < (B), asteroid mining is the cheaper way to deliver this mass to LLO.
"""

import numpy as np
import pandas as pd

from orbital_calculation import calculate_trajectory_mass, composition


# ============================================================================
# ECONOMICS  (commodity-style cost model)
# ============================================================================
# Sources:
#   Propellant feedstock prices — https://www.nextbigfuture.com/2022/02/
#     spacex-reusable-rocket-costs-versus-airplanes.html
#   Mining rig fixed cost — rough order-of-magnitude proxy: https://en.wikipedia.org/wiki/M60_tank
G0                       = 9.80665                  # m/s²

oxygen_price_per_kg      = 0.16                     # $/kg LOX
methane_price_per_kg     = 0.40                     # $/kg CH4
# Methalox stoichiometric ratio ≈ 4:1 (O:F by mass) → 80% O₂ / 20% CH₄
propellant_price_per_kg  = 0.8 * oxygen_price_per_kg + 0.2 * methane_price_per_kg

LAUNCH_COST_PER_KG_LEO   = 1000.0                   # $/kg to LEO (Falcon-Heavy-class)
FIXED_MISSION_COST_USD   = 5_000_000.0              # one mining rig

# Methalox upper-stage parameters for the LEO→LLO comparison launch
ISP_LAUNCH_S             = 450.0                    # s vacuum (Raptor / BE-4 class)
DV_LEO_TO_LLO_M_S        = 4100.0                   # m/s  (TLI 3.15 + LOI 0.95)


def mission_cost(dry_mass, launch_propellant_mass):
    """
    Total cost of the asteroid mining mission.

    dry_mass + launch_propellant_mass are loaded into LEO (so we pay
    $/kg-to-LEO on both), then the propellant itself has a per-kg feedstock
    cost on top, plus the fixed mining-rig overhead.
    """
    return (
        LAUNCH_COST_PER_KG_LEO * (dry_mass + launch_propellant_mass)
        + FIXED_MISSION_COST_USD
        + launch_propellant_mass * propellant_price_per_kg
    )


def equivalent_launch_to_low_lunar_orbit(LLO_cargo, dry_mass):
    """
    Cost of delivering (LLO_cargo + dry_mass) to LLO by direct chemical
    launch from LEO instead of via asteroid mining.

    Methalox single-stage rocket equation:
        m_prop = (cargo + dry) · (exp(Δv / (Isp · g₀)) − 1)

    Then total cost is the same as mission_cost: launch cost + propellant
    feedstock + fixed overhead.
    """
    mass_ratio = np.exp(DV_LEO_TO_LLO_M_S / (ISP_LAUNCH_S * G0))
    launch_propellant_mass = (LLO_cargo + dry_mass) * (mass_ratio - 1.0)

    return (
        LAUNCH_COST_PER_KG_LEO * (dry_mass + LLO_cargo + launch_propellant_mass)
        + FIXED_MISSION_COST_USD
        + launch_propellant_mass * propellant_price_per_kg
    )


# ============================================================================
# net_cost(data) — single-asteroid economic + timing summary
# ============================================================================
# "Net cost" here is defined as
#       net_cost = mission_cost − revenue_from_LLO_cargo
# so that NEGATIVE values mean the mission is profitable. This is a "cost"
# from the optimizer's perspective — minimize it. (For Pareto vs time, the
# other axis to minimize is mission_time_days.)
#
# `data` accepts either an int (a dataset_index) or a pandas Series / dict
# carrying a 'dataset_index' field. Returns a dict of everything we know
# about the asteroid that's relevant to the GP / Pareto analysis.
# ============================================================================
def net_cost(data, asteroid_mass=500_000.0):
    from orbital_calculation import calculate_trajectory_mass, composition

    # Resolve the dataset index from whatever the user passed
    if isinstance(data, (int, np.integer)):
        asteroid_index = int(data)
    elif hasattr(data, 'get'):                 # pandas Series / dict
        asteroid_index = int(data['dataset_index'])
    else:
        raise TypeError(f"net_cost: don't know how to interpret data={data!r}")

    # Composition
    total_mass_kg, value_per_kg, cls = composition(asteroid_index)

    # Trajectory + economics
    (dv_out, cargo, dry_mass_out, prop_mass_out,
     tof_out_d, loiter_d, tof_ret_d) = calculate_trajectory_mass(
            asteroid_index = asteroid_index,
            asteroid_mass  = asteroid_mass,
            plot           = False,
        )

    LLO_cargo = cargo[4]                       # index 4 = 'LLO' in DESTINATIONS

    record = {
        'dataset_index'     : asteroid_index,
        'comp_class'        : cls,
        'value_per_kg'      : value_per_kg,
        'total_mass_kg'     : total_mass_kg,
        'dv_out_kms'        : dv_out,
        'cargo_LLO_kg'      : LLO_cargo,
        'dry_mass_kg'       : dry_mass_out,
        'prop_mass_kg'      : prop_mass_out,
        'tof_out_days'      : tof_out_d,
        'loiter_days'       : loiter_d,
        'tof_ret_days'      : tof_ret_d,
        'mission_time_days' : np.nan,
        'mission_cost_usd'  : np.nan,
        'revenue_LLO_usd'   : np.nan,
        'net_cost_usd'      : np.nan,
        'direct_cost_usd'   : np.nan,
        'feasible'          : False,
    }

    if np.isnan(dv_out):
        return record                          # outbound infeasible

    record['mission_cost_usd'] = mission_cost(dry_mass_out, prop_mass_out)

    if np.isnan(LLO_cargo) or np.isnan(tof_ret_d):
        return record                          # return / LLO infeasible

    record['mission_time_days'] = float(tof_out_d + loiter_d + tof_ret_d)
    record['revenue_LLO_usd']   = float(LLO_cargo * value_per_kg)
    record['net_cost_usd']      = float(record['mission_cost_usd']
                                        - record['revenue_LLO_usd'])
    record['direct_cost_usd']   = equivalent_launch_to_low_lunar_orbit(
                                        LLO_cargo, dry_mass_out)
    record['feasible']          = True
    return record


# ============================================================================
# Load classified asteroid catalog
# ============================================================================
classified = pd.read_csv("classified_asteroids.csv")
print(f"Loaded {len(classified):,} classified asteroids "
      f"(composition + orbit both known)")
print(f"  Class breakdown: {dict(classified['comp_class'].value_counts())}")


# ============================================================================
# CASE: pick a target
# ============================================================================
neo_classified = classified[classified['neo'] == 'Y'].dropna(subset=['value_usd'])
target = neo_classified.sort_values('value_usd', ascending=False).iloc[3]

ASTEROID_INDEX       = int(target['dataset_index'])
ASTEROID_MASS_RETURN = 500_000.0     # kg wet at asteroid departure

print(f"\nSelected target: {target['full_name'].strip()}")
print(f"  dataset_index : {ASTEROID_INDEX}")
print(f"  comp_class    : {target['comp_class']}  (via {target['class_method']})")
print(f"  E[M_useful]   : {target['E_M_useful_kg']:.3e} kg")


# ============================================================================
# Step 1: composition
# ============================================================================
total_mass_kg, value_per_kg, cls = composition(ASTEROID_INDEX)
print(f"\n  composition() → total_mass={total_mass_kg:.3e} kg, "
      f"${value_per_kg:.6f}/kg, class={cls}")


# ============================================================================
# Step 2: trajectory + cargo + spacecraft sizing
# ============================================================================
(dv_out, cargo, dry_mass_out, prop_mass_out,
 tof_out_d, loiter_d, tof_ret_d) = calculate_trajectory_mass(
    asteroid_index = ASTEROID_INDEX,
    asteroid_mass  = ASTEROID_MASS_RETURN,
    plot           = True,
)

DESTINATIONS = ['LEO', 'GTO', 'GEO', 'L1/L2', 'LLO']

print(f"\n=== Mission analysis (return wet mass = {ASTEROID_MASS_RETURN:,.0f} kg) ===")
print(f"  Δv outbound (v∞ at Earth) : {dv_out:.3f} km/s")
print(f"  Outbound dry mass         : {dry_mass_out:,.0f} kg")
print(f"  Outbound propellant       : {prop_mass_out:,.0f} kg")
print()
print(f"  {'Destination':10s}  {'Cargo (kg)':>14s}  {'Value ($)':>16s}")
print(f"  {'-'*10:10s}  {'-'*14:>14s}  {'-'*16:>16s}")
for dest, cargo_kg in zip(DESTINATIONS, cargo):
    if np.isnan(cargo_kg):
        print(f"  {dest:10s}  {'INFEASIBLE':>14s}  {'—':>16s}")
    else:
        cargo_value_usd = cargo_kg * value_per_kg
        print(f"  {dest:10s}  {cargo_kg:>14,.1f}  {cargo_value_usd:>16,.2f}")


# ============================================================================
# Step 3: ECONOMICS — asteroid mission vs. direct LEO→LLO launch
# ============================================================================
print(f"\n=== Economics ===")

# (a) Total mission cost — uses outbound dry + launch prop from the trajectory
if not (np.isnan(dry_mass_out) or np.isnan(prop_mass_out)):
    cost_mission = mission_cost(dry_mass_out, prop_mass_out)
    print(f"  (a) Mission cost                  : ${cost_mission:>18,.0f}")
    print(f"      dry mass                      : {dry_mass_out:>12,.0f} kg")
    print(f"      launch propellant             : {prop_mass_out:>12,.0f} kg")
else:
    cost_mission = float('nan')
    print(f"  (a) Mission cost                  : N/A (outbound infeasible)")

# (b) Direct LEO→LLO launch — fair comparison: deliver the same LLO cargo
#     using a methalox stage from LEO, with the same dry mass overhead.
LLO_cargo_from_asteroid = cargo[DESTINATIONS.index('LLO')]
if not np.isnan(LLO_cargo_from_asteroid):
    cost_direct = equivalent_launch_to_low_lunar_orbit(
        LLO_cargo    = LLO_cargo_from_asteroid,
        dry_mass     = dry_mass_out,
    )
    print(f"  (b) Direct LEO→LLO launch cost    : ${cost_direct:>18,.0f}")
    print(f"      LLO cargo delivered           : {LLO_cargo_from_asteroid:>12,.0f} kg")
    print(f"      dry mass (assumed same)       : {dry_mass_out:>12,.0f} kg")
    print(f"      Δv assumed                    : {DV_LEO_TO_LLO_M_S/1000:>12.2f} km/s")
    print(f"      Isp methalox (vacuum)         : {ISP_LAUNCH_S:>12.0f} s")
else:
    cost_direct = float('nan')
    print(f"  (b) Direct LEO→LLO launch cost    : N/A (no LLO cargo)")




# (c) Verdict
print()
if not (np.isnan(cost_mission) or np.isnan(cost_direct)):
    savings = cost_direct - cost_mission
    if savings > 0:
        print(f"  → Asteroid mining is CHEAPER by ${savings:,.0f} "
              f"({100*savings/cost_direct:.1f} % of direct cost)")
    else:
        print(f"  → Direct launch is CHEAPER by ${-savings:,.0f} "
              f"({100*(-savings)/cost_mission:.1f} % of mission cost)")
