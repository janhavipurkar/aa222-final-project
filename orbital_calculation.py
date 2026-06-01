#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbital_calculation.py

Encapsulates the Earth→asteroid→Earth mining-mission analysis from
orbit_simulation_2.py into a single callable:

    calculate_trajectory_mass(asteroid_index, asteroid_mass, plot=False)
        → (dv_outbound_kms,
            (cargo_LEO, cargo_GTO, cargo_GEO, cargo_L1, cargo_LLO),
            dry_mass_out_kg,
            prop_mass_out_kg,
            tof_outbound_days,
            loiter_days,
            tof_return_days)

Designed for use inside a surrogate-training loop where many evaluations
are needed and plotting / console output would slow things down.

Inputs
------
asteroid_index : int
    Row index into dataset_cleaned.csv (after dropping rows missing any of
    a, e, i, Ω, ω, M, epoch). Same indexing scheme as orbit_simulation_2.py.

asteroid_mass : float
    Wet mass at asteroid departure for the return leg, in kg.
    = spacecraft structure + mined cargo + return propellant.

plot : bool, default False
    If True: print verbose mission summary and save porkchop / trajectory plots.
    If False: run silently with no I/O, return values only.

Returns
-------
dv_outbound_kms : float
    Outbound launch Δv₁ in km/s. This is the hyperbolic excess velocity
    v∞ at Earth departure — what the launcher must deliver beyond LEO escape.
    NaN if the outbound transfer is infeasible.

cargo_kg : tuple of 5 floats
    Net asteroid material delivered to each Earth-vicinity orbit, in kg.
    = (mass arriving in destination orbit) − DRY_MASS_RET.
    Order: (LEO, GTO, GEO, L1/L2, LLO).
    NaN if the return transfer or capture into that orbit is infeasible.

Requires: numpy, pandas, matplotlib, pykep
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

import pykep as pk


# =============================================================================
# MODULE-LEVEL CONSTANTS  (edit here if you want different spacecraft / windows)
# =============================================================================
G0 = 9.80665                # m/s²

# --- Outbound spacecraft (chemical impulsive) ---
ISP_OUT             = 450.0        # s
DRY_MASS_OUT        = 55000.0      # kg https://en.wikipedia.org/wiki/M60_tank
DV_MARGIN_KMS       = 0.5          # safety margin on auto-sized outbound prop

# --- Return spacecraft (NTP impulsive-equivalent) ---
ISP_RET             = 850        # s (for NEP: https://arc.aiaa.org/doi/pdf/10.2514/6.2016-4950)
DRY_MASS_RET        = DRY_MASS_OUT / 2.0   # kg  (structure left for return leg)

# --- Launch / loiter / ToF windows ---
T0_LOW_STR          = "2035-01-01"
T0_HIGH_STR         = "2045-01-01"
TOF_MIN_DAYS        = 100
TOF_MAX_DAYS        = 1800
RET_DEP_OFFSET_MAX  = 8.0 * 365.25     # 5-year loiter window at asteroid
RET_TOF_MIN_DAYS    = 1
RET_TOF_MAX_DAYS    = 10 * 365

# --- Grid resolutions (smaller than the standalone script for speed) ---
N_T0                = 60
N_TOF               = 60
N_RET_DEP           = 60
N_RET_TOF           = 60

EPH_MAX_MJD2000     = 50.0 * 365.25    # pykep jpl_lp Earth valid only to ~2050

# --- Earth-vicinity capture Δv (km/s) ---
CAPTURE_DVS_KMS = [
    ("LEO (aerocapture-assisted)",    0.5),
    ("GTO (geostationary transfer)",  0.7),
    ("GEO (propulsive capture)",      3.0),
    ("L1 / L2 (Earth-Moon Lagrange)", 0.6),
    ("LLO (low lunar orbit)",         0.7),
]


# =============================================================================
# DATAFRAME CACHE  (avoid re-reading dataset_cleaned.csv on every call)
# =============================================================================
_df_orb_cache = None

def _load_data():
    global _df_orb_cache
    if _df_orb_cache is None:
        df = pd.read_csv("dataset_cleaned.csv", low_memory=False)
        req = ['a', 'e', 'i', 'om', 'w', 'ma', 'epoch_mjd']
        _df_orb_cache = df.dropna(subset=req).reset_index(drop=True)
    return _df_orb_cache


# =============================================================================
# UTILITIES
# =============================================================================
def _date_to_mjd2000(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (dt - datetime(2000, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0


def _build_asteroid(row):
    """Build a pk.planet.keplerian from a dataset row."""
    MJD_TO_MJD2000 = 51544.0
    mjd2000_ref    = float(row['epoch_mjd']) - MJD_TO_MJD2000
    ref_epoch      = pk.epoch(mjd2000_ref, "mjd2000")

    a_m = float(row['a']) * pk.AU
    e   = float(row['e'])
    i   = np.radians(float(row['i']))
    W   = np.radians(float(row['om']))
    w   = np.radians(float(row['w']))
    M   = np.radians(float(row['ma']))

    diam_km = row.get('diameter', np.nan)
    radius_m      = float(diam_km) * 500.0 if pd.notna(diam_km) else 1000.0
    safe_radius_m = radius_m * 1.1

    return pk.planet.keplerian(
        ref_epoch,
        (a_m, e, i, W, w, M),
        pk.MU_SUN, 1.0,
        radius_m, safe_radius_m,
        str(row['pdes']),
    )


def _lambert_dv(t0_mjd2000, tof_days, earth, asteroid):
    """Outbound Lambert: Earth(t0) → asteroid(t0+tof).
    Returns (dv1, dv2) in m/s, NaN on degenerate geometry."""
    try:
        t0 = pk.epoch(t0_mjd2000,            "mjd2000")
        tf = pk.epoch(t0_mjd2000 + tof_days, "mjd2000")
        r1, v_e = earth.eph(t0)
        r2, v_a = asteroid.eph(tf)
        lam = pk.lambert_problem(r1, r2, tof_days * pk.DAY2SEC, pk.MU_SUN,
                                 cw=False, max_revs=0)
        v1 = np.array(lam.get_v1()[0])
        v2 = np.array(lam.get_v2()[0])
        return (float(np.linalg.norm(v1 - np.array(v_e))),
                float(np.linalg.norm(np.array(v_a) - v2)))
    except Exception:
        return np.nan, np.nan


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def calculate_trajectory_mass(asteroid_index, asteroid_mass, plot=False):
    """See module docstring."""
    # NOTE: we deliberately do NOT call matplotlib.use(...) here — switching
    # backends mid-process is fragile and stickily breaks subsequent calls
    # with the opposite plot flag. The user's environment (Spyder, Jupyter,
    # terminal) sets the right backend at startup; we just respect it.

    def vprint(*args, **kwargs):
        if plot:
            print(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Load asteroid
    # -------------------------------------------------------------------------
    df_orb = _load_data()
    row    = df_orb.iloc[asteroid_index]

    vprint(f"\n=== Selected asteroid (index {asteroid_index}) ===")
    vprint(f"  full_name : {row['full_name']}")
    vprint(f"  pdes      : {row['pdes']}")
    vprint(f"  class     : {row['class']}")
    vprint(f"  a,e,i     : {row['a']:.3f} AU, {row['e']:.3f}, {row['i']:.2f}°")

    asteroid = _build_asteroid(row)
    earth    = pk.planet.jpl_lp("earth")

    t0_low  = _date_to_mjd2000(T0_LOW_STR)
    t0_high = _date_to_mjd2000(T0_HIGH_STR)

    # -------------------------------------------------------------------------
    # OUTBOUND PORKCHOP
    # -------------------------------------------------------------------------
    t0_grid  = np.linspace(t0_low,        t0_high,        N_T0)
    tof_grid = np.linspace(TOF_MIN_DAYS,  TOF_MAX_DAYS,   N_TOF)

    DV1 = np.full((N_T0, N_TOF), np.nan)
    DV2 = np.full((N_T0, N_TOF), np.nan)
    for i, t0 in enumerate(t0_grid):
        for j, tof in enumerate(tof_grid):
            d1, d2 = _lambert_dv(t0, tof, earth, asteroid)
            DV1[i, j] = d1
            DV2[i, j] = d2
    DV_total = DV1 + DV2

    if np.all(np.isnan(DV_total)):
        return (float('nan'), (float('nan'),) * 5,
                float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'))

    # Auto-size outbound propellant from the min-Δv cell
    dv_min_kms   = float(np.nanmin(DV_total)) / 1000.0
    dv_sized     = dv_min_kms + DV_MARGIN_KMS
    PROP_MASS    = DRY_MASS_OUT * (np.exp(dv_sized * 1000.0 / (ISP_OUT * G0)) - 1.0)
    WET_MASS_OUT = DRY_MASS_OUT + PROP_MASS

    mass_ratio  = np.exp(DV_total / (ISP_OUT * G0))
    prop_needed = WET_MASS_OUT * (1.0 - 1.0 / mass_ratio)
    feasible    = prop_needed <= PROP_MASS

    if not feasible.any():
        return (float('nan'), (float('nan'),) * 5,
                float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'))

    # Pick minimum-ToF feasible cell
    tof_mesh = np.tile(tof_grid, (N_T0, 1))
    tof_feas = np.where(feasible, tof_mesh, np.inf)
    i_best, j_best = np.unravel_index(np.argmin(tof_feas), tof_feas.shape)
    t0_best  = float(t0_grid[i_best])
    tof_best = float(tof_grid[j_best])
    dv1_best = float(DV1[i_best, j_best])
    dv2_best = float(DV2[i_best, j_best])
    dv_total_best = dv1_best + dv2_best
    t0_ep    = pk.epoch(t0_best,            "mjd2000")
    tf_ep    = pk.epoch(t0_best + tof_best, "mjd2000")

    vprint(f"\n=== Outbound best (min-ToF feasible) ===")
    vprint(f"  Δv1 (v∞ at Earth) : {dv1_best/1000:.3f} km/s")
    vprint(f"  Δv2 (rendezvous)  : {dv2_best/1000:.3f} km/s")
    vprint(f"  ToF               : {tof_best:.1f} d")
    vprint(f"  PROP_MASS (auto)  : {PROP_MASS:,.0f} kg")

    # -------------------------------------------------------------------------
    # RETURN PORKCHOP
    # -------------------------------------------------------------------------
    WET_MASS_RET    = float(asteroid_mass)
    if WET_MASS_RET <= DRY_MASS_RET:
        # Wet mass below dry mass — no propellant available, return infeasible
        vprint(f"  Return infeasible: asteroid_mass ({asteroid_mass}) ≤ "
               f"DRY_MASS_RET ({DRY_MASS_RET})")
        return (dv1_best / 1000.0, (float('nan'),) * 5,
                float(DRY_MASS_OUT), float(PROP_MASS),
                float(tof_best), float('nan'), float('nan'))
    PROP_BUDGET_RET = WET_MASS_RET - DRY_MASS_RET

    ret_dep_offsets = np.linspace(0.0, RET_DEP_OFFSET_MAX, N_RET_DEP)
    ret_tofs        = np.linspace(RET_TOF_MIN_DAYS, RET_TOF_MAX_DAYS, N_RET_TOF)

    RET_DV1 = np.full((N_RET_DEP, N_RET_TOF), np.nan)
    RET_DV2 = np.full((N_RET_DEP, N_RET_TOF), np.nan)

    for i, dep_off in enumerate(ret_dep_offsets):
        dep_mjd = tf_ep.mjd2000 + dep_off
        if dep_mjd > EPH_MAX_MJD2000:
            continue
        try:
            r_ast_d, v_ast_d = asteroid.eph(pk.epoch(dep_mjd, "mjd2000"))
        except Exception:
            continue
        r_ast_d = np.asarray(r_ast_d, float)
        v_ast_d = np.asarray(v_ast_d, float)
        for j, tof_d in enumerate(ret_tofs):
            arr_mjd = dep_mjd + tof_d
            if arr_mjd > EPH_MAX_MJD2000:
                continue
            try:
                r_e_a, v_e_a = earth.eph(pk.epoch(arr_mjd, "mjd2000"))
            except Exception:
                continue
            r_e_a = np.asarray(r_e_a, float)
            v_e_a = np.asarray(v_e_a, float)
            try:
                lam = pk.lambert_problem(r_ast_d, r_e_a,
                                         tof_d * pk.DAY2SEC, pk.MU_SUN,
                                         cw=False, max_revs=0)
                v1 = np.asarray(lam.get_v1()[0], float)
                v2 = np.asarray(lam.get_v2()[0], float)
                RET_DV1[i, j] = np.linalg.norm(v1 - v_ast_d)
                RET_DV2[i, j] = np.linalg.norm(v_e_a - v2)
            except Exception:
                pass

    RET_DV_TOTAL    = RET_DV1 + RET_DV2
    RET_MASS_RATIO  = np.exp(RET_DV_TOTAL / (ISP_RET * G0))
    RET_PROP_NEEDED = WET_MASS_RET * (1.0 - 1.0 / RET_MASS_RATIO)
    RET_FEASIBLE    = RET_PROP_NEEDED <= PROP_BUDGET_RET

    if not RET_FEASIBLE.any():
        vprint("  Return infeasible: no porkchop cell fits the propellant budget.")
        return (dv1_best / 1000.0, (float('nan'),) * 5,
                float(DRY_MASS_OUT), float(PROP_MASS),
                float(tof_best), float('nan'), float('nan'))

    # Pick global minimum-Δv feasible cell
    ret_dv_feas = np.where(RET_FEASIBLE, RET_DV_TOTAL, np.inf)
    i_b, j_b    = np.unravel_index(int(np.argmin(ret_dv_feas)), ret_dv_feas.shape)
    dv1_ret      = float(RET_DV1[i_b, j_b])
    dv2_ret      = float(RET_DV2[i_b, j_b])
    dv_total_ret = dv1_ret + dv2_ret

    vprint(f"\n=== Return best (min-Δv feasible) ===")
    vprint(f"  Δv total          : {dv_total_ret/1000:.3f} km/s")
    vprint(f"  Loiter at asteroid: {float(ret_dep_offsets[i_b]):.0f} d")
    vprint(f"  ToF return        : {float(ret_tofs[j_b]):.0f} d")

    # -------------------------------------------------------------------------
    # CAPTURE → cargo into each destination orbit
    #
    # mass_in_orbit = WET / exp((dv_lambert + dv_capture) / (Isp · g0))
    # cargo         = mass_in_orbit − DRY_MASS_RET     (subtract spacecraft structure)
    # Cell NaN if mass_in_orbit < DRY_MASS_RET (propellant exhausted before arrival).
    # -------------------------------------------------------------------------
    dv_lambert_kms = dv_total_ret / 1000.0
    cargo_values   = []
    for dest, dv_cap_kms in CAPTURE_DVS_KMS:
        dv_tot_kms    = dv_lambert_kms + dv_cap_kms
        mass_in_orbit = WET_MASS_RET / np.exp(dv_tot_kms * 1000.0 / (ISP_RET * G0))
        if mass_in_orbit < DRY_MASS_RET:
            cargo_values.append(float('nan'))
        else:
            cargo_values.append(float(mass_in_orbit - DRY_MASS_RET))

    if plot:
        print("\n  --- Net asteroid cargo (mass in orbit − DRY_MASS_RET) ---")
        for (dest, dv_cap), cargo in zip(CAPTURE_DVS_KMS, cargo_values):
            cargo_str = f"{cargo:,.1f} kg" if not np.isnan(cargo) \
                         else "INFEASIBLE"
            print(f"      {dest:32s} Δv_cap={dv_cap:.2f} km/s   →  {cargo_str}")

    # -------------------------------------------------------------------------
    # OPTIONAL PLOTTING
    # -------------------------------------------------------------------------
    if plot:
        _generate_plots(
            row, asteroid, earth,
            t0_grid, tof_grid, DV_total,
            t0_best, tof_best, dv1_best, dv2_best, dv_total_best,
            PROP_MASS, WET_MASS_OUT,
            ret_dep_offsets, ret_tofs, RET_DV_TOTAL,
            i_b, j_b, dv_total_ret,
            WET_MASS_RET, DRY_MASS_RET,
            t0_ep, tf_ep,
        )
        plt.show()

    # Return:
    #   dv_outbound_kms   — v∞ at Earth departure (km/s)
    #   cargo_kg          — 5-tuple of net asteroid material delivered (kg)
    #   dry_mass_kg       — outbound dry mass of spacecraft (kg)
    #   prop_mass_kg      — auto-sized outbound propellant (kg)
    #   tof_out_days      — outbound time of flight  (days)
    #   loiter_days       — wait at asteroid before return (days)
    #   tof_ret_days      — return time of flight    (days)
    return (
        dv1_best / 1000.0,
        tuple(cargo_values),
        float(DRY_MASS_OUT),
        float(PROP_MASS),
        float(tof_best),
        float(ret_dep_offsets[i_b]),
        float(ret_tofs[j_b]),
    )


# =============================================================================
# PLOTTING  (separated so the main function stays readable; only invoked when plot=True)
# =============================================================================
def _generate_plots(row, asteroid, earth,
                    t0_grid, tof_grid, DV_total,
                    t0_best, tof_best, dv1_best, dv2_best, dv_total_best,
                    PROP_MASS, WET_MASS_OUT,
                    ret_dep_offsets, ret_tofs, RET_DV_TOTAL,
                    i_b, j_b, dv_total_ret,
                    WET_MASS_RET, DRY_MASS_RET,
                    t0_ep, tf_ep):
    """Render the 3 standard plots (outbound porkchop, return porkchop, full
    mission map). Figures are NOT closed — Spyder will pick them up for inline
    display."""
    pdes = str(row['pdes'])

    def _mjd2000_to_date(m):
        return datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(days=float(m))

    # ========== 1. Outbound porkchop ==========================================
    t0_dates = np.array([_mjd2000_to_date(m) for m in t0_grid])
    dv_km    = DV_total / 1000.0
    levels   = np.linspace(np.nanmin(dv_km), np.nanpercentile(dv_km, 90), 30)

    fig, ax = plt.subplots(figsize=(13, 8), facecolor='#0e0e16')
    ax.set_facecolor('#0e0e16')
    cf = ax.contourf(t0_dates, tof_grid, dv_km.T, levels=levels, cmap='viridis_r')
    ax.contour(t0_dates, tof_grid, dv_km.T, levels=10,
               colors='black', alpha=0.25, linewidths=0.5)
    budget_dv = ISP_OUT * G0 * np.log(WET_MASS_OUT / DRY_MASS_OUT) / 1000.0
    ax.contour(t0_dates, tof_grid, dv_km.T, levels=[budget_dv],
               colors='#ff5533', linewidths=1.5, linestyles='--')
    ax.scatter([_mjd2000_to_date(t0_best)], [tof_best], s=200, marker='*',
               color='#ffd633', edgecolor='black', linewidth=1, zorder=10)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label('Total Δv (km/s)', color='#dddddd')
    plt.setp(cb.ax.get_yticklabels(), color='#cccccc')
    ax.set_xlabel('Departure date', color='#cccccc')
    ax.set_ylabel('ToF (days)',     color='#cccccc')
    ax.tick_params(colors='#aaaaaa')
    for s in ax.spines.values(): s.set_edgecolor('#333344')
    ax.set_title(f"Outbound porkchop: Earth → {pdes}",
                 color='#eeeeff', fontsize=11)
    plt.tight_layout()
    out1 = f"porkchop_outbound_{pdes}.png"
    plt.savefig(out1, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  saved {out1}")

    # ========== 2. Return porkchop ============================================
    ret_dep_dates = np.array(
        [_mjd2000_to_date(tf_ep.mjd2000 + off) for off in ret_dep_offsets])
    dv_km_ret = RET_DV_TOTAL / 1000.0
    levels_ret = np.linspace(np.nanmin(dv_km_ret),
                             np.nanpercentile(dv_km_ret, 90), 30)

    fig2, ax2 = plt.subplots(figsize=(13, 8), facecolor='#0e0e16')
    ax2.set_facecolor('#0e0e16')
    cf2 = ax2.contourf(ret_dep_dates, ret_tofs, dv_km_ret.T,
                       levels=levels_ret, cmap='viridis_r')
    ax2.contour(ret_dep_dates, ret_tofs, dv_km_ret.T, levels=10,
                colors='black', alpha=0.25, linewidths=0.5)
    budget_ret = ISP_RET * G0 * np.log(WET_MASS_RET / DRY_MASS_RET) / 1000.0
    ax2.contour(ret_dep_dates, ret_tofs, dv_km_ret.T, levels=[budget_ret],
                colors='#ff5533', linewidths=1.5, linestyles='--')
    ax2.scatter([_mjd2000_to_date(tf_ep.mjd2000 + ret_dep_offsets[i_b])],
                [ret_tofs[j_b]], s=200, marker='*', color='#ffd633',
                edgecolor='black', linewidth=1, zorder=10)
    cb2 = fig2.colorbar(cf2, ax=ax2, pad=0.02)
    cb2.set_label('Return Δv (km/s)', color='#dddddd')
    plt.setp(cb2.ax.get_yticklabels(), color='#cccccc')
    ax2.set_xlabel('Depart asteroid',   color='#cccccc')
    ax2.set_ylabel('Return ToF (days)', color='#cccccc')
    ax2.tick_params(colors='#aaaaaa')
    for s in ax2.spines.values(): s.set_edgecolor('#333344')
    ax2.set_title(f"Return porkchop: {pdes} → Earth",
                  color='#eeeeff', fontsize=11)
    plt.tight_layout()
    out2 = f"porkchop_return_{pdes}.png"
    plt.savefig(out2, dpi=160, bbox_inches='tight', facecolor=fig2.get_facecolor())
    print(f"  saved {out2}")

    # ========== 3. Full-mission heliocentric map ==============================
    dep_off_best = float(ret_dep_offsets[i_b])
    tof_best_ret = float(ret_tofs[j_b])
    ret_dep_ep   = pk.epoch(tf_ep.mjd2000 + dep_off_best,                 "mjd2000")
    ret_arr_ep   = pk.epoch(tf_ep.mjd2000 + dep_off_best + tof_best_ret,  "mjd2000")

    # Outbound Lambert arc
    r1_out, v_e_out = earth.eph(t0_ep)
    r2_out, v_a_out = asteroid.eph(tf_ep)
    lam_out = pk.lambert_problem(r1_out, r2_out, tof_best * pk.DAY2SEC,
                                 pk.MU_SUN, cw=False, max_revs=0)
    v1_out_lambert = np.asarray(lam_out.get_v1()[0], float)
    N_ARC = 250
    arc_out = np.zeros((N_ARC, 3))
    ts_out  = np.linspace(0.0, tof_best * pk.DAY2SEC, N_ARC)
    for k, ts in enumerate(ts_out):
        r_k, _ = pk.propagate_lagrangian(list(r1_out), list(v1_out_lambert),
                                         ts, pk.MU_SUN)
        arc_out[k] = r_k
    arc_out_au = arc_out / pk.AU

    # Return Lambert arc
    r1_ret, v_ast_ret = asteroid.eph(ret_dep_ep)
    r2_ret, v_e_ret   = earth.eph(ret_arr_ep)
    lam_ret = pk.lambert_problem(r1_ret, r2_ret, tof_best_ret * pk.DAY2SEC,
                                 pk.MU_SUN, cw=False, max_revs=0)
    v1_ret_lambert = np.asarray(lam_ret.get_v1()[0], float)
    arc_ret = np.zeros((N_ARC, 3))
    ts_ret  = np.linspace(0.0, tof_best_ret * pk.DAY2SEC, N_ARC)
    for k, ts in enumerate(ts_ret):
        r_k, _ = pk.propagate_lagrangian(list(r1_ret), list(v1_ret_lambert),
                                         ts, pk.MU_SUN)
        arc_ret[k] = r_k
    arc_ret_au = arc_ret / pk.AU

    # Reference orbits — Earth (one year) + asteroid (one period)
    def _ref_orbit(planet, ref_epoch, n=400):
        r0, v0 = planet.eph(ref_epoch)
        r0a = np.asarray(r0, float); v0a = np.asarray(v0, float)
        sma = 1.0 / (2.0 / np.linalg.norm(r0a) - np.dot(v0a, v0a) / pk.MU_SUN)
        T   = 2.0 * np.pi * np.sqrt(sma**3 / pk.MU_SUN)
        pts = np.zeros((n, 3))
        for k, t in enumerate(np.linspace(0.0, T, n)):
            r_k, _ = pk.propagate_lagrangian(list(r0a), list(v0a), t, pk.MU_SUN)
            pts[k] = r_k
        return pts / pk.AU

    earth_orbit_au = _ref_orbit(earth,    t0_ep)
    ast_orbit_au   = _ref_orbit(asteroid, t0_ep)

    # Loiter arc along asteroid's orbit (between tf_ep and ret_dep_ep)
    loiter_au = None
    if dep_off_best > 0.5:
        N_LO = max(60, int(dep_off_best * 0.5))
        loiter_arr = np.zeros((N_LO, 3))
        for k, mjd in enumerate(np.linspace(tf_ep.mjd2000,
                                            tf_ep.mjd2000 + dep_off_best, N_LO)):
            r_k, _ = asteroid.eph(pk.epoch(mjd, "mjd2000"))
            loiter_arr[k] = np.asarray(r_k, float)
        loiter_au = loiter_arr / pk.AU

    fig3, ax3 = plt.subplots(figsize=(11, 11), facecolor='#0a0a12')
    ax3.set_facecolor('#0a0a12'); ax3.set_aspect('equal')

    theta = np.linspace(0, 2*np.pi, 360)
    for pa, pcolor in [(0.387, '#aaaaaa'), (0.723, '#e8c97a'), (1.524, '#dd4422')]:
        ax3.plot(pa*np.cos(theta), pa*np.sin(theta),
                 color=pcolor, linewidth=0.6, alpha=0.4)

    ax3.plot(earth_orbit_au[:, 0], earth_orbit_au[:, 1],
             color='#4488ff', linewidth=1.0, alpha=0.7, label='Earth orbit')
    ax3.plot(ast_orbit_au[:, 0], ast_orbit_au[:, 1],
             color='#ff8855', linewidth=1.0, alpha=0.7, label=f"{pdes} orbit")
    ax3.plot(arc_out_au[:, 0], arc_out_au[:, 1],
             color='#ffd633', linewidth=1.8, label='Outbound')
    if loiter_au is not None:
        ax3.plot(loiter_au[:, 0], loiter_au[:, 1],
                 color='#ff8855', linewidth=2.4, alpha=0.95, linestyle=':',
                 label=f'Loiter ({dep_off_best:.0f} d)')
    ax3.plot(arc_ret_au[:, 0], arc_ret_au[:, 1],
             color='#66ddff', linewidth=1.5, label='Return (NTP)')

    r1_au = np.asarray(r1_out, float) / pk.AU
    r2_au = np.asarray(r2_out, float) / pk.AU
    r_ret_dep_au = np.asarray(r1_ret, float) / pk.AU
    r_ret_arr_au = np.asarray(r2_ret, float) / pk.AU

    ax3.scatter([0], [0], s=150, color='#ffe066', marker='*', zorder=11)
    ax3.scatter([r1_au[0]], [r1_au[1]], s=60, c='#4488ff', edgecolor='white',
                linewidth=0.7, zorder=10, label='Earth @ launch')
    ax3.scatter([r2_au[0]], [r2_au[1]], s=60, c='#ff5533', edgecolor='white',
                linewidth=0.7, zorder=10, label='Asteroid @ arrival')
    ax3.scatter([r_ret_dep_au[0]], [r_ret_dep_au[1]], s=60, c='#ffaa33',
                edgecolor='white', linewidth=0.7, zorder=10, marker='s',
                label='Asteroid @ return depart')
    ax3.scatter([r_ret_arr_au[0]], [r_ret_arr_au[1]], s=60, c='#4488ff',
                edgecolor='white', linewidth=0.7, zorder=10, marker='D',
                label='Earth @ return')

    lim = max(np.max(np.abs(arc_out_au[:, :2])),
              np.max(np.abs(arc_ret_au[:, :2])),
              np.max(np.abs(ast_orbit_au[:, :2]))) * 1.1
    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim)
    ax3.grid(color='#1e1e2e', linewidth=0.4, linestyle='--')
    ax3.set_xlabel('X (AU)', color='#888899')
    ax3.set_ylabel('Y (AU)', color='#888899')
    ax3.tick_params(colors='#666677')
    for s in ax3.spines.values(): s.set_edgecolor('#222233')
    ax3.legend(loc='upper left', fontsize=8.5, framealpha=0.2,
               facecolor='#0d0d1a', edgecolor='#333344', labelcolor='#dddddd')
    ax3.set_title(
        f"Mining mission: Earth → {pdes} → Earth\n"
        f"Outbound: {tof_best:.0f} d, Δv={dv_total_best/1000:.2f} km/s    "
        f"Return: {tof_best_ret:.0f} d, Δv={dv_total_ret/1000:.2f} km/s",
        color='#eeeeff', fontsize=10.5, pad=10)
    plt.tight_layout()
    out3 = f"mission_map_{pdes}.png"
    plt.savefig(out3, dpi=160, bbox_inches='tight', facecolor=fig3.get_facecolor())
    print(f"  saved {out3}")


# =============================================================================
# COMPOSITION LOOKUP  (thin wrapper around composition_value.py)
# =============================================================================
def composition(asteroid_index):
    """
    Return (total_mass_kg, value_per_kg_usd, classification) for one asteroid.

    Wraps composition_value.py:
        - get_composition_class()  →  resource class ('C', 'S', 'M', or None)
        - estimate_mass()          →  bulk mass from diameter or H magnitude
        - value_per_kg()           →  $/kg of bulk asteroid material at the
                                       Asterank / Webster (2013) reference prices
                                       baked into composition_value.PRICE_PER_KG.

    Parameters
    ----------
    asteroid_index : int
        Row index into the cleaned dataset (same indexing as
        calculate_trajectory_mass).

    Returns
    -------
    total_mass_kg : float
        Estimated bulk mass of the asteroid (kg).  NaN if no diameter / H
        magnitude / albedo combination is available to estimate it.
    value_per_kg_usd : float
        $/kg of *bulk* material (already weighted by composition fractions).
        Zero for S-type (no extractable mass) and for 'unknown'.
    classification : str
        Resource class — one of 'C', 'S', 'M', or 'unknown'.
    """
    from composition_value import (
        get_composition_class, estimate_mass, value_per_kg,
        RHO, RHO_DEFAULT,
    )

    df  = _load_data()
    row = df.iloc[asteroid_index]

    comp_class, _ = get_composition_class(row)
    if comp_class is None:
        return float('nan'), 0.0, 'unknown'

    rho    = RHO.get(comp_class, RHO_DEFAULT)
    m_bulk = estimate_mass(row, rho)
    m_bulk = float(m_bulk) if pd.notna(m_bulk) else float('nan')

    return m_bulk, float(value_per_kg(comp_class)), comp_class


# =============================================================================
# EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Itokawa (index 25142)
    print("Quick test: Itokawa, asteroid_mass = 100_000 kg, plot=True")
    dv_out, cargo, dry_mass, prop_mass, tof_out, loiter, tof_ret = \
        calculate_trajectory_mass(25142, 100_000.0, plot=True)
    print(f"\n→ Δv outbound (km/s) : {dv_out:.3f}")
    print(f"→ Dry mass (kg)      : {dry_mass:,.0f}")
    print(f"→ Prop mass (kg)     : {prop_mass:,.0f}")
    print(f"→ Cargo to destinations (kg):")
    for (dest, _), m in zip(CAPTURE_DVS_KMS, cargo):
        m_str = f"{m:,.1f}" if not np.isnan(m) else "INFEASIBLE"
        print(f"    {dest:32s}  {m_str}")

    # And a silent call to demonstrate the surrogate-training use case
    print("\nSilent call (plot=False, no I/O):")
    dv_out2, cargo2, _, _, _, _, _ = calculate_trajectory_mass(25142, 100_000.0, plot=False)
    print(f"  Δv = {dv_out2:.3f},  cargo = {cargo2}")
