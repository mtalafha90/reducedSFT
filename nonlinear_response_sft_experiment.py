#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled nonlinear response experiments for a reduced Babcock--Leighton/SFT theory.

Version 2 modifications
-----------------------
This version implements the corrections suggested after the first run:

1. The main response variable is now the unsigned axial dipole |D(T)|,
   not only the polar-cap field.

2. All main response plots include normalized curves:

       P_hat(T) = P(T) / P(T_ref)

   where T_ref = 1 is the reference solar-like cycle amplitude.

3. Latitude quenching is implemented as a dipole-efficiency reduction,
   not merely as placing flux closer to the polar caps. This prevents the
   artificial increase of the polar-cap average seen in the first version.

4. The SFT equation is solved on a mu = sin(lambda) grid. This avoids the
   numerical amplification near the poles caused by 1/cos(lambda) factors.

5. The stability map explores a wider parameter range and stronger dynamo
   gain so that stable, unstable/modulated, and no-fixed-point regions can
   appear.

Theoretical reduced model
-------------------------
The reduced model is

    P(T) = a T Q(T)

with

    Q(T) = Q_TQ(T) Q_LQ(T) Q_I(T),

where

    Q_TQ(T) = 1 / (1 + b_TQ T^2),
    Q_I(T)  = 1 / (1 + b_I T^2),
    Q_LQ(T) = exp(-[((lambda0 + b_LQ T^2)^2 - lambda0^2)/(2 lambda_R^2)]).

Outputs are saved to:

    ./figures_nonlinear_response_v2/

Author: Mohammed Talafha / ChatGPT draft
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Parameters
# ============================================================

@dataclass
class SFTParams:
    """Physical and numerical parameters for the 1D SFT experiment."""

    # Solar radius [cm]
    R_sun: float = 6.96e10

    # Surface diffusivity [km^2/s]
    eta_km2_s: float = 500.0

    # Meridional-flow amplitude [m/s]
    u0_m_s: float = 12.0

    # Decay timescale [yr]. Use np.inf for no decay.
    tau_yr: float = np.inf

    # Simulation length [yr]
    cycle_years: float = 11.0

    # mu = sin(latitude) grid
    # Number of finite-volume cells in mu. 181 is enough for the controlled
    # experiment and makes the explicit diffusion timestep stable and fast.
    nmu: int = 181
    mu_min: float = -1.0
    mu_max: float = 1.0

    # Time step [days]. The diffusion term is solved implicitly, so a 1-day
    # timestep is stable for the controlled experiment.
    dt_days: float = 1.0

    # Source parameters
    source_width_deg: float = 5.0
    source_time_width_yr: float = 2.2
    source_peak_time_yr: float = 4.5
    base_lat_deg: float = 15.0
    lat_drift_deg: float = 12.0

    # BMR-pair latitudinal separation proxy [deg]
    polarity_sep_deg: float = 5.0

    # Source normalization. Arbitrary because this is a normalized experiment.
    source_amp: float = 1.0

    # Nonlinear parameters for the reference run
    b_TQ: float = 0.35
    b_LQ: float = 4.0      # degrees per T^2 in the reduced closure
    b_I: float = 0.35

    # Latitude-effectivity scale for reduced theory [deg]
    lambda_R_deg: float = 20.0

    # Inflow amplitude [m/s] for T=1, scaled as T^2.
    # The first stable run used 5 m/s, which was intentionally strong but
    # caused dipole reversal in the inflow/combined cases. For a clean
    # response-theory validation, use a moderate value. Stronger values can be
    # explored later as an over-quenching experiment.
    inflow_amp_m_s: float = 2.0
    inflow_width_deg: float = 12.0

    # Polar cap threshold [deg]
    polar_cap_deg: float = 60.0

    # Reference amplitude for normalized plots
    T_ref: float = 1.0

    # Latitude-quenching implementation.
    # "efficiency" means use Q_LQ as a dipole-efficiency reduction.
    # "shift" means physically shift the source belt poleward.
    # The default is "efficiency" because it tests the reduced theory cleanly.
    latitude_quenching_mode: str = "efficiency"

    # Inflow implementation.
    # "efficiency" multiplies the source by Q_I(T), giving a clean reduced-model
    # validation analogous to latitude quenching.
    # "physical" applies an explicit converging flow. This is useful for
    # exploratory tests, but it can introduce sign reversals and interaction
    # effects not captured by the simple algebraic Q_I closure.
    inflow_mode: str = "efficiency"


# ============================================================
# 2. Unit conversion and grids
# ============================================================

def make_grid(p: SFTParams) -> Dict[str, np.ndarray | float]:
    """Create a finite-volume mu = sin(latitude) grid.

    Important correction:
    ---------------------
    The previous version placed grid points exactly at mu = +/-1. This can
    make the finite-difference update fragile near the poles. Here we use
    cell centres and keep the poles as cell interfaces. This is the standard
    finite-volume choice and avoids NaNs in the axial-dipole runs.
    """
    mu_faces = np.linspace(p.mu_min, p.mu_max, p.nmu + 1)
    dmu = mu_faces[1] - mu_faces[0]
    mu = 0.5 * (mu_faces[:-1] + mu_faces[1:])

    lat = np.arcsin(np.clip(mu, -1.0, 1.0))
    lat_deg = np.rad2deg(lat)
    coslat = np.sqrt(np.maximum(1.0 - mu**2, 0.0))
    cos_faces = np.sqrt(np.maximum(1.0 - mu_faces**2, 0.0))

    t_yr = np.arange(0.0, p.cycle_years + p.dt_days / 365.25, p.dt_days / 365.25)
    dt = p.dt_days * 86400.0

    eta = p.eta_km2_s * 1.0e10  # km^2/s -> cm^2/s
    u0 = p.u0_m_s * 100.0       # m/s -> cm/s

    if np.isfinite(p.tau_yr):
        tau = p.tau_yr * 365.25 * 86400.0
    else:
        tau = np.inf

    # Coefficients for the implicit diffusion solve:
    # (I - dt L_diff) B_new = B_star.
    gamma = dt * eta / p.R_sun**2 / dmu**2
    A_faces = 1.0 - mu_faces**2
    lower = np.zeros(p.nmu - 1, dtype=float)
    diag = np.ones(p.nmu, dtype=float)
    upper = np.zeros(p.nmu - 1, dtype=float)

    for i in range(p.nmu):
        A_left = A_faces[i]
        A_right = A_faces[i + 1]
        diag[i] = 1.0 + gamma * (A_left + A_right)
        if i > 0:
            lower[i - 1] = -gamma * A_left
        if i < p.nmu - 1:
            upper[i] = -gamma * A_right

    return {
        "mu": mu,
        "mu_faces": mu_faces,
        "dmu": dmu,
        "lat": lat,
        "lat_deg": lat_deg,
        "coslat": coslat,
        "cos_faces": cos_faces,
        "t_yr": t_yr,
        "dt": dt,
        "eta": eta,
        "u0": u0,
        "tau": tau,
        "diff_lower": lower,
        "diff_diag": diag,
        "diff_upper": upper,
    }


# ============================================================
# 3. Transport model on mu = sin(lambda)
# ============================================================

def meridional_flow(lat: np.ndarray, u0: float) -> np.ndarray:
    """
    Simple poleward meridional flow.

    Latitude lambda is positive northward. A simple antisymmetric profile is:

        u(lambda) = u0 sin(2 lambda),

    which is poleward in both hemispheres for |lambda| < 90 deg.
    """
    return u0 * np.sin(2.0 * lat)


def activity_inflow(lat_deg: np.ndarray, t: float, T: float, p: SFTParams) -> np.ndarray:
    """
    Idealized converging inflow toward the two activity belts.

    The inflow amplitude scales as T^2. Positive velocity means northward.
    This is a simplified transport feedback, not a full active-region inflow model.
    """
    lat0 = activity_belt_latitude(t, T, p, latitude_quenching=False)
    width = p.inflow_width_deg
    amp = p.inflow_amp_m_s * 100.0 * T**2

    gN = np.exp(-0.5 * ((lat_deg - lat0) / width) ** 2)
    gS = np.exp(-0.5 * ((lat_deg + lat0) / width) ** 2)

    inflow_N = amp * (lat0 - lat_deg) / width * gN
    inflow_S = amp * (-lat0 - lat_deg) / width * gS

    return inflow_N + inflow_S


def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm."""
    n = len(diag)
    a = lower.copy()
    b = diag.copy()
    c = upper.copy()
    d = rhs.copy()

    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    x = np.empty(n, dtype=float)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def implicit_diffusion_step(B_star: np.ndarray, grid: Dict[str, np.ndarray | float]) -> np.ndarray:
    """Apply one implicit diffusion step."""
    return solve_tridiagonal(
        grid["diff_lower"],
        grid["diff_diag"],
        grid["diff_upper"],
        B_star,
    )


def sft_rhs_mu(
    B: np.ndarray,
    t: float,
    T: float,
    p: SFTParams,
    grid: Dict[str, np.ndarray | float],
    source: np.ndarray,
    use_inflow: bool,
) -> np.ndarray:
    """
    Explicit part of the finite-volume SFT update in mu = sin(latitude).

    The diffusion term is NOT included here. It is solved implicitly by
    implicit_diffusion_step(). This operator splitting avoids the very small
    explicit diffusion timestep that caused overflow in the previous run.

    Explicit part:

        dB/dt = - 1/R d/dmu [u B sqrt(1 - mu^2)] + S - B/tau.
    """
    R = p.R_sun
    u0 = float(grid["u0"])
    tau = float(grid["tau"])
    dmu = float(grid["dmu"])

    mu = grid["mu"]
    mu_faces = grid["mu_faces"]
    lat = grid["lat"]
    lat_deg = grid["lat_deg"]
    cos_faces = grid["cos_faces"]

    u = meridional_flow(lat, u0)
    if use_inflow:
        u = u + activity_inflow(lat_deg, t, T, p)

    u_faces = np.interp(mu_faces, mu, u, left=0.0, right=0.0)

    B_up = np.zeros_like(mu_faces)
    for j in range(1, len(mu_faces) - 1):
        if u_faces[j] >= 0.0:
            B_up[j] = B[j - 1]
        else:
            B_up[j] = B[j]

    F_adv = u_faces * B_up * cos_faces
    F_adv[0] = 0.0
    F_adv[-1] = 0.0
    adv = -(F_adv[1:] - F_adv[:-1]) / (R * dmu)

    if np.isfinite(tau):
        decay = -B / tau
    else:
        decay = 0.0

    rhs = adv + source + decay

    if not np.all(np.isfinite(rhs)):
        raise FloatingPointError(
            f"Non-finite explicit RHS detected at t={t:.4f} yr, T={T:.3f}. "
            "Check transport/source parameters."
        )

    return rhs


# ============================================================
# 4. Source model
# ============================================================

def cycle_envelope(t: float, p: SFTParams) -> float:
    """Smooth source envelope over the cycle."""
    return np.exp(-0.5 * ((t - p.source_peak_time_yr) / p.source_time_width_yr) ** 2)


def activity_belt_latitude(t: float, T: float, p: SFTParams, latitude_quenching: bool) -> float:
    """
    Time-dependent emergence latitude.

    At the beginning of the cycle, emergence starts at higher latitude and drifts equatorward.
    If p.latitude_quenching_mode == "shift", latitude quenching shifts the belt poleward.
    If p.latitude_quenching_mode == "efficiency", the belt is not shifted; instead, the
    reduced model's Q_LQ factor is applied to the source efficiency.
    """
    phase = np.clip(t / p.cycle_years, 0.0, 1.0)
    base = p.base_lat_deg + p.lat_drift_deg * (1.0 - phase)

    if latitude_quenching and p.latitude_quenching_mode == "shift":
        base += p.b_LQ * T**2

    return base


def tilt_factor(T: float, p: SFTParams, tilt_quenching: bool) -> float:
    """Tilt-quenching factor."""
    if not tilt_quenching:
        return 1.0
    return 1.0 / (1.0 + p.b_TQ * T**2)


def q_latitude_efficiency(T: float, p: SFTParams) -> float:
    """Reduced latitude-quenching efficiency factor Q_LQ(T)."""
    lam0 = p.base_lat_deg
    lamR = p.lambda_R_deg
    lamT = lam0 + p.b_LQ * T**2
    return float(np.exp(-((lamT**2 - lam0**2) / (2.0 * lamR**2))))


def bipolar_ring_source(
    lat_deg: np.ndarray,
    t: float,
    T: float,
    p: SFTParams,
    tilt_quenching: bool,
    latitude_quenching: bool,
    inflow_quenching: bool = False,
) -> np.ndarray:
    """
    Idealized axisymmetric BMR source.

    We use two opposite-polarity Gaussian rings in each hemisphere.
    The polarity separation is reduced through a tilt-like factor.

    Latitude quenching is treated in two possible ways:

    1. efficiency mode, default:
       The source geometry is not shifted poleward. Instead, the source amplitude
       is multiplied by Q_LQ(T). This represents the reduced dipole efficiency
       of high-latitude emergence and cleanly tests the reduced theory.

    2. shift mode:
       The activity belt is physically shifted poleward. This can be useful for
       exploratory tests, but in an axisymmetric source it can artificially
       increase the polar-cap average.
    """
    env = cycle_envelope(t, p)
    lat0 = activity_belt_latitude(t, T, p, latitude_quenching=latitude_quenching)
    width = p.source_width_deg

    tf = tilt_factor(T, p, tilt_quenching=tilt_quenching)
    sep = p.polarity_sep_deg * tf

    # Northern hemisphere: following polarity poleward, leading polarity equatorward.
    # Southern hemisphere: antisymmetric sign.
    N_follow = np.exp(-0.5 * ((lat_deg - (lat0 + sep / 2.0)) / width) ** 2)
    N_lead   = np.exp(-0.5 * ((lat_deg - (lat0 - sep / 2.0)) / width) ** 2)
    S_follow = np.exp(-0.5 * ((lat_deg - (-lat0 - sep / 2.0)) / width) ** 2)
    S_lead   = np.exp(-0.5 * ((lat_deg - (-lat0 + sep / 2.0)) / width) ** 2)

    source = (N_follow - N_lead) - (S_follow - S_lead)

    amp = p.source_amp * T * env

    if latitude_quenching and p.latitude_quenching_mode == "efficiency":
        amp *= q_latitude_efficiency(T, p)

    # For the clean response-theory validation, inflow quenching can be treated
    # as a transport-efficiency factor Q_I(T). The explicit physical inflow mode
    # remains available through sft_rhs_mu when p.inflow_mode == "physical".
    if inflow_quenching and p.inflow_mode == "efficiency":
        amp *= float(Q_I(T, p))

    source *= amp

    # Convert arbitrary source per year to per second.
    source /= 365.25 * 86400.0

    # Remove monopole. On a mu grid, equal dmu gives the area weighting directly.
    source = source - np.mean(source)

    return source


# ============================================================
# 5. Reduced theory factors
# ============================================================

def Q_TQ(T: np.ndarray | float, p: SFTParams) -> np.ndarray | float:
    T = np.asarray(T, dtype=float)
    return 1.0 / (1.0 + p.b_TQ * T**2)


def Q_I(T: np.ndarray | float, p: SFTParams) -> np.ndarray | float:
    T = np.asarray(T, dtype=float)
    return 1.0 / (1.0 + p.b_I * T**2)


def Q_LQ(T: np.ndarray | float, p: SFTParams) -> np.ndarray | float:
    T = np.asarray(T, dtype=float)
    lam0 = p.base_lat_deg
    lamR = p.lambda_R_deg
    lamT = lam0 + p.b_LQ * T**2
    return np.exp(-((lamT**2 - lam0**2) / (2.0 * lamR**2)))


def Q_theory(T: np.ndarray | float, p: SFTParams, case: str) -> np.ndarray | float:
    """Theoretical nonlinear efficiency factor for each case."""
    T = np.asarray(T, dtype=float)
    q = np.ones_like(T, dtype=float)

    if case in ["tilt", "combined"]:
        q *= Q_TQ(T, p)
    if case in ["latitude", "combined"]:
        q *= Q_LQ(T, p)
    if case in ["inflow", "combined"]:
        q *= Q_I(T, p)

    return q


def theory_normalized(T: np.ndarray, p: SFTParams, case: str) -> np.ndarray:
    """Normalized theoretical curve T Q(T) / [T_ref Q(T_ref)]."""
    T = np.asarray(T, dtype=float)
    tref = p.T_ref
    numerator = T * Q_theory(T, p, case)
    denominator = tref * Q_theory(tref, p, case)
    return numerator / denominator


# ============================================================
# 6. Diagnostics
# ============================================================

def polar_field_proxy(B: np.ndarray, mu: np.ndarray, cap_deg: float = 60.0) -> float:
    """Unsigned average polar-cap field proxy."""
    mu_cap = np.sin(np.deg2rad(cap_deg))
    north = mu >= mu_cap
    south = mu <= -mu_cap
    PN = np.mean(B[north])
    PS = np.mean(B[south])
    return 0.5 * (abs(PN) + abs(PS))


def axial_dipole(B: np.ndarray, mu: np.ndarray) -> float:
    """
    Axisymmetric axial dipole proxy on a mu grid:

        D = 3/2 int_{-1}^{1} B(mu) mu dmu.

    The normalization is suitable for comparison between runs.
    """
    return 1.5 * np.trapz(B * mu, mu)


def nonlinear_response(T_values: np.ndarray, P_values: np.ndarray) -> np.ndarray:
    """Numerical derivative dP/dT."""
    return np.gradient(P_values, T_values, edge_order=2)


def normalize_at_Tref(T: np.ndarray, Y: np.ndarray, T_ref: float = 1.0) -> np.ndarray:
    """Normalize Y by interpolation at T_ref."""
    yref = np.interp(T_ref, T, Y)
    if abs(yref) < 1e-30:
        return np.full_like(Y, np.nan, dtype=float)
    return Y / yref


# ============================================================
# 7. Simulation driver
# ============================================================

def run_sft_case(T: float, case: str, p: SFTParams) -> Tuple[float, float, float, np.ndarray]:
    """Run one SFT experiment for a given T and nonlinearity case."""
    grid = make_grid(p)
    mu = grid["mu"]
    lat_deg = grid["lat_deg"]
    t_yr = grid["t_yr"]
    dt = float(grid["dt"])

    B = np.zeros_like(mu, dtype=float)

    use_tilt = case in ["tilt", "combined"]
    use_latitude = case in ["latitude", "combined"]
    use_inflow = (case in ["inflow", "combined"]) and (p.inflow_mode == "physical")

    for t in t_yr:
        S = bipolar_ring_source(
            lat_deg=lat_deg,
            t=float(t),
            T=T,
            p=p,
            tilt_quenching=use_tilt,
            latitude_quenching=use_latitude,
            inflow_quenching=(case in ["inflow", "combined"]),
        )

        rhs = sft_rhs_mu(
            B=B,
            t=float(t),
            T=T,
            p=p,
            grid=grid,
            source=S,
            use_inflow=use_inflow,
        )

        B_star = B + dt * rhs

        # Apply implicit diffusion. This is the key correction that removes
        # the explicit diffusion overflow.
        B = implicit_diffusion_step(B_star, grid)

        if not np.all(np.isfinite(B)):
            raise FloatingPointError(
                f"Non-finite magnetic field detected at t={t:.4f} yr, T={T:.3f}, case={case}. "
                "Try reducing dt_days or inflow_amp_m_s."
            )

        # Remove numerical monopole on the mu grid.
        B = B - np.mean(B)

    P_polar = polar_field_proxy(B, mu, p.polar_cap_deg)
    D_signed = axial_dipole(B, mu)

    # Use the signed axial dipole as the main response. The sign contains
    # physically important information: if a nonlinear feedback reverses the
    # sign of the dipole, that is over-quenching, not merely a small positive
    # response. Unsigned |D| is still saved separately.
    P_main = D_signed
    P_abs = abs(D_signed)

    return P_main, D_signed, P_abs, P_polar, B


def check_explicit_stability(p: SFTParams) -> None:
    """Print a numerical setup summary.

    Diffusion is solved implicitly in this version, so the explicit diffusion
    CFL limit no longer restricts dt_days. We still print the old explicit
    limit as a diagnostic only.
    """
    grid = make_grid(p)
    dmu = float(grid["dmu"])
    dt = float(grid["dt"])
    eta = float(grid["eta"])
    R = p.R_sun
    dt_lim = 0.45 * dmu**2 * R**2 / eta

    print("Numerical setup check")
    print(f"  nmu        = {p.nmu}")
    print(f"  dmu        = {dmu:.4e}")
    print(f"  dt         = {dt/86400.0:.4f} days")
    print(f"  explicit diffusion limit would be ~{dt_lim/86400.0:.4f} days")
    print("  diffusion  = implicit")



def run_experiment(p: SFTParams, T_values: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """Run all nonlinear response experiments."""
    check_explicit_stability(p)
    cases = ["linear", "tilt", "latitude", "inflow", "combined"]
    results: Dict[str, Dict[str, List[float] | List[np.ndarray]]] = {}

    for case in cases:
        print(f"\nRunning case: {case}")
        results[case] = {"T": [], "P": [], "D": [], "P_polar": [], "B_final": []}

        for T in T_values:
            print(f"  T = {T:.2f}")
            P, D, P_abs, P_polar, B = run_sft_case(float(T), case, p)
            results[case]["T"].append(float(T))
            results[case]["P"].append(float(P))
            results[case]["D"].append(float(D))
            results[case].setdefault("P_abs", []).append(float(P_abs))
            results[case]["P_polar"].append(float(P_polar))
            results[case]["B_final"].append(B)

    final_results: Dict[str, Dict[str, np.ndarray]] = {}
    for case in cases:
        T = np.array(results[case]["T"], dtype=float)
        Pvals = np.array(results[case]["P"], dtype=float)
        Dvals = np.array(results[case]["D"], dtype=float)
        Pabs = np.array(results[case]["P_abs"], dtype=float)
        Ppolar = np.array(results[case]["P_polar"], dtype=float)
        Rvals = nonlinear_response(T, Pvals)
        Phat = normalize_at_Tref(T, Pvals, p.T_ref)
        Rhat = nonlinear_response(T, Phat)

        final_results[case] = {
            "T": T,
            "P": Pvals,
            "D": Dvals,
            "P_abs": Pabs,
            "P_polar": Ppolar,
            "P_hat": Phat,
            "R_NL": Rvals,
            "R_hat": Rhat,
            "B_final": np.array(results[case]["B_final"]),
        }

    return final_results


# ============================================================
# 8. Plotting
# ============================================================

def plot_results(results: Dict[str, Dict[str, np.ndarray]], p: SFTParams, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    cases = ["linear", "tilt", "latitude", "inflow", "combined"]
    labels = {
        "linear": "Linear",
        "tilt": "Tilt quenching",
        "latitude": "Latitude quenching",
        "inflow": "Inflow quenching",
        "combined": "Combined",
    }

    T = results["linear"]["T"]
    Tfine = np.linspace(T.min(), T.max(), 500)

    # --------------------------------------------------------
    # Figure 1: raw signed main response D(T)
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for case in cases:
        plt.plot(results[case]["T"], results[case]["P"], "o-", label=labels[case])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Main response $P(T)=D(T)$")
    plt.title(r"Signed axial-dipole response")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig1_main_signed_axial_dipole_response_raw.png"), dpi=200)
    plt.close()

    # Also save the unsigned response as a secondary diagnostic.
    plt.figure(figsize=(8, 5))
    for case in cases:
        plt.plot(results[case]["T"], results[case]["P_abs"], "o-", label=labels[case])
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Secondary response $|D(T)|$")
    plt.title(r"Unsigned axial-dipole response")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig1b_unsigned_axial_dipole_response.png"), dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Figure 2: normalized P_hat(T) with normalized theory
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for case in cases:
        plt.plot(results[case]["T"], results[case]["P_hat"], "o-", label=f"SFT: {labels[case]}")
    for case in cases:
        plt.plot(Tfine, theory_normalized(Tfine, p, case), "--", alpha=0.8, label=f"Theory: {labels[case]}")
    plt.axhline(1.0, linewidth=1)
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Normalized response $\hat{P}(T)=P(T)/P(1)$")
    plt.title(r"Normalized nonlinear response")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig2_normalized_response_vs_theory.png"), dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Figure 3: nonlinear response dP_hat/dT
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for case in cases:
        plt.plot(results[case]["T"], results[case]["R_hat"], "o-", label=labels[case])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"$d\hat{P}/dT$")
    plt.title(r"Normalized nonlinear response function")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig3_normalized_nonlinear_response.png"), dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Figure 4: signed axial dipole D(T)
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for case in cases:
        plt.plot(results[case]["T"], results[case]["D"], "o-", label=labels[case])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Signed axial dipole $D(T)$")
    plt.title(r"Signed axial dipole response")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig4_signed_axial_dipole_response.png"), dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Figure 5: polar-cap response retained as secondary diagnostic
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for case in cases:
        plt.plot(results[case]["T"], results[case]["P_polar"], "o-", label=labels[case])
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Polar-cap proxy")
    plt.title(r"Polar-cap field response, secondary diagnostic")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig5_polar_cap_response_secondary.png"), dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Figure 6: reduced nonlinear efficiency factors Q(T)
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for case in cases:
        q = Q_theory(Tfine, p, case)
        plt.plot(Tfine, q, label=labels[case])
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Nonlinear efficiency $Q(T)$")
    plt.title(r"Reduced nonlinear efficiency factors")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig6_Q_efficiency.png"), dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Figure 7: effective dynamo gain
    # --------------------------------------------------------
    D0 = 1.8
    plt.figure(figsize=(8, 5))
    for case in cases:
        Deff = D0 * Q_theory(Tfine, p, case)
        plt.plot(Tfine, Deff, label=labels[case])
    plt.axhline(1.0, linewidth=1, linestyle="--")
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Effective gain $D_{\rm eff}=D_0Q(T)$")
    plt.title(r"Finite-amplitude saturation condition")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig7_effective_gain.png"), dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Figure 8: final B(mu) profiles for combined case
    # --------------------------------------------------------
    grid = make_grid(p)
    lat_deg = grid["lat_deg"]
    selected = [0, len(T)//2, -1]
    plt.figure(figsize=(8, 5))
    for idx in selected:
        Tval = results["combined"]["T"][idx]
        B = results["combined"]["B_final"][idx]
        plt.plot(lat_deg, B, label=fr"$T={Tval:.2f}$")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Latitude [deg]")
    plt.ylabel("Final surface field [arb. units]")
    plt.title("Final latitude profiles: combined nonlinear case")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig8_final_profiles_combined.png"), dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Save numerical table
    # --------------------------------------------------------
    table_path = os.path.join(outdir, "nonlinear_response_results_v2.csv")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("case,T,P_signed_main,D_signed,P_abs_dipole,P_polar,P_hat,R_NL,R_hat")
        for case in cases:
            for Ti, Pi, Di, Pai, Ppi, Phi, Ri, Rhi in zip(
                results[case]["T"],
                results[case]["P"],
                results[case]["D"],
                results[case]["P_abs"],
                results[case]["P_polar"],
                results[case]["P_hat"],
                results[case]["R_NL"],
                results[case]["R_hat"],
            ):
                f.write(f"{case},{Ti:.6f},{Pi:.10e},{Di:.10e},{Pai:.10e},{Ppi:.10e},{Phi:.10e},{Ri:.10e},{Rhi:.10e}")

    print(f"\nSaved figures and table to: {outdir}")


# ============================================================
# 9. Stability map from reduced theory
# ============================================================

def reduced_Q_full(
    T: np.ndarray,
    b_TQ: float,
    b_LQ: float,
    b_I: float,
    lambda0: float,
    lambdaR: float,
) -> np.ndarray:
    """Full Q(T) for arbitrary nonlinear parameters."""
    QTQ = 1.0 / (1.0 + b_TQ * T**2)
    QI = 1.0 / (1.0 + b_I * T**2)
    lamT = lambda0 + b_LQ * T**2
    QLQ = np.exp(-((lamT**2 - lambda0**2) / (2.0 * lambdaR**2)))
    return QTQ * QI * QLQ


def make_stability_map(p: SFTParams, outdir: str) -> None:
    """
    Reduced-theory stability map in the (b_TQ, b_LQ) plane.

    Classification:
        0: no nonzero fixed point found in the scanned T interval
        1: stable fixed point, |M'| < 1
        2: unstable/modulated, |M'| >= 1

    The scan is intentionally broad to show the qualitative structure.
    """
    os.makedirs(outdir, exist_ok=True)

    D0 = 3.0
    bI_fixed = 0.8
    bTQ_vals = np.linspace(0.0, 5.0, 180)
    bLQ_vals = np.linspace(0.0, 20.0, 180)
    Tgrid = np.linspace(0.02, 6.0, 4000)

    state = np.zeros((len(bLQ_vals), len(bTQ_vals)))
    Tstar_map = np.full_like(state, np.nan, dtype=float)
    Mprime_map = np.full_like(state, np.nan, dtype=float)

    for i, bLQ in enumerate(bLQ_vals):
        for j, bTQ in enumerate(bTQ_vals):
            Q = reduced_Q_full(
                Tgrid,
                b_TQ=bTQ,
                b_LQ=bLQ,
                b_I=bI_fixed,
                lambda0=p.base_lat_deg,
                lambdaR=p.lambda_R_deg,
            )
            Deff = D0 * Q
            y = Deff - 1.0

            # Need a crossing from above to below for finite saturation.
            crossing = np.where((y[:-1] >= 0.0) & (y[1:] <= 0.0))[0]

            if len(crossing) == 0:
                state[i, j] = 0
                continue

            k = crossing[0]
            Tstar = Tgrid[k]

            M = D0 * Tgrid * Q
            dM = np.gradient(M, Tgrid, edge_order=2)
            Mprime = dM[k]

            Tstar_map[i, j] = Tstar
            Mprime_map[i, j] = Mprime

            if abs(Mprime) < 1.0:
                state[i, j] = 1
            else:
                state[i, j] = 2

    # Discrete stability classification map.
    plt.figure(figsize=(7, 5))
    plt.imshow(
        state,
        origin="lower",
        aspect="auto",
        extent=[bTQ_vals.min(), bTQ_vals.max(), bLQ_vals.min(), bLQ_vals.max()],
        interpolation="nearest",
    )
    plt.xlabel(r"Tilt-quenching strength $b_{\rm TQ}$")
    plt.ylabel(r"Latitude-quenching strength $b_{\rm LQ}$")
    plt.title(r"Reduced-theory stability map")
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["No fixed point", "Stable", "Unstable/modulated"])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig9_stability_map_classification.png"), dpi=200)
    plt.close()

    # Tstar map.
    plt.figure(figsize=(7, 5))
    plt.imshow(
        Tstar_map,
        origin="lower",
        aspect="auto",
        extent=[bTQ_vals.min(), bTQ_vals.max(), bLQ_vals.min(), bLQ_vals.max()],
        interpolation="nearest",
    )
    plt.xlabel(r"Tilt-quenching strength $b_{\rm TQ}$")
    plt.ylabel(r"Latitude-quenching strength $b_{\rm LQ}$")
    plt.title(r"Saturated amplitude $T_*$")
    plt.colorbar(label=r"$T_*$")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig10_Tstar_map.png"), dpi=200)
    plt.close()

    # Mprime map.
    plt.figure(figsize=(7, 5))
    plt.imshow(
        Mprime_map,
        origin="lower",
        aspect="auto",
        extent=[bTQ_vals.min(), bTQ_vals.max(), bLQ_vals.min(), bLQ_vals.max()],
        interpolation="nearest",
    )
    plt.xlabel(r"Tilt-quenching strength $b_{\rm TQ}$")
    plt.ylabel(r"Latitude-quenching strength $b_{\rm LQ}$")
    plt.title(r"Fixed-point derivative $\mathcal{M}'(T_*)$")
    plt.colorbar(label=r"$\mathcal{M}'(T_*)$")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig11_Mprime_map.png"), dpi=200)
    plt.close()

    # Save stability table.
    table_path = os.path.join(outdir, "stability_map_v2.csv")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("b_TQ,b_LQ,state,Tstar,Mprime\n")
        for i, bLQ in enumerate(bLQ_vals):
            for j, bTQ in enumerate(bTQ_vals):
                f.write(f"{bTQ:.8f},{bLQ:.8f},{int(state[i,j])},{Tstar_map[i,j]:.10e},{Mprime_map[i,j]:.10e}\n")


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":
    params = SFTParams()

    # Cycle amplitudes to test. Include exactly T=1 for clean normalization.
    T_values = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.2, 1.4,
                         1.6, 1.8, 2.0, 2.2, 2.5], dtype=float)

    outdir = "figures_nonlinear_response_v2"

    results = run_experiment(params, T_values)
    plot_results(results, params, outdir)
    make_stability_map(params, outdir)

    print("\nDone.")
    print("Main outputs:")
    print("  fig1_main_signed_axial_dipole_response_raw.png")
    print("  fig1b_unsigned_axial_dipole_response.png")
    print("  fig2_normalized_response_vs_theory.png")
    print("  fig3_normalized_nonlinear_response.png")
    print("  fig4_signed_axial_dipole_response.png")
    print("  fig5_polar_cap_response_secondary.png")
    print("  fig6_Q_efficiency.png")
    print("  fig7_effective_gain.png")
    print("  fig8_final_profiles_combined.png")
    print("  fig9_stability_map_classification.png")
    print("  fig10_Tstar_map.png")
    print("  fig11_Mprime_map.png")
    print("  nonlinear_response_results_v2.csv")
    print("  stability_map_v2.csv")
