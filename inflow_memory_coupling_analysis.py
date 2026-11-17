#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inflow-memory coupling: does amplitude-dependent transport of the surviving
(inherited) flux change the inferred quenching or the onset of modulation?
(see docs/inflow_memory_coupling.md).

With transport prescribed by the cycle amplitude T (activity-dependent
inflows), the SFT is still linear in B for a given T, so the end-of-cycle
field obeys the exact split

    B_end = s b(T) + G(T) B_start,      D_end = s P(T) + w^T G(T) B_start,

where b(T) is the field built by the new source from a clean slate (with the
inflows of the same cycle acting on it), P(T) = w^T b(T) its dipole, and G(T)
the one-cycle propagator, which now depends on T through the inflows. The
reduced multi-cycle map collapses the second term to r D_start with a
constant r. This script measures, for the reference SFT:

1. remnant survival r as a function of (i) the amplitude of the cycle whose
   inflows act on the inherited field and (ii) the amplitude of the cycle
   that left that field behind;
2. an adjoint budget of dD/dT into source change, transport of new flux and
   transport of inherited flux, checked against finite differences;
3. the stability of the COMPLETE field evolution (leading eigenvalue of the
   full-state cycle-to-cycle Jacobian), with and without the amplitude
   dependence of inherited-flux transport, against the scalar maps and
   against a two-variable "settled-dipole" map that also carries the dipole
   of the field after one further cycle of transport, with a nonlinear
   multi-cycle check of the predicted thresholds;
4. the error of scalar remnant subtraction, Delta D_n = D_n - rho D_{n-1},
   along a prescribed multi-cycle sequence, with rho = exp(-P/tau) (the form
   used by Talafha, Petrovay and Opitz 2025) and with the best constant r,
   and its effect on the inferred quenching strength;
5. controls: inflow strength, inflow timing (constant in time, as in the
   reference solver, or following the activity envelope), inflow centre
   (fixed belt, as in the reference solver, or following a poleward-shifted
   belt), finite decay, and grid and time-step refinement.

The reference solver (nonlinear_response_sft_experiment.py) is not modified.
A vectorised copy of its update is used here and validated against it.

Outputs are saved to ./figures_inflow_memory/.
"""

from __future__ import annotations

import os

# One BLAS thread per process: the analysis parallelises over processes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402

import numpy as np  # noqa: E402

import nonlinear_response_sft_experiment as nre  # noqa: E402
import mathematical_framework_analysis as mfa  # noqa: E402


OUTDIR = "figures_inflow_memory"
SECONDS_PER_YEAR = 365.25 * 86400.0


# ============================================================
# 1. Model configuration and vectorised solver
# ============================================================

@dataclass(frozen=True)
class Coupling:
    """Nonlinearities and inflow prescription for one experiment.

    inflow_amp_m_s : inflow speed scale at T = 1 (0 = no physical inflow).
    inflow_timing  : "constant" (reference solver: full strength all cycle)
                     or "envelope" (speed follows the activity envelope).
    inflow_centre  : "fixed" (reference solver: unshifted belt) or
                     "shifted" (follows the belt displaced by b_LQ T^2).
    lat_mode       : "none", "efficiency" (source factor, reference) or
                     "shift" (belt displaced poleward by b_LQ T^2).
    q_inflow_eff   : multiply the source by Q_I(T) (efficiency-mode inflow).
    """

    inflow_amp_m_s: float = 2.0
    inflow_timing: str = "constant"
    inflow_centre: str = "fixed"
    lat_mode: str = "efficiency"
    q_inflow_eff: bool = False


class Model:
    """Vectorised copy of the reference SFT update.

    Every column of B is an independent run; each column may have its own
    source amplitude T_src, sign and flow amplitude T_flow, all sharing the
    same phase step. The update is the reference solver's, exactly:
    upwind advection + source + decay explicitly, implicit diffusion, then
    removal of the monopole.
    """

    def __init__(self, p: nre.SFTParams, c: Coupling):
        self.p = p
        self.c = c
        g = nre.make_grid(p)
        self.grid = g
        self.mu = np.asarray(g["mu"])
        self.lat_deg = np.asarray(g["lat_deg"])
        self.lat_faces_deg = np.asarray(g["lat_faces_deg"])
        self.cos_faces = np.asarray(g["cos_faces"])
        self.t_yr = np.asarray(g["t_yr"])
        self.N = len(self.t_yr)
        self.dt = float(g["dt"])
        self.dmu = float(g["dmu"])
        self.tau = float(g["tau"])
        self.n = p.nmu
        M = (np.diag(np.asarray(g["diff_diag"]))
             + np.diag(np.asarray(g["diff_lower"]), -1)
             + np.diag(np.asarray(g["diff_upper"]), 1))
        self.Minv = np.linalg.inv(M)
        self.w = mfa.dipole_weight(p, g)
        self.u_mer = nre.meridional_flow(np.asarray(g["lat_faces"]), float(g["u0"]))
        phase = np.clip(self.t_yr / p.cycle_years, 0.0, 1.0)
        self.lam0 = p.base_lat_deg + p.lat_drift_deg * (1.0 - phase)
        self.env = np.array([nre.cycle_envelope(float(t), p) for t in self.t_yr])

    # ---------------- flow ----------------
    def inflow(self, k: int, T_flow: np.ndarray) -> np.ndarray:
        """Inflow at the faces, shape (n+1, m), for flow amplitudes T_flow."""
        p, c = self.p, self.c
        T_flow = np.atleast_1d(T_flow)
        if c.inflow_amp_m_s == 0.0:
            return np.zeros((self.n + 1, T_flow.size))
        lat0 = np.full(T_flow.size, self.lam0[k])
        if c.inflow_centre == "shifted" and c.lat_mode == "shift":
            lat0 = lat0 + p.b_LQ * T_flow**2
        amp = c.inflow_amp_m_s * 100.0 * T_flow**2
        if c.inflow_timing == "envelope":
            amp = amp * self.env[k]
        lat = self.lat_faces_deg[:, None]
        wdt = p.inflow_width_deg
        gN = np.exp(-0.5 * ((lat - lat0) / wdt) ** 2)
        gS = np.exp(-0.5 * ((lat + lat0) / wdt) ** 2)
        return amp * ((lat0 - lat) / wdt * gN + (-lat0 - lat) / wdt * gS)

    def dinflow_dT(self, k: int, T_flow: np.ndarray) -> np.ndarray:
        """d(inflow)/dT_flow at the faces."""
        T_flow = np.atleast_1d(T_flow)
        if self.c.inflow_amp_m_s == 0.0:
            return np.zeros((self.n + 1, T_flow.size))
        if self.c.inflow_centre == "shifted" and self.c.lat_mode == "shift":
            h = 1e-6
            return (self.inflow(k, T_flow + h) - self.inflow(k, T_flow - h)) / (2 * h)
        return self.inflow(k, T_flow) * 2.0 / T_flow

    def velocity(self, k: int, T_flow: np.ndarray) -> np.ndarray:
        return self.u_mer[:, None] + self.inflow(k, T_flow)

    def advect(self, B: np.ndarray, u_val: np.ndarray, u_dir: np.ndarray) -> np.ndarray:
        """Upwind advection of B with face speed u_val, upwind side from u_dir."""
        Bup = np.where(u_dir[1:-1] >= 0.0, B[:-1], B[1:])
        F = np.zeros((self.n + 1, B.shape[1]))
        F[1:-1] = u_val[1:-1] * Bup * self.cos_faces[1:-1, None]
        return -(F[1:] - F[:-1]) / (self.p.R_sun * self.dmu)

    def advect_T(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Transpose of advect(., u, u) applied to v (for the adjoint)."""
        g = ((v[1:] - v[:-1]) / (self.p.R_sun * self.dmu)
             * u[1:-1] * self.cos_faces[1:-1, None])
        out = np.zeros_like(v)
        pos = u[1:-1] >= 0.0
        out[:-1] += np.where(pos, g, 0.0)
        out[1:] += np.where(pos, 0.0, g)
        return out

    # ---------------- source ----------------
    def source(self, k: int, T_src: np.ndarray, sign) -> np.ndarray:
        """Source at phase step k, shape (n, m) [field per second]."""
        p, c = self.p, self.c
        T_src = np.atleast_1d(T_src)
        lat0 = np.full(T_src.size, self.lam0[k])
        if c.lat_mode == "shift":
            lat0 = lat0 + p.b_LQ * T_src**2
        sep = p.polarity_sep_deg
        wd = p.source_width_deg
        lat = self.lat_deg[:, None]
        NF = np.exp(-0.5 * ((lat - (lat0 + sep / 2.0)) / wd) ** 2)
        NL = np.exp(-0.5 * ((lat - (lat0 - sep / 2.0)) / wd) ** 2)
        SF = np.exp(-0.5 * ((lat - (-lat0 - sep / 2.0)) / wd) ** 2)
        SL = np.exp(-0.5 * ((lat - (-lat0 + sep / 2.0)) / wd) ** 2)
        if p.balanced_polarities:
            phis = [r.mean(axis=0) for r in (NF, NL, SF, SL)]
            scale = sum(phis) / 4.0
            NF, NL, SF, SL = (scale * r / ph for r, ph in zip((NF, NL, SF, SL), phis))
        shape = (NF - NL) - (SF - SL)
        amp = p.source_amp * T_src * self.env[k]
        if c.lat_mode == "efficiency":
            delta = p.b_LQ * T_src**2
            amp = amp * np.exp(-((2.0 * self.lam0[k] * delta + delta**2)
                                 / (2.0 * p.lambda_R_deg**2)))
        if c.q_inflow_eff:
            amp = amp / (1.0 + p.b_I * T_src**2)
        S = shape * (amp * sign) / SECONDS_PER_YEAR
        return S - S.mean(axis=0)

    # ---------------- one cycle ----------------
    def cycle(self, B0: np.ndarray, T_src, sign, T_flow=None,
              record: bool = False):
        """Advance columns of B0 through one cycle.

        sign = 0 switches the source off for that column; T_flow defaults to
        T_src and sets the inflow amplitude.
        """
        B = np.array(B0, dtype=float, copy=True)
        if B.ndim == 1:
            B = B[:, None]
        m = B.shape[1]
        T_src = np.broadcast_to(np.asarray(T_src, dtype=float), (m,)).copy()
        sign = np.broadcast_to(np.asarray(sign, dtype=float), (m,)).copy()
        T_flow = T_src.copy() if T_flow is None else \
            np.broadcast_to(np.asarray(T_flow, dtype=float), (m,)).copy()
        src_on = bool(np.any(sign * T_src != 0.0))
        # Evaluate flows and source shapes once per distinct amplitude.
        Tf_u, Tf_inv = np.unique(T_flow, return_inverse=True)
        Ts_u, Ts_inv = np.unique(T_src, return_inverse=True)
        hist = np.empty((self.N, self.n, m)) if record else None
        for k in range(self.N):
            if record:
                hist[k] = B
            u = self.velocity(k, Tf_u)[:, Tf_inv]
            rhs = self.advect(B, u, u)
            if src_on:
                rhs = rhs + self.source(k, Ts_u, 1.0)[:, Ts_inv] * sign
            if np.isfinite(self.tau):
                rhs = rhs - B / self.tau
            B = self.Minv @ (B + self.dt * rhs)
            B = B - B.mean(axis=0)
        return (B, hist) if record else B

    def dipole(self, B: np.ndarray) -> np.ndarray:
        return self.w @ B

    # ---------------- building blocks of the cycle map ----------------
    def new_flux(self, T) -> np.ndarray:
        """b(T): end-of-cycle field from a clean slate, positive source sign."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return self.cycle(np.zeros((self.n, T.size)), T, 1.0)

    def propagator(self, T: float) -> np.ndarray:
        """G(T): one-cycle propagator with the inflows of a cycle of amplitude T."""
        return self.cycle(np.eye(self.n), 0.0, 0.0, T_flow=T)

    def periodic_state(self, T: float, G: np.ndarray | None = None,
                       b: np.ndarray | None = None) -> np.ndarray:
        """End-of-cycle field B* of the Hale-alternating state at constant T.

        B* solves (I + G) B* = b: a cycle of amplitude T and negative source
        sign maps B* onto -B*.
        """
        G = self.propagator(T) if G is None else G
        b = self.new_flux(T)[:, 0] if b is None else b
        return np.linalg.solve(np.eye(self.n) + G, b)


def linear_yield(p: nre.SFTParams) -> float:
    """a: dipole of a linear (unquenched, no-inflow) T = 1 cycle."""
    mdl = Model(p, Coupling(inflow_amp_m_s=0.0, lat_mode="none"))
    return float(mdl.dipole(mdl.new_flux(1.0))[0])


# ============================================================
# 2. Diagnostics
# ============================================================

def validate_against_reference(p: nre.SFTParams) -> float:
    """Max relative difference between Model and the reference solver."""
    worst = 0.0
    for case, pp, c in [
        ("linear", p, Coupling(inflow_amp_m_s=0.0, lat_mode="none")),
        ("inflow", replace(p, inflow_mode="physical", inflow_amp_m_s=2.0),
         Coupling(inflow_amp_m_s=2.0, lat_mode="none")),
        ("inflow", replace(p, inflow_mode="physical", inflow_amp_m_s=5.0),
         Coupling(inflow_amp_m_s=5.0, lat_mode="none")),
        ("latitude", p, Coupling(inflow_amp_m_s=0.0, lat_mode="efficiency")),
        ("latitude", replace(p, latitude_quenching_mode="shift"),
         Coupling(inflow_amp_m_s=0.0, lat_mode="shift")),
    ]:
        mdl = Model(pp, c)
        Ts = np.array([0.5, 1.0, 2.0])
        D_vec = mdl.dipole(mdl.new_flux(Ts))
        for T, Dv in zip(Ts, D_vec):
            D_ref = nre.run_sft_case(float(T), case, pp)[1]
            worst = max(worst, abs(Dv - D_ref) / abs(D_ref))
    return worst


def remnant_ratio(mdl: Model, T_flow: np.ndarray, B_start: np.ndarray) -> np.ndarray:
    """r = w^T G(T_flow) B_start / w^T B_start, for each T_flow (no source)."""
    T_flow = np.atleast_1d(T_flow)
    B0 = np.repeat(B_start[:, None], T_flow.size, axis=1)
    Bend = mdl.cycle(B0, 0.0, 0.0, T_flow=T_flow)
    return mdl.dipole(Bend) / float(mdl.w @ B_start)


def fit_rational_b(T: np.ndarray, P: np.ndarray, a: float) -> float:
    """Least-squares b in P = a T / (1 + b T^2), with a fixed (linear yield)."""
    y = a * T / P - 1.0          # = b T^2 for an exact rational response
    x = T**2
    return float(np.sum(x * y) / np.sum(x * x))


def adjoint_budget(mdl: Model, T0: float, B_start: np.ndarray, sign: float):
    """Adjoint split of dD_end/dT at amplitude T0 from initial field B_start.

    The backward recursion uses the time-dependent, inflow-modified
    operator, so the split is exact to first order:
        dD/dT = sum_k dt K_k . dS_k/dT                (source)
              + sum_k dt K_k . (dA_k/dT) B_new^k      (transport of new flux)
              + sum_k dt K_k . (dA_k/dT) B_old^k      (transport of inherited flux)
    Returns the three terms and the time-latitude densities of the last two.
    """
    n, N, dt = mdl.n, mdl.N, mdl.dt
    T = np.array([T0])
    _, h_new = mdl.cycle(np.zeros((n, 1)), T, sign, T_flow=T, record=True)
    _, h_old = mdl.cycle(B_start[:, None], 0.0, 0.0, T_flow=T, record=True)

    def project(v):
        return v - v.mean(axis=0)

    K = np.empty((N, n))
    lam = mdl.w[:, None]
    for k in range(N - 1, -1, -1):
        Kk = mdl.Minv @ project(lam)
        K[k] = Kk[:, 0]
        lam = Kk + dt * mdl.advect_T(Kk, mdl.velocity(k, T))
        if np.isfinite(mdl.tau):
            lam = lam - dt * Kk / mdl.tau
    h = 1e-4
    src_term = 0.0
    dens_new = np.empty((N, n))
    dens_old = np.empty((N, n))
    for k in range(N):
        dS = (mdl.source(k, T + h, sign) - mdl.source(k, T - h, sign)) / (2 * h)
        src_term += dt * float(K[k] @ dS[:, 0])
        u = mdl.velocity(k, T)
        du = mdl.dinflow_dT(k, T)
        dens_new[k] = dt * K[k] * mdl.advect(h_new[k], du, u)[:, 0]
        dens_old[k] = dt * K[k] * mdl.advect(h_old[k], du, u)[:, 0]
    return src_term, float(dens_new.sum()), float(dens_old.sum()), dens_new, dens_old


def finite_difference_budget(mdl: Model, T0: float, B_start: np.ndarray,
                             sign: float, h: float = 1e-3):
    """Same split by central differences of separate runs."""
    n = mdl.n
    Tpm = np.array([T0 + h, T0 - h])
    z = np.zeros((n, 2))
    Bs = mdl.cycle(z, Tpm, sign, T_flow=np.array([T0, T0]))
    Bn = mdl.cycle(z, np.array([T0, T0]), sign, T_flow=Tpm)
    Bo = mdl.cycle(np.repeat(B_start[:, None], 2, axis=1), 0.0, 0.0, T_flow=Tpm)
    d = lambda X: float(mdl.dipole(X)[0] - mdl.dipole(X)[1]) / (2 * h)
    return d(Bs), d(Bn), d(Bo)


def full_state_stability(mdl: Model, Tstar: float, a: float, h: float = 1e-4):
    """Fixed point and leading multiplier of the cycle map at amplitude T*.

    Complete field. B* solves (I+G)B* = b, D* = w^T B*, the feedback gain is
    k = T*/D* and D0 = k a. The perturbation map of the complete field is
    J = -G(T*) - k (c_old - c_new) w^T, with c_old = d(G B*)/dT and
    c_new = db/dT at T*. Its eigenvalue of largest modulus is the leading
    multiplier. Dropping c_old removes the amplitude dependence of
    inherited-flux transport ("frozen").

    Scalar map. M = 1 - (1 + r*) q* - T* dr/dT, with r* = w^T G B*/D*.

    Settled-dipole (two-variable) map. After one cycle of transport any
    field lies in the slow dipole mode, so the complete map closes on the
    end-of-cycle dipole and the settled dipole E_n = w^T G(T_{n+1}) B_n:
        d_n = P(T_n) - e_{n-1},      T_{n+1} = k d_n,
        e_n = P_s(T_n, T_{n+1}) - lambda_1(T_{n+1}) e_{n-1},
    where P_s(T, T') = w^T G(T') b(T) is the dipole of the new flux after it
    has settled, and lambda_1 the survival of already settled flux. Its
    Jacobian at the fixed point is evaluated here with all four derivatives.
    """
    w = mdl.w
    G = mdl.propagator(Tstar)
    b = mdl.new_flux(Tstar)[:, 0]
    Bs = mdl.periodic_state(Tstar, G, b)
    Ds = float(w @ Bs)
    k = Tstar / Ds
    Gb = G @ b
    Ts = np.array([Tstar + h, Tstar - h])
    # Source-free runs with the flow amplitude varied, for B*, b and G b.
    X = mdl.cycle(np.column_stack([Bs, Bs, b, b, Gb, Gb]), 0.0, 0.0,
                  T_flow=np.tile(Ts, 3))
    bb = mdl.new_flux(Ts)
    c_old = (X[:, 0] - X[:, 1]) / (2 * h)
    c_new = (bb[:, 0] - bb[:, 1]) / (2 * h)

    def leading(J):
        ev = np.linalg.eigvals(J)
        return ev[np.argmax(np.abs(ev))]

    full = leading(-G - k * np.outer(c_old - c_new, w))
    frozen = leading(-G + k * np.outer(c_new, w))

    # Scalar map
    r_star = float(w @ (G @ Bs)) / Ds
    Tr_prime = Tstar * float(w @ c_old) / Ds
    P = float(w @ b)
    dP = float(w @ c_new)
    q = 1.0 - Tstar * dP / P
    scalar = 1.0 - (1.0 + r_star) * q - Tr_prime

    # Settled-dipole map
    Ps = float(w @ Gb)
    d1Ps = float(w @ (G @ c_new))                    # structure of the new flux
    d2Ps = float(w @ (X[:, 2] - X[:, 3])) / (2 * h)  # flow acting while it settles
    lam1 = float(w @ (G @ Gb)) / Ps
    dlam1 = float(w @ (X[:, 4] - X[:, 5])) / (2 * h) / Ps
    e_star = Ps / (1.0 + lam1)
    k2 = Tstar / (P - e_star)
    beta = d2Ps - dlam1 * e_star
    J2 = np.array([[k2 * dP, -k2],
                   [d1Ps + beta * k2 * dP, -lam1 - beta * k2]])
    two_var = leading(J2)
    return dict(T=Tstar, D0=k * a, Dstar=Ds, k=k, P=P, q=q,
                full=float(full.real), full_imag=float(abs(full.imag)),
                frozen=float(frozen.real), scalar=scalar,
                r_star=r_star, Tr_prime=Tr_prime,
                two_var=float(two_var.real), D0_two_var=k2 * a,
                r1=Ps / P, lam1=lam1, Tdln_Ps=Tstar * d1Ps / Ps,
                r_star_two_var=e_star / (P - e_star))


def scalar_map_threshold(T_grid, P_new, a, r_const):
    """Period-doubling gain of the scalar map with constant r (tabulated Q)."""
    Q = P_new / (a * T_grid)
    q = -T_grid * np.gradient(np.log(Q), T_grid, edge_order=2)
    D0 = (1.0 + r_const) / Q
    M = 1.0 - (1.0 + r_const) * q
    return crossing(D0, M, -1.0)


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """First x where y falls through level (linear interpolation)."""
    x, y = np.asarray(x), np.asarray(y)
    idx = np.where((y[:-1] > level) & (y[1:] <= level))[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def run_feedback(mdl: Model, k_gains: np.ndarray, n_cycles: int, T0: float = 1.0):
    """Complete multi-cycle SFT with feedback T_{n+1} = k |D_n|, Hale signs."""
    m = len(k_gains)
    B = np.zeros((mdl.n, m))
    T = np.full(m, T0)
    s = np.ones(m)
    out = np.empty((n_cycles, m))
    for i in range(n_cycles):
        out[i] = T
        B = mdl.cycle(B, T, s)
        D = mdl.dipole(B)
        T = k_gains * np.abs(D)
        s = -np.sign(D)
    return out


def is_modulated(T_seq: np.ndarray, n_keep: int) -> np.ndarray:
    """True where the even-odd alternation of T_n persists (columns).

    Near a multiplier of -1 a transient alternation decays very slowly, so
    a fixed tolerance on the spread would misread it as modulation. The
    alternation is taken as persistent if it is not decaying: its amplitude
    over the last n_keep cycles is at least 0.99 times that over the
    preceding n_keep cycles (a decaying transient at multiplier M gives
    |M|^n_keep, e.g. 0.79 for M = -0.99), and not negligibly small.
    """
    def amp(X):
        return np.abs(X[1:] - X[:-1]).mean(axis=0) / X.mean(axis=0)
    late = amp(T_seq[-n_keep:])
    early = amp(T_seq[-2 * n_keep:-n_keep])
    return (late > 1e-6) & (late >= 0.99 * early)


def run_scalar_map(k: float, n_cycles: int, T_tab, P_tab, r: float, T0: float = 1.0):
    """Scalar memory map D_n = s_n P(T_n) + r D_{n-1}, T_{n+1} = k |D_n|."""
    T, s, D_prev = T0, 1.0, 0.0
    out = np.empty(n_cycles)
    for i in range(n_cycles):
        out[i] = T
        D = s * np.interp(T, T_tab, P_tab) + r * D_prev
        T, s, D_prev = k * abs(D), -np.sign(D), D
    return out


def _bilinear(x: float, y: float, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """Bilinear interpolation of Z[i, j] on the grid (X[i], Y[j]), clamped."""
    i = int(np.clip(np.searchsorted(X, x) - 1, 0, len(X) - 2))
    j = int(np.clip(np.searchsorted(Y, y) - 1, 0, len(Y) - 2))
    tx = float(np.clip((x - X[i]) / (X[i + 1] - X[i]), 0.0, 1.0))
    ty = float(np.clip((y - Y[j]) / (Y[j + 1] - Y[j]), 0.0, 1.0))
    return ((1 - tx) * (1 - ty) * Z[i, j] + tx * (1 - ty) * Z[i + 1, j]
            + (1 - tx) * ty * Z[i, j + 1] + tx * ty * Z[i + 1, j + 1])


def settled_tables(mdl: Model, T_tab: np.ndarray, T_flow_tab: np.ndarray):
    """Tables for the settled-dipole map.

    Ps[i, j] = w^T G(T_flow_j) b(T_i): dipole of the flux of a cycle of
    amplitude T_i after one further cycle of transport under the inflows of
    a cycle of amplitude T_flow_j. lam[j]: survival of an already settled
    field (the T = 1 flux after one cycle) under the same inflows.
    """
    b = mdl.new_flux(T_tab)
    phi = mdl.cycle(mdl.new_flux(1.0), 0.0, 0.0, T_flow=1.0)[:, 0]
    Ps = np.empty((len(T_tab), len(T_flow_tab)))
    lam = np.empty(len(T_flow_tab))
    for j, Tf in enumerate(T_flow_tab):
        X = mdl.cycle(np.column_stack([b, phi]), 0.0, 0.0, T_flow=float(Tf))
        d = mdl.dipole(X)
        Ps[:, j] = d[:-1]
        lam[j] = d[-1] / float(mdl.w @ phi)
    return Ps, lam


def run_settled_map(k: float, n_cycles: int, T_tab, P_tab, T_flow_tab, Ps_tab,
                    lam_tab, T0: float = 1.0):
    """Settled-dipole map (two variables), signed form.

    D_n = s_n P(T_n) + E_{n-1},  T_{n+1} = k |D_n|,  s_{n+1} = -sign(D_n),
    E_n = s_n P_s(T_n, T_{n+1}) + lambda_1(T_{n+1}) E_{n-1}.
    """
    T, s, E = T0, 1.0, 0.0
    out = np.empty(n_cycles)
    for i in range(n_cycles):
        out[i] = T
        D = s * np.interp(T, T_tab, P_tab) + E
        T_next = k * abs(D)
        E = (s * _bilinear(T, T_next, T_tab, T_flow_tab, Ps_tab)
             + np.interp(T_next, T_flow_tab, lam_tab) * E)
        T, s = T_next, -np.sign(D)
    return out


def sequence_test(mdl: Model, T_seq: np.ndarray):
    """Exact split along a prescribed sequence with Hale alternation.

    Each cycle advances two columns from the same field: the complete run
    and a source-free run with the same inflows, so the inherited-flux term
    R_n = w^T G(T_n) B_{n-1} is measured directly. Returns D_n, s_n, P(T_n),
    R_n and the largest superposition residual |D - s P - R| / |P|.
    """
    n = mdl.n
    B = np.zeros(n)
    s = 1.0
    D, S, R = [], [], []
    for T in T_seq:
        X = mdl.cycle(np.column_stack([B, B]), T, np.array([s, 0.0]),
                      T_flow=np.array([T, T]))
        B = X[:, 0]
        Dn, Rn = mdl.dipole(X)
        D.append(Dn)
        S.append(s)
        R.append(Rn)
        s = -np.sign(Dn)
    D, S, R = map(np.asarray, (D, S, R))
    P = mdl.dipole(mdl.new_flux(T_seq))
    resid = float(np.max(np.abs(D - S * P - R) / np.abs(P)))
    return D, S, P, R, resid


# ============================================================
# 3. Parallel jobs (top level so that they can be pickled)
# ============================================================

def _job_stability(args):
    p, c, a, T = args
    return full_state_stability(Model(p, c), T, a)


def _job_new_flux(args):
    p, c, T_grid = args
    mdl = Model(p, c)
    return mdl.dipole(mdl.new_flux(T_grid))


def _job_remnant_table(args):
    """r on the periodic-state profile of T_prev, under the flow of T_flow."""
    p, c, T_prev, T_flow = args
    mdl = Model(p, c)
    Bs = mdl.periodic_state(T_prev)
    return remnant_ratio(mdl, T_flow, Bs)


def _job_budget(args):
    p, c, T0 = args
    mdl = Model(p, c)
    Bs = mdl.periodic_state(T0)
    # A cycle of positive source sign starting from -B* (the Hale state).
    adj = adjoint_budget(mdl, T0, -Bs, 1.0)
    fd = finite_difference_budget(mdl, T0, -Bs, 1.0)
    return adj, fd


def _job_feedback(args):
    p, c, k_gains, n_cycles = args
    return run_feedback(Model(p, c), k_gains, n_cycles)


def _job_settled_tables(args):
    p, c, T_tab, T_flow_chunk = args
    return settled_tables(Model(p, c), T_tab, T_flow_chunk)


def _job_sequence(args):
    p, c, T_seq = args
    mdl = Model(p, c)
    D, S, P, R, resid = sequence_test(mdl, T_seq)
    r_best = float(remnant_ratio(mdl, np.array([1.0]), mdl.periodic_state(1.0))[0])
    return D, S, P, R, resid, r_best


def stability_scan(pool, p, c, a, T_grid):
    """Multipliers along the fixed-point family and the gains where they
    reach -1. The settled-dipole map is scanned along its own fixed points."""
    rows = list(pool.map(_job_stability, [(p, c, a, float(T)) for T in T_grid]))
    D0 = np.array([r["D0"] for r in rows])
    thr = {key: crossing(D0, np.array([r[key] for r in rows]), -1.0)
           for key in ("full", "frozen", "scalar")}
    thr["two_var"] = crossing([r["D0_two_var"] for r in rows],
                              [r["two_var"] for r in rows], -1.0)
    return rows, thr


# ============================================================
# 4. Figures
# ============================================================

def make_figures(res: dict) -> None:
    """Draw figures I1-I5 from the results gathered by the main analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p, mdl_ref = res["p"], res["mdl_ref"]
    surv, T_axis, budgets = res["surv"], res["T_axis"], res["budgets"]
    stab, bif = res["stab"], res["bif"]
    n_keep, n_cyc = res["n_keep"], res["n_cyc"]
    seq, T_seq, burn = res["seq"], res["T_seq"], res["burn"]

    # ---------------------------------------------------------------
    # Figures (no titles; identity by marker and line style as well as colour)
    # ---------------------------------------------------------------
    styles = {"fixed transport": ("C0", "o", "-"),
              "inflows 2 m/s": ("C1", "s", "--"),
              "inflows 5 m/s": ("C3", "^", ":")}

    # I1: remnant survival, flow effect vs structure effect
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for name, (rf, rp) in surv.items():
        col, mk, ls = styles[name]
        axes[0].plot(T_axis, rf, ls, marker=mk, color=col, markersize=4, label=name)
        axes[1].plot(T_axis, rp, ls, marker=mk, color=col, markersize=4, label=name)
    axes[0].set_xlabel(r"Amplitude of the cycle whose inflows act, $T_{n}$")
    axes[1].set_xlabel(r"Amplitude of the cycle that left the field, $T_{n-1}$")
    axes[0].set_ylabel(r"Inherited-dipole survival over one cycle, $r$")
    axes[0].text(0.03, 0.04, "(a) inherited profile fixed ($T_{n-1}=1$)",
                 transform=axes[0].transAxes, fontsize=9)
    axes[1].text(0.03, 0.04, "(b) inflows fixed ($T_n=1$)",
                 transform=axes[1].transAxes, fontsize=9)
    axes[1].legend(fontsize=8, loc="center left")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figI1_remnant_survival.png"), dpi=200)
    plt.close(fig)

    # I2: adjoint budget
    src, tnew, told, dnew, dold, _ = budgets["inflows 2 m/s"]
    fig = plt.figure(figsize=(12, 4.2))
    ax0 = fig.add_subplot(1, 3, 1)
    labels = ["source", "transport of\nnew flux", "transport of\ninherited flux"]
    x = np.arange(3)
    for j, (name, hatch) in enumerate((("inflows 2 m/s", ""), ("inflows 5 m/s", "//"))):
        vals = budgets[name][:3]
        xs = x + (j - 0.5) * 0.36
        ax0.bar(xs, vals, width=0.34, color=styles[name][0],
                hatch=hatch, edgecolor="white", label=name)
        # The inherited-flux bars are too small to see: label their values.
        ax0.annotate(f"{vals[2]:+.1e}", (xs[2], vals[2]), xytext=(0, 6 + 12 * j),
                     textcoords="offset points", ha="center", fontsize=7)
    ax0.axhline(0.0, color="0.3", linewidth=0.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=8)
    ax0.set_ylabel(r"Contribution to $dD_{\rm end}/dT$ at $T=1$ [arb. units]")
    ax0.legend(fontsize=8)
    t_yr = mdl_ref.t_yr
    for j, (dens, lab) in enumerate(((dnew, "new flux"), (dold, "inherited flux"))):
        ax = fig.add_subplot(1, 3, 2 + j)
        Z = dens / (mdl_ref.dt / SECONDS_PER_YEAR)   # per year
        vmax = float(np.max(np.abs(Z)))
        im = ax.pcolormesh(t_yr[::10], mdl_ref.lat_deg, Z[::10].T, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax.set_xlabel("Time within the cycle [yr]")
        ax.set_ylabel("Latitude [deg]")
        ax.text(0.03, 0.04, f"{lab}: total {dens.sum():+.2e}", transform=ax.transAxes,
                fontsize=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        fig.colorbar(im, ax=ax, label="Transport sensitivity density [arb. units]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figI2_adjoint_budget.png"), dpi=200)
    plt.close(fig)

    # I3: leading multiplier of the complete field, the settled-dipole map and
    # the scalar map. The leading eigenvalue of the complete field switches
    # from a positive to a negative branch near q = 1/2 and never passes
    # through zero, so its line is broken at the change of sign.
    def plot_broken(ax, x, y, *args, label=None, **kw):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        cut = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0] + 1
        for i, (xs, ys) in enumerate(zip(np.split(x, cut), np.split(y, cut))):
            ax.plot(xs, ys, *args, label=label if i == 0 else None, **kw)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (name, (rows, thr)) in zip(axes, stab.items()):
        D0 = [r["D0"] for r in rows]
        plot_broken(ax, D0, [r["full"] for r in rows], "-", color="k", marker=".",
                    markersize=3, label="complete field")
        ax.plot([r["D0_two_var"] for r in rows], [r["two_var"] for r in rows], "o",
                markerfacecolor="none", markeredgecolor="C1", markersize=6,
                label="settled-dipole map (two variables)")
        ax.plot(D0, [r["scalar"] for r in rows], ":", color="C0", linewidth=2,
                label=r"scalar map with $r(T)$")
        ax.axhline(-1.0, color="0.5", linewidth=0.8)
        ax.set_xlabel(r"Linear gain $D_0 = k\,a$")
        ax.text(0.97, 0.94, name, transform=ax.transAxes, ha="right", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Leading cycle-to-cycle multiplier")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figI3_stability_multipliers.png"), dpi=200)
    plt.close(fig)

    # I4: nonlinear multi-cycle attractors
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (name, (D0s, T_full, T_map, T_two)) in zip(axes, bif.items()):
        thr = stab[name][1]
        for j, D0 in enumerate(D0s):
            ax.plot([D0] * n_keep, T_map[-n_keep:, j], "_", color="0.5", markersize=9,
                    label="scalar map, constant $r$" if j == 0 else None)
            ax.plot([D0] * n_keep, T_two[-n_keep:, j], "o", markerfacecolor="none",
                    markeredgecolor="C0", markersize=7,
                    label="settled-dipole map (two variables)" if j == 0 else None)
            ax.plot([D0] * n_keep, T_full[-n_keep:, j], "o", color="C3", markersize=3,
                    label="complete multi-cycle SFT" if j == 0 else None)
        ax.axvline(thr["full"], color="k", linestyle="--", linewidth=1,
                   label=f"complete-field threshold {thr['full']:.3f}")
        ax.axvline(thr["const_r"], color="0.4", linestyle=":", linewidth=1.2,
                   label=f"scalar-map threshold {thr['const_r']:.3f}")
        ax.set_xlabel(r"Linear gain $D_0 = k\,a$")
        ax.set_ylabel(rf"$T_n$ in the last {n_keep} of {n_cyc} cycles")
        ax.text(0.03, 0.95, name, transform=ax.transAxes, fontsize=9)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figI4_nonlinear_onset.png"), dpi=200)
    plt.close(fig)

    # I5: error of scalar remnant subtraction along the prescribed sequence,
    # against the amplitude of the PREVIOUS cycle, which sets how well the
    # inherited dipole survives.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    seq_styles = {0.0: ("C0", "o"), 2.0: ("C1", "s"), 5.0: ("C3", "^")}
    for ax, tau in zip(axes, (np.inf, 8.0)):
        rho = float(np.exp(-p.cycle_years / tau))
        for amp in (0.0, 2.0, 5.0):
            D, S, P, R, _ = seq[(amp, tau)]
            dD = S[1:] * (D[1:] - rho * D[:-1])
            err_pc = 100.0 * (dD / P[1:] - 1.0)
            col, mk = seq_styles[amp]
            ax.plot(T_seq[burn - 1:-1], err_pc[burn - 1:], mk, color=col, markersize=5,
                    label=f"inflows {amp:g} m/s" if amp else "no inflows")
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.set_xlabel(r"Amplitude of the previous cycle, $T_{n-1}$")
        lab = r"$\tau=\infty$, $\rho=1$" if not np.isfinite(tau) else \
            rf"$\tau={tau:g}$ yr, $\rho=e^{{-11/\tau}}={rho:.3f}$"
        ax.text(0.03, 0.94, lab, transform=ax.transAxes, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"Error of $s_n(D_n-\rho D_{n-1})$ relative to $P(T_n)$ [%]")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figI5_scalar_subtraction.png"), dpi=200)
    plt.close(fig)



# ============================================================
# 5. Main analysis
# ============================================================

if __name__ == "__main__":
    # "--quick" runs every stage on small grids as a smoke test (about three
    # minutes) and writes to a temporary folder, never to the tracked outputs.
    QUICK = "--quick" in sys.argv
    if QUICK:
        OUTDIR = tempfile.mkdtemp(prefix="inflow_memory_quick_")
        print(f"Quick smoke test: results are not converged; writing to {OUTDIR}")
    os.makedirs(OUTDIR, exist_ok=True)
    t_start = time.time()
    p = nre.SFTParams()
    a = linear_yield(p)
    summary: list[tuple[str, float]] = [("linear_yield_a", a)]

    REF = Coupling(inflow_amp_m_s=2.0, lat_mode="efficiency")
    STRONG = Coupling(inflow_amp_m_s=5.0, lat_mode="efficiency")
    FIXED = Coupling(inflow_amp_m_s=0.0, lat_mode="efficiency", q_inflow_eff=True)
    configs = {"fixed transport": FIXED, "inflows 2 m/s": REF,
               "inflows 5 m/s": STRONG}

    pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    # --- 0. validation -------------------------------------------------------
    err = validate_against_reference(p)
    summary.append(("validation_max_rel_diff", err))
    print(f"[0] Vectorised model vs reference solver: max relative difference {err:.2e}")

    # --- 1. remnant survival: flow effect vs structure effect -----------------
    T_axis = np.array([0.5, 1.0, 1.5]) if QUICK else \
        np.round(np.arange(0.25, 2.001, 0.125), 4)
    surv = {}
    for name, c in configs.items():
        # (i) flow of the current cycle varied, inherited profile fixed (T_prev = 1)
        flow = pool.submit(_job_remnant_table, (p, c, 1.0, T_axis))
        # (ii) profile left by cycles of amplitude T_prev, flow fixed (T = 1)
        prof = list(pool.map(_job_remnant_table,
                             [(p, c, float(Tp), np.array([1.0])) for Tp in T_axis]))
        surv[name] = (flow.result(), np.array([x[0] for x in prof]))
    with open(os.path.join(OUTDIR, "remnant_survival.csv"), "w", encoding="utf-8") as f:
        f.write("config,T,r_vs_flow_amplitude,r_vs_previous_amplitude\n")
        for name, (rf, rp) in surv.items():
            for T, x, y in zip(T_axis, rf, rp):
                f.write(f"{name},{T:.4f},{x:.6f},{y:.6f}\n")
    i05, i15 = int(np.argmin(abs(T_axis - 0.5))), int(np.argmin(abs(T_axis - 1.5)))
    for name, (rf, rp) in surv.items():
        print(f"[1] {name}: r over T = 0.5-1.5: flow-only change "
              f"{rf[i15] - rf[i05]:+.4f}, profile-only change {rp[i15] - rp[i05]:+.4f}")
        summary.append((f"dr_flow_0.5_to_1.5[{name}]", rf[i15] - rf[i05]))
        summary.append((f"dr_profile_0.5_to_1.5[{name}]", rp[i15] - rp[i05]))

    # --- 2. adjoint budget at T = 1 -------------------------------------------
    budgets = {}
    for name in ("inflows 2 m/s", "inflows 5 m/s"):
        budgets[name] = pool.submit(_job_budget, (p, configs[name], 1.0))
    for name, fut in budgets.items():
        (src, tnew, told, dnew, dold), fd = fut.result()
        budgets[name] = (src, tnew, told, dnew, dold, fd)
        dev = max(abs(x - y) / max(abs(y), 1e-30) for x, y in zip((src, tnew, told), fd))
        print(f"[2] {name}: dD/dT = {src:+.5f} (source) {tnew:+.5f} (new-flux "
              f"transport) {told:+.6f} (inherited-flux transport); "
              f"adjoint vs finite differences {dev:.1e}")
        summary += [(f"budget_source[{name}]", src), (f"budget_transport_new[{name}]", tnew),
                    (f"budget_transport_inherited[{name}]", told),
                    (f"budget_adjoint_vs_fd_max_rel[{name}]", dev)]

    # --- 3. stability of the complete field evolution -------------------------
    T_star = np.array([0.9, 1.0, 1.25, 1.35, 1.6]) if QUICK else \
        np.round(np.arange(0.70, 1.801, 0.05), 3)
    # Fine table: the scalar map interpolates P(T) linearly, and its slope
    # must match the smooth derivative used for the analytic threshold.
    T_tab = np.linspace(0.3, 2.5, 45 if QUICK else 441)
    # The remnant coefficient of the published scalar map: source-free
    # survival, over one cycle, of the field left by a linear T = 1 cycle.
    mdl_lin = Model(p, Coupling(inflow_amp_m_s=0.0, lat_mode="none"))
    r_clean = float(remnant_ratio(mdl_lin, np.array([1.0]),
                                  mdl_lin.new_flux(1.0)[:, 0])[0])
    summary.append(("r_clean_slate", r_clean))
    print(f"[3] Clean-slate remnant coefficient of the scalar map: r = {r_clean:.6f}")
    stab, P_tabs = {}, {}
    for name, c in configs.items():
        stab[name] = stability_scan(pool, p, c, a, T_star)
        P_tabs[name] = pool.submit(_job_new_flux, (p, c, T_tab)).result()
        rows, thr = stab[name]
        row1 = [r for r in rows if abs(r["T"] - 1.0) < 1e-9][0]
        row16 = [r for r in rows if abs(r["T"] - 1.6) < 1e-9][0]
        thr["r_star_T1"] = row1["r_star"]
        thr["const_r"] = scalar_map_threshold(T_tab, P_tabs[name], a, row1["r_star"])
        thr["clean_r"] = scalar_map_threshold(T_tab, P_tabs[name], a, r_clean)
        # Bias of Q read from equilibrium amplitudes with a constant r.
        q_bias = (1.0 + row1["r_star"]) / (1.0 + row16["r_star"]) - 1.0
        closure = max(abs(r["two_var"] - r["full"]) for r in rows)
        print(f"[3] {name}: period-doubling gain D0: complete field {thr['full']:.4f}, "
              f"inherited transport frozen {thr['frozen']:.4f}, settled-dipole map "
              f"{thr['two_var']:.4f}, scalar map r(T) {thr['scalar']:.4f}, "
              f"r*(1) {thr['const_r']:.4f}, clean-slate r {thr['clean_r']:.4f}; "
              f"max |settled-dipole - complete| multiplier {closure:.1e}; "
              f"first-cycle survival r1 at T*=1: {row1['r1']:.5f}, settled "
              f"survival {row1['lam1']:.5f}")
        for key in ("full", "frozen", "two_var", "scalar", "const_r", "clean_r"):
            summary.append((f"D0_pd_{key}[{name}]", thr[key]))
        # Amplitude and stiffness at the complete-field threshold, and the
        # amplitude at which the settled dipole of the new flux peaks.
        Ts_ = [r["T"] for r in rows]
        T_pd = crossing(Ts_, [r["full"] for r in rows], -1.0)
        T_peak = crossing(Ts_, [r["Tdln_Ps"] for r in rows], 0.0)
        q_pd = float(np.interp(T_pd, Ts_, [r["q"] for r in rows]))
        u_pd = 1.0 - float(np.interp(T_pd, Ts_, [r["r1"] for r in rows]))
        # Critical stiffness of the scalar map at the same amplitude.
        q_scalar = 2.0 / (1.0 + float(np.interp(T_pd, Ts_, [r["r_star"] for r in rows])))
        rel = {key: thr[key] / thr["full"] - 1.0
               for key in ("frozen", "two_var", "scalar", "const_r", "clean_r")}
        print(f"[3] {name}: complete-field threshold at T* = {T_pd:.4f} "
              f"(stiffness q* = {q_pd:.4f}; scalar map 2/(1+r*) = {q_scalar:.4f}); "
              f"settled dipole of the new flux peaks at T = {T_peak:.4f}; "
              f"unsettled fraction at the threshold {u_pd:.4f}")
        print(f"[3] {name}: threshold relative to the complete field: frozen "
              f"{100*rel['frozen']:+.3f}%, settled-dipole map {100*rel['two_var']:+.3f}%, "
              f"scalar r(T) {100*rel['scalar']:+.2f}%, r*(1) {100*rel['const_r']:+.2f}%, "
              f"clean-slate r {100*rel['clean_r']:+.2f}%")
        summary += [(f"T_star_pd_full[{name}]", T_pd), (f"q_star_pd_full[{name}]", q_pd),
                    (f"q_star_pd_scalar_map[{name}]", q_scalar),
                    (f"T_peak_settled_dipole[{name}]", T_peak),
                    (f"unsettled_fraction_at_pd[{name}]", u_pd)]
        summary += [(f"D0_pd_rel_to_full_{key}[{name}]", val) for key, val in rel.items()]
        summary += [(f"two_var_max_multiplier_diff[{name}]", closure),
                    (f"r1_first_cycle_survival_T1[{name}]", row1["r1"]),
                    (f"lambda1_settled_survival_T1[{name}]", row1["lam1"]),
                    (f"r_star_T1[{name}]", row1["r_star"]),
                    (f"r_star_T1.6[{name}]", row16["r_star"]),
                    (f"Q_bias_const_r_T1.6[{name}]", q_bias)]
    with open(os.path.join(OUTDIR, "stability_scan.csv"), "w", encoding="utf-8") as f:
        f.write("config,T_star,D0,q,r_star,T_dr_dT,mult_full,mult_frozen,mult_scalar,"
                "D0_two_var,mult_two_var,r1_first_cycle,lambda1_settled,T_dlnPs_dT\n")
        for name, (rows, _) in stab.items():
            for r in rows:
                f.write(f"{name},{r['T']:.3f},{r['D0']:.6f},{r['q']:.6f},"
                        f"{r['r_star']:.6f},{r['Tr_prime']:.6f},{r['full']:.6f},"
                        f"{r['frozen']:.6f},{r['scalar']:.6f},{r['D0_two_var']:.6f},"
                        f"{r['two_var']:.6f},{r['r1']:.6f},{r['lam1']:.6f},"
                        f"{r['Tdln_Ps']:.6f}\n")

    # --- 3b. nonlinear multi-cycle check ----------------------------------------
    # Gains bracket both thresholds; transients near a multiplier of -1 decay
    # slowly, so the runs are long and the onset is read from the last cycles.
    n_cyc, n_keep = (60 if QUICK else 400), 24
    n_gain = 4 if QUICK else 16
    bif, futs = {}, {}
    half = max(os.cpu_count() // 2, 1)
    T_flow_tab = np.linspace(0.3, 2.5, 4 if QUICK else 12)
    for name in ("inflows 2 m/s", "inflows 5 m/s"):
        thr = stab[name][1]
        D0s = np.linspace(thr["full"] - 0.12, thr["const_r"] + 0.12, n_gain)
        futs[name] = (D0s,
                      [pool.submit(_job_feedback, (p, configs[name], ch, n_cyc))
                       for ch in np.array_split(D0s / a, half)],
                      [pool.submit(_job_settled_tables, (p, configs[name], T_tab, ch))
                       for ch in np.array_split(T_flow_tab, half)])
    for name, (D0s, fl, tl) in futs.items():
        thr = stab[name][1]
        T_full = np.concatenate([f.result() for f in fl], axis=1)
        Ps_tab = np.concatenate([f.result()[0] for f in tl], axis=1)
        lam_tab = np.concatenate([f.result()[1] for f in tl])
        # The maps start from the same state as the SFT (no field, T = 1) and
        # run for the same number of cycles, so they can be compared cycle
        # by cycle and the onset is read in the same way.
        T_map = np.column_stack([run_scalar_map(k, n_cyc, T_tab, P_tabs[name],
                                                thr["r_star_T1"]) for k in D0s / a])
        T_two = np.column_stack([run_settled_map(k, n_cyc, T_tab, P_tabs[name],
                                                 T_flow_tab, Ps_tab, lam_tab)
                                 for k in D0s / a])
        bif[name] = (D0s, T_full, T_map, T_two)

        def onset(X):
            mod = is_modulated(X, n_keep)
            return D0s[np.argmax(mod)] if mod.any() else np.nan

        # Agreement with the complete SFT: (i) cycle by cycle, per gain (a
        # period-two state reached with the opposite phase counts as a
        # miss), and (ii) of the attractor, from the sorted last cycles.
        def agreement(X):
            per_gain = np.max(np.abs(X - T_full), axis=0)
            locked = per_gain < 0.01
            worst = float(per_gain[locked].max()) if locked.any() else np.nan
            attr = float(np.max(np.abs(np.sort(X[-n_keep:], axis=0)
                                       - np.sort(T_full[-n_keep:], axis=0))))
            return int(locked.sum()), worst, attr

        (n_two, w_two, at_two), (n_map, w_map, at_map) = agreement(T_two), agreement(T_map)
        print(f"[3b] {name}: first modulated gain in the scan: complete SFT "
              f"{onset(T_full):.3f}, settled-dipole map {onset(T_two):.3f}, scalar "
              f"map {onset(T_map):.3f}")
        print(f"[3b] {name}: settled-dipole map follows the SFT cycle by cycle "
              f"(within 0.01 over {n_cyc} cycles) at {n_two} of {len(D0s)} gains, "
              f"largest difference there {w_two:.1e}, attractor difference "
              f"{at_two:.1e}; scalar map: {n_map} of {len(D0s)} gains, attractor "
              f"difference {at_map:.1e}")
        summary += [(f"nonlinear_onset_full[{name}]", onset(T_full)),
                    (f"nonlinear_onset_two_var[{name}]", onset(T_two)),
                    (f"nonlinear_onset_scalar[{name}]", onset(T_map)),
                    (f"gains_followed_cycle_by_cycle_two_var[{name}]", n_two),
                    (f"gains_followed_cycle_by_cycle_scalar[{name}]", n_map),
                    (f"cycle_by_cycle_max_diff_two_var[{name}]", w_two),
                    (f"attractor_max_diff_two_var[{name}]", at_two),
                    (f"attractor_max_diff_scalar[{name}]", at_map)]

    # --- 4. scalar remnant subtraction along a prescribed sequence ------------
    rng = np.random.default_rng(2026)
    T_seq = np.clip(np.exp(0.3 * rng.standard_normal(16 if QUICK else 60)), 0.4, 2.0)
    burn = 4 if QUICK else 10
    seq_cases = [(amp, tau) for tau in (np.inf, 8.0) for amp in (0.0, 2.0, 5.0)]
    futs = {}
    for amp, tau in seq_cases:
        pt = replace(p, tau_yr=tau)
        futs[(amp, tau)] = pool.submit(
            _job_sequence, (pt, Coupling(inflow_amp_m_s=amp, lat_mode="none"), T_seq))
    a_tau = {tau: linear_yield(replace(p, tau_yr=tau)) for tau in (np.inf, 8.0)}
    seq = {}
    with open(os.path.join(OUTDIR, "sequence_test.csv"), "w", encoding="utf-8") as f, \
            open(os.path.join(OUTDIR, "sequence_cycles.csv"), "w", encoding="utf-8") as fc:
        f.write("inflow_m_s,tau_yr,estimator,rho,rms_err,max_abs_err,b_true,b_inferred,"
                "superposition_residual\n")
        fc.write("inflow_m_s,tau_yr,cycle,T_n,T_previous,source_sign,D_n,"
                 "new_flux_dipole_P,inherited_dipole_R,true_survival_R_over_D_previous\n")
        for (amp, tau), fut in futs.items():
            D, S, P, R, resid, r_best = fut.result()
            seq[(amp, tau)] = (D, S, P, R, r_best)
            for i in range(1, len(T_seq)):
                fc.write(f"{amp},{tau},{i},{T_seq[i]:.6f},{T_seq[i-1]:.6f},{S[i]:+.0f},"
                         f"{D[i]:.8e},{P[i]:.8e},{R[i]:.8e},{R[i] / D[i-1]:.6f}\n")
            sl = slice(burn, None)
            b_true = fit_rational_b(T_seq[sl], P[sl], a_tau[tau])
            summary.append((f"seq_b_true[{amp},{tau}]", b_true))
            for est, rho in (("exp(-P/tau)", float(np.exp(-p.cycle_years / tau))),
                             ("best constant r", r_best)):
                dD = S[1:] * (D[1:] - rho * D[:-1])
                errs = (dD / P[1:] - 1.0)[burn - 1:]
                b_inf = fit_rational_b(T_seq[sl], dD[burn - 1:], a_tau[tau])
                f.write(f"{amp},{tau},{est},{rho:.6f},{np.sqrt(np.mean(errs**2)):.6e},"
                        f"{np.max(np.abs(errs)):.6e},{b_true:.6f},{b_inf:.6f},{resid:.3e}\n")
                print(f"[4] inflow {amp} m/s, tau {tau}: {est:16s} rho={rho:.4f} "
                      f"rms err {100*np.sqrt(np.mean(errs**2)):.2f}%, max "
                      f"{100*np.max(np.abs(errs)):.2f}%, b true {b_true:.4f} -> "
                      f"inferred {b_inf:.4f} (superposition residual {resid:.1e})")
                summary += [(f"seq_rms_err[{amp},{tau},{est}]", float(np.sqrt(np.mean(errs**2)))),
                            (f"seq_max_err[{amp},{tau},{est}]", float(np.max(np.abs(errs)))),
                            (f"seq_b_inferred[{amp},{tau},{est}]", b_inf)]
        # Inflow-induced reduction of the cycle contribution, as in the 2025 paper.
        for tau in (np.inf, 8.0):
            rho = float(np.exp(-p.cycle_years / tau))
            D0_, S0_, P0_, _, _ = seq[(0.0, tau)]
            for amp in (2.0, 5.0):
                D_, S_, P_, _, _ = seq[(amp, tau)]
                dD_inf = S_[1:] * (D_[1:] - rho * D_[:-1])
                dD_ref = S0_[1:] * (D0_[1:] - rho * D0_[:-1])
                red_est = 1.0 - np.mean(dD_inf[burn - 1:]) / np.mean(dD_ref[burn - 1:])
                red_true = 1.0 - np.mean(P_[burn:]) / np.mean(P0_[burn:])
                print(f"[4] reduction of the cycle contribution, {amp} m/s, tau {tau}: "
                      f"true {100*red_true:.2f}%, from D_n - rho D_(n-1) {100*red_est:.2f}%")
                summary += [(f"reduction_true[{amp},{tau}]", red_true),
                            (f"reduction_estimated[{amp},{tau}]", red_est)]

    # --- 5. controls and refinement --------------------------------------------
    # Envelope-timed inflows, calibrated so that P(T = 1) matches the reference.
    P1_ref = float(_job_new_flux((p, REF, np.array([1.0])))[0])
    lo, hi = 2.0, 12.0
    for _ in range(6 if QUICK else 30):
        mid = 0.5 * (lo + hi)
        P1 = float(_job_new_flux((p, replace(REF, inflow_amp_m_s=mid,
                                             inflow_timing="envelope"),
                                  np.array([1.0])))[0])
        lo, hi = (mid, hi) if P1 > P1_ref else (lo, mid)
    amp_env = 0.5 * (lo + hi)
    print(f"[5] Envelope-timed inflow calibrated to the reference P(1): "
          f"{amp_env:.3f} m/s at activity maximum")
    summary.append(("envelope_inflow_amplitude_m_s", amp_env))
    p_shift = replace(p, latitude_quenching_mode="shift")
    controls = {
        "reference (2 m/s)": (p, REF),
        "envelope-timed inflow": (p, replace(REF, inflow_amp_m_s=amp_env,
                                             inflow_timing="envelope")),
        "shifted belt, no inflow": (p_shift, replace(REF, lat_mode="shift",
                                                     inflow_amp_m_s=0.0)),
        "shifted belt, fixed-centre inflow": (p_shift, replace(REF, lat_mode="shift")),
        "shifted belt, inflow follows belt": (p_shift, replace(REF, lat_mode="shift",
                                                               inflow_centre="shifted")),
        "decay tau = 8 yr": (replace(p, tau_yr=8.0), REF),
        "grid N_mu = 361": (replace(p, nmu=361), REF),
        "grid N_mu = 721": (replace(p, nmu=721), REF),
        "time step 0.5 day": (replace(p, dt_days=0.5), REF),
    }
    if QUICK:
        controls = {k: controls[k] for k in ("reference (2 m/s)",
                                             "shifted belt, inflow follows belt")}
    ctrl_rows = []
    for name, (pc, c) in controls.items():
        ac = linear_yield(pc)
        refine = name.startswith(("grid", "time"))
        grid_T = np.round(np.arange(1.15, 1.501, 0.05), 3) if refine else \
            np.round(np.arange(0.70, 2.301, 0.05), 3)
        if QUICK:
            grid_T = np.array([1.0, 1.25, 1.35, 1.6])
        bud = pool.submit(_job_budget, (pc, c, 1.0))
        tab = pool.submit(_job_remnant_table, (pc, c, 1.0, np.array([0.5, 1.5])))
        tabp = [pool.submit(_job_remnant_table, (pc, c, Tp, np.array([1.0])))
                for Tp in (0.5, 1.5)]
        rows, thr = stability_scan(pool, pc, c, ac, grid_T)
        P1 = float(_job_new_flux((pc, c, np.array([1.0])))[0])
        Plin = float(_job_new_flux((pc, replace(c, inflow_amp_m_s=0.0),
                                    np.array([1.0])))[0])
        (src, tnew, told, _, _), _fd = bud.result()
        rf = tab.result()
        rp = [x.result()[0] for x in tabp]
        row = dict(name=name, inflow_reduction_T1=1.0 - P1 / Plin,
                   ratio_inherited_to_new=told / tnew if tnew != 0.0 else np.nan,
                   dr_flow=rf[1] - rf[0], dr_profile=rp[1] - rp[0],
                   D0_full=thr["full"], D0_frozen=thr["frozen"],
                   D0_two_var=thr["two_var"], D0_scalar=thr["scalar"],
                   rel_frozen=thr["frozen"] / thr["full"] - 1.0,
                   rel_two_var=thr["two_var"] / thr["full"] - 1.0,
                   rel_scalar=thr["scalar"] / thr["full"] - 1.0)
        ctrl_rows.append(row)
        print(f"[5] {name}: inflow reduction at T=1 {100*row['inflow_reduction_T1']:.1f}%, "
              f"inherited/new transport term {row['ratio_inherited_to_new']:+.4f}, "
              f"dr flow {row['dr_flow']:+.4f}, dr profile {row['dr_profile']:+.4f}, "
              f"D0_pd complete {thr['full']:.4f} frozen {thr['frozen']:.4f} "
              f"settled-dipole {thr['two_var']:.4f} scalar {thr['scalar']:.4f} "
              f"(relative to complete: {100*row['rel_frozen']:+.3f}%, "
              f"{100*row['rel_two_var']:+.3f}%, {100*row['rel_scalar']:+.2f}%)")
    with open(os.path.join(OUTDIR, "controls.csv"), "w", encoding="utf-8") as f:
        keys = list(ctrl_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for row in ctrl_rows:
            f.write(",".join(f"\"{row[k]}\"" if k == "name" else f"{row[k]:.6g}"
                             for k in keys) + "\n")
    pool.shutdown()

    # Grid-convergence trend over N_mu = 181, 361, 721: ratio of successive
    # differences and the limit it implies if that ratio persists (a rough
    # guide only; the scheme is first order in space).
    by_name = {row["name"]: row for row in ctrl_rows}
    grids = ("reference (2 m/s)", "grid N_mu = 361", "grid N_mu = 721")
    if all(g in by_name for g in grids):
        for key in ("D0_full", "D0_scalar", "rel_scalar", "rel_frozen",
                    "dr_profile", "dr_flow", "ratio_inherited_to_new"):
            v = [by_name[g][key] for g in grids]
            d1, d2 = v[1] - v[0], v[2] - v[1]
            ratio = d2 / d1 if d1 != 0.0 else np.nan
            limit = v[2] + d2 * ratio / (1.0 - ratio) if 0.0 < ratio < 1.0 else np.nan
            print(f"[5] grid trend {key}: {v[0]:.5g}, {v[1]:.5g}, {v[2]:.5g}; ratio of "
                  f"successive differences {ratio:.2f}; implied limit {limit:.5g}")
            summary += [(f"grid_ratio[{key}]", ratio), (f"grid_limit[{key}]", limit)]

    with open(os.path.join(OUTDIR, "inflow_memory_results.csv"), "w", encoding="utf-8") as f:
        f.write("quantity,value\n")
        for key, val in summary:
            f.write(f"\"{key}\",{val:.6g}\n")

    make_figures(dict(p=p, mdl_ref=Model(p, REF), surv=surv, T_axis=T_axis,
                      budgets=budgets, stab=stab, bif=bif, n_keep=n_keep,
                      n_cyc=n_cyc, seq=seq, T_seq=T_seq, burn=burn))

    print(f"\nSaved figures and tables to {OUTDIR}/ "
          f"(total {time.time() - t_start:.0f} s)")
