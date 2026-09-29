"""Membrane separations: gas permeation by the solution-diffusion model (permeance, selectivity, the
complete-mixing and cross-flow models for a binary), reverse osmosis (osmotic pressure by van 't Hoff, water
and salt flux, rejection, recovery, concentration polarisation), ultrafiltration (gel-polarisation limiting
flux) and dialysis.

>>> from engsep import membranes as mb
>>> round(mb.osmotic_pressure(c_ions=2 * 0.6, T=298.15) / 1e5, 1)      # seawater-like: 0.6 M NaCl -> about 29 bar
29.7
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from .constants import R


# ------------------------------------------------------------------ gas permeation
def permeance(permeability: float, thickness: float) -> float:
    """Permeance (mol/(m2 s Pa)) = permeability / thickness."""
    return permeability / thickness


def selectivity(perm_A: float, perm_B: float) -> float:
    return perm_A / perm_B


def gas_flux(Q: float, p_feed: float, x: float, p_perm: float, y: float) -> float:
    """Flux of a species through the membrane: Q (p_feed x - p_perm y)."""
    return Q * (p_feed * x - p_perm * y)


def complete_mixing(alpha: float, x_f: float, pressure_ratio: float, theta: float) -> dict:
    """Binary gas separation with complete mixing on both sides (Weller-Steiner): given the ideal selectivity
    alpha = Q_A/Q_B, feed mole fraction x_f, pressure ratio r = p_perm/p_feed and stage cut theta, find the
    permeate y and retentate x from the flux ratio and the material balance."""
    r = pressure_ratio

    def residual(y):
        x = (x_f - theta * y) / (1 - theta)
        if not (0 < x < 1):
            return 1e3
        # flux ratio: y/(1-y) = alpha (x - r y)/((1 - x) - r (1 - y))
        return y / (1 - y) - alpha * (x - r * y) / ((1 - x) - r * (1 - y))

    y = float(optimize.brentq(residual, x_f + 1e-9, 1 - 1e-9))
    x = (x_f - theta * y) / (1 - theta)
    return {"y_permeate": y, "x_retentate": x}


def cross_flow(alpha: float, x_f: float, pressure_ratio: float, theta: float, n: int = 2000) -> dict:
    """Cross-flow (plug flow on the feed side, permeate withdrawn locally) by integrating the local permeate
    composition along the membrane until the stage cut is reached."""
    r = pressure_ratio

    def local_y(x):
        f = lambda y: y / (1 - y) - alpha * (x - r * y) / ((1 - x) - r * (1 - y))  # noqa: E731
        return float(optimize.brentq(f, x + 1e-12 if x + 1e-12 < 1 else x, 1 - 1e-12))

    # retentate flow L (per unit feed) and composition x along the membrane, V = permeate withdrawn so far:
    # dL/dV = -1 and, from d(L x)/dV = -y, dx/dV = -(y - x)/L
    sol = integrate.solve_ivp(
        lambda v, s: [-1.0, -(local_y(s[1]) - s[1]) / s[0]],
        (0, theta),
        [1.0, x_f],
        rtol=1e-9,
        atol=1e-12,
        max_step=theta / 200,
    )
    L_out, x_out = sol.y[0][-1], sol.y[1][-1]
    y_avg = (x_f - L_out * x_out) / theta
    return {"y_permeate": float(y_avg), "x_retentate": float(x_out)}


def membrane_area(
    feed_flow: float, theta: float, y: float, Q_A: float, p_feed: float, x_avg: float, p_perm: float
) -> float:
    """Area from the permeate flow of A and the average driving force (complete-mixing estimate)."""
    return feed_flow * theta * y / (Q_A * (p_feed * x_avg - p_perm * y))


# ------------------------------------------------------------------ reverse osmosis
def osmotic_pressure(c_ions: float, T: float) -> float:
    """van 't Hoff: pi = c R T with c the total molar concentration of dissolved particles (mol/m3 -> here mol/L
    converted internally: pass c in mol/L)."""
    return c_ions * 1000 * R * T


def ro_water_flux(A: float, dP: float, d_pi: float) -> float:
    """J_w = A (Delta P - Delta pi), A the water permeability (m/(s Pa) or as given)."""
    return A * (dP - d_pi)


def ro_salt_flux(B: float, c_feed: float, c_perm: float) -> float:
    return B * (c_feed - c_perm)


def ro_permeate_concentration(
    A: float, B: float, dP: float, c_feed: float, T: float, ions: int = 2, iters: int = 50
) -> dict:
    """Solve the coupled water and salt fluxes for the permeate concentration: c_p = J_s/J_w with Delta pi from
    c_feed - c_p (ideal solution)."""
    c_p = 0.0
    for _ in range(iters):
        d_pi = osmotic_pressure(ions * (c_feed - c_p), T)
        J_w = ro_water_flux(A, dP, d_pi)
        J_s = ro_salt_flux(B, c_feed, c_p)
        c_p_new = J_s / J_w
        if abs(c_p_new - c_p) < 1e-12:
            break
        c_p = c_p_new
    return {"c_permeate": c_p, "rejection": 1 - c_p / c_feed, "J_w": J_w, "J_s": J_s}


def ro_recovery(c_feed: float, dP: float, T: float, ions: int = 2) -> float:
    """Maximum recovery before the concentrate's osmotic pressure equals the applied pressure (complete rejection)."""
    pi_f = osmotic_pressure(ions * c_feed, T)
    return max(0.0, 1 - pi_f / dP)


def concentration_polarisation(J_w: float, k_mass: float) -> float:
    """Film model: c_membrane/c_bulk = exp(J_w / k) for complete rejection."""
    return float(np.exp(J_w / k_mass))


# ------------------------------------------------------------------ ultrafiltration and dialysis
def uf_limiting_flux(k_mass: float, c_gel: float, c_bulk: float) -> float:
    """Gel-polarisation model: J = k ln(c_gel / c_bulk), independent of pressure once the gel forms."""
    return k_mass * np.log(c_gel / c_bulk)


def dialysis_flux(P_m: float, c1: float, c2: float) -> float:
    """Solute flux across a dialysis membrane: P_m (c1 - c2)."""
    return P_m * (c1 - c2)


def dialysis_countercurrent(P_m: float, area: float, Q1: float, Q2: float, c1_in: float, c2_in: float = 0.0) -> dict:
    """Countercurrent dialyser with plug flow on both sides (heat-exchanger analogy): NTU = P_m A/Q1, R = Q1/Q2."""
    NTU, Rr = P_m * area / Q1, Q1 / Q2
    if abs(Rr - 1) < 1e-12:
        eff = NTU / (1 + NTU)
    else:
        eff = (1 - np.exp(-NTU * (1 - Rr))) / (1 - Rr * np.exp(-NTU * (1 - Rr)))
    c1_out = c1_in - eff * (c1_in - c2_in)
    return {"c1_out": c1_out, "removal": (c1_in - c1_out) / c1_in, "NTU": NTU, "effectiveness": eff}
