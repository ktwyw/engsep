"""Adsorption and fixed beds: Langmuir, Freundlich and BET isotherms and their fitting; batch adsorber balances;
fixed-bed breakthrough by the ideal (shock) wave, the constant-pattern (Bohart-Adams / Thomas) model and the
length of unused bed (LUB) scale-up; the Klinkenberg solution for linear isotherms; and a pressure-swing
adsorption cycle balance.

>>> from engsep import adsorption as ad
>>> round(ad.stoichiometric_time(q0=2.0, c0=10.0, rho_b=500.0, eps=0.4, L=1.0, u=0.01))  # ideal breakthrough time (s)
10040
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

from ._common import trapezoid


# ------------------------------------------------------------------ isotherms
def langmuir(c, q_max: float, K: float):
    c = np.asarray(c, float)
    return q_max * K * c / (1 + K * c)


def freundlich(c, k: float, n: float):
    return k * np.asarray(c, float) ** (1 / n)


def bet(x, v_m: float, C: float):
    """BET: v = v_m C x / ((1 - x)(1 + (C - 1) x)) with x = p/p0."""
    x = np.asarray(x, float)
    return v_m * C * x / ((1 - x) * (1 + (C - 1) * x))


def fit_langmuir(c, q) -> dict:
    c, q = np.asarray(c, float), np.asarray(q, float)
    slope, intercept = np.polyfit(c, c / q, 1)  # c/q = c/q_max + 1/(K q_max)
    q0, K0 = 1 / slope, slope / intercept
    res = optimize.least_squares(lambda p: langmuir(c, p[0], p[1]) - q, [q0, K0])
    return {"q_max": float(res.x[0]), "K": float(res.x[1]), "rss": float(np.sum(res.fun**2))}


def fit_freundlich(c, q) -> dict:
    slope, intercept = np.polyfit(np.log(c), np.log(q), 1)
    return {"k": float(np.exp(intercept)), "n": float(1 / slope)}


def batch_adsorber(c0: float, V: float, m: float, isotherm) -> dict:
    """Equilibrium of a batch of solution (c0, volume V) with mass m of adsorbent: V (c0 - c) = m q(c)."""
    c = float(optimize.brentq(lambda cc: V * (c0 - cc) - m * isotherm(cc), 0.0, c0))
    return {"c": c, "q": float(isotherm(c)), "removal": 1 - c / c0}


# ------------------------------------------------------------------ fixed beds
def stoichiometric_time(q0: float, c0: float, rho_b: float, eps: float, L: float, u: float) -> float:
    """Ideal (shock-wave) breakthrough time: t* = L (eps + rho_b q0/c0) / u for a bed of length L, superficial
    velocity u, bulk density rho_b (kg/m3), loading q0 (mol/kg) in equilibrium with the feed c0 (mol/m3)."""
    return L * (eps + rho_b * q0 / c0) / u


def wave_velocity(q0: float, c0: float, rho_b: float, eps: float, u: float) -> float:
    return u / (eps + rho_b * q0 / c0)


def bohart_adams(t, c0: float, q0: float, rho_b: float, L: float, u: float, k: float):
    """Constant-pattern breakthrough curve for a favourable (rectangular-like) isotherm with a solid-phase
    rate constant k (m3/(mol s)): c/c0 = 1 / (1 + exp(k q0 rho_b L/u - k c0 t))."""
    t = np.asarray(t, float)
    arg = k * q0 * rho_b * L / u - k * c0 * t
    return 1 / (1 + np.exp(np.clip(arg, -700, 700)))


def klinkenberg(t, z, c0: float, K_lin: float, rho_b: float, eps: float, u: float, k_overall: float):
    """Klinkenberg's approximate solution for a linear isotherm q = K_lin c with an overall mass-transfer
    coefficient k_overall (1/s): c/c0 = 0.5 [1 + erf(sqrt(tau) - sqrt(xi) + (1/8)/sqrt(tau) + (1/8)/sqrt(xi))],
    with xi = k z (1 - eps)... dimensionless bed length and tau = k (t - z eps/u) the dimensionless time."""
    t, z = np.asarray(t, float), np.asarray(z, float)
    xi = k_overall * K_lin * rho_b * z / u
    tau = np.maximum(k_overall * (t - z * eps / u), 1e-12)
    return 0.5 * (1 + special.erf(np.sqrt(tau) - np.sqrt(xi) + 1 / (8 * np.sqrt(tau)) + 1 / (8 * np.sqrt(xi))))


def breakthrough_analysis(t, c_over_c0, c_break: float = 0.05, c_exhaust: float = 0.95) -> dict:
    """From a measured breakthrough curve: breakthrough and exhaustion times, the stoichiometric time (equal-area
    point), the used-bed fraction and the length of unused bed as a fraction of the bed."""
    t, c = np.asarray(t, float), np.asarray(c_over_c0, float)
    t_b = float(np.interp(c_break, c, t))
    t_e = float(np.interp(c_exhaust, c, t))
    t_star = float(trapezoid(1 - np.clip(c, 0, 1), t))  # stoichiometric time = area above the curve
    return {
        "t_break": t_b,
        "t_exhaust": t_e,
        "t_stoich": t_star,
        "used_fraction": t_b / t_star,
        "LUB_fraction": 1 - t_b / t_star,
    }


def lub_scale_up(LUB: float, L_lab: float, t_break_lab: float, L_new: float) -> dict:
    """The length of unused bed is independent of bed length (constant pattern): the breakthrough time of a longer
    bed scales with the used length."""
    t_star_lab = t_break_lab * L_lab / (L_lab - LUB)
    t_break_new = t_star_lab * (L_new - LUB) / L_lab
    return {"t_break_new": t_break_new, "used_fraction_new": (L_new - LUB) / L_new}


def psa_cycle(q_high: float, q_low: float, rho_b: float, V_bed: float, y_feed: float, c_feed: float) -> dict:
    """A simple pressure-swing balance: working capacity per cycle Delta q = q(p_high) - q(p_low), feed that can be
    processed per cycle, and the fraction of feed adsorbed."""
    working = (q_high - q_low) * rho_b * V_bed
    return {"working_capacity_mol": working, "feed_volume_per_cycle": working / c_feed if c_feed else np.inf}
