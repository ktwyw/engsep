"""Absorption and stripping: the Kremser equation for dilute countercurrent cascades, stage-by-stage
calculations with a linear or curved equilibrium line, minimum solvent and stripping-gas rates, packed-tower
design by the height of a transfer unit (HTU/NTU, overall gas- or liquid-phase basis), the Colburn
equation for dilute systems, and the two-film theory relating overall to individual coefficients.

>>> from engsep import absorption as ab
>>> round(ab.kremser_fraction_absorbed(A=1.4, N=6), 4)      # absorption factor 1.4, six stages
0.9581
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ._common import trapezoid


# ------------------------------------------------------------------ Kremser
def absorption_factor(L: float, V: float, m: float) -> float:
    """A = L/(m V) for the equilibrium line y = m x (molar flows of solvent and carrier gas)."""
    return L / (m * V)


def kremser_fraction_absorbed(A: float, N: int) -> float:
    """Fraction of solute absorbed in N equilibrium stages with pure solvent: (A^(N+1) - A)/(A^(N+1) - 1)."""
    if abs(A - 1) < 1e-12:
        return N / (N + 1)
    return (A ** (N + 1) - A) / (A ** (N + 1) - 1)


def kremser_stages(A: float, fraction_absorbed: float) -> float:
    """Stages for a given recovery with pure solvent: from (A^(N+1) - A)/(A^(N+1) - 1) = phi,
    A^(N+1) = (A - phi)/(1 - phi), so N = ln[(A - phi)/(1 - phi)]/ln A - 1 (A != 1); N = phi/(1 - phi) for A = 1."""
    phi = fraction_absorbed
    if A < 1 and phi >= A:
        raise ValueError(
            f"A fraction of {phi} cannot be absorbed with A = {A} < 1: the limit with infinite stages is A."
        )
    if abs(A - 1) < 1e-12:
        return phi / (1 - phi)
    return _kremser_N(A, phi)


def _kremser_N(A: float, phi: float) -> float:
    """N from (A^(N+1) - A)/(A^(N+1) - 1) = phi  ->  A^(N+1) = (A - phi)/(1 - phi)."""
    return float(np.log((A - phi) / (1 - phi)) / np.log(A) - 1)


def kremser_general(A: float, N: int, y_in: float, x_in: float, m: float) -> dict:
    """Kremser with solute in the entering solvent: (y_in - y_out)/(y_in - m x_in) = (A^(N+1) - A)/(A^(N+1) - 1)."""
    phi = kremser_fraction_absorbed(A, N)
    y_out = y_in - phi * (y_in - m * x_in)
    return {"y_out": y_out, "fraction_absorbed": (y_in - y_out) / y_in}


def stripping_factor(L: float, V: float, m: float) -> float:
    return m * V / L


def kremser_fraction_stripped(S: float, N: int) -> float:
    return kremser_fraction_absorbed(S, N)


def minimum_solvent(V: float, y_in: float, y_out: float, m: float, x_in: float = 0.0) -> float:
    """L_min for absorption with a linear equilibrium line: pinch at the rich end, x_out = y_in/m."""
    x_out_max = y_in / m
    return V * (y_in - y_out) / (x_out_max - x_in)


# ------------------------------------------------------------------ stage-by-stage with any equilibrium
def _x_in_equilibrium(y_eq_of_x, y: float) -> float:
    return float(optimize.brentq(lambda xx: y_eq_of_x(xx) - y, 0.0, 10.0))


def stages_absorber(
    y_eq_of_x, L: float, V: float, y_in: float, y_out: float, x_in: float = 0.0, max_stages: int = 200
) -> dict:
    """Countercurrent absorber stepped from the lean end (top): the gas leaving stage 1 is y_out; each stage
    is in equilibrium, and the operating line (mole-ratio or dilute mole-fraction form) is y = y_out + (L/V)(x - x_in).
    Returns the stages needed to reduce the gas from y_in to y_out and the stair-step points."""
    x, y = x_in, y_out
    xs, ys, n = [x], [y], 0
    while y < y_in and n < max_stages:
        x = _x_in_equilibrium(y_eq_of_x, y)  # liquid leaving stage n, in equilibrium with the gas leaving it
        n += 1
        xs.append(x)
        ys.append(y)
        y = y_out + L / V * (x - x_in)  # gas entering stage n from below
        xs.append(x)
        ys.append(y)
    return {"stages": n, "x": np.array(xs), "y": np.array(ys), "x_out": x}


# ------------------------------------------------------------------ packed towers
def ntu_gas(y_in: float, y_out: float, y_star_of_x, x_of_y, n: int = 400) -> float:
    """Overall gas-phase number of transfer units N_OG = int dy/(y - y*) along the operating line
    (x_of_y gives the liquid composition at the same height, y_star_of_x the equilibrium gas composition)."""
    ys = np.linspace(y_out, y_in, n)
    integrand = 1 / (ys - np.array([y_star_of_x(x_of_y(y)) for y in ys]))
    return float(trapezoid(integrand, ys))


def ntu_colburn(y_in: float, y_out: float, m: float, x_in: float, A: float) -> float:
    """Colburn's equation for dilute systems with a linear equilibrium line:
    N_OG = ln[(1 - 1/A)(y_in - m x_in)/(y_out - m x_in) + 1/A] / (1 - 1/A)."""
    if abs(A - 1) < 1e-12:
        return (y_in - y_out) / (y_out - m * x_in)
    return float(np.log((1 - 1 / A) * (y_in - m * x_in) / (y_out - m * x_in) + 1 / A) / (1 - 1 / A))


def htu_gas(V_flux: float, K_y_a: float) -> float:
    """H_OG = G_M / (K_y a) with G_M the molar gas flux (mol/(m2 s)) and K_y a in mol/(m3 s)."""
    return V_flux / K_y_a


def packed_height(HTU: float, NTU: float) -> float:
    return HTU * NTU


def overall_from_film(k_y: float, k_x: float, m: float) -> dict:
    """Two-film theory: 1/K_y = 1/k_y + m/k_x; 1/K_x = 1/k_x + 1/(m k_y); the fraction of the resistance in the gas."""
    K_y = 1 / (1 / k_y + m / k_x)
    K_x = 1 / (1 / k_x + 1 / (m * k_y))
    return {"K_y": K_y, "K_x": K_x, "gas_resistance_fraction": (1 / k_y) / (1 / K_y)}


def htu_overall_from_individual(H_G: float, H_L: float, A: float) -> float:
    """H_OG = H_G + H_L / A."""
    return H_G + H_L / A


def stripping_tower_ntu(x_in: float, x_out: float, m: float, y_in: float, S: float) -> float:
    """N_OL for stripping with a linear equilibrium line (Colburn form in the liquid phase)."""
    if abs(S - 1) < 1e-12:
        return (x_in - x_out) / (x_out - y_in / m)
    return float(np.log((1 - 1 / S) * (x_in - y_in / m) / (x_out - y_in / m) + 1 / S) / (1 - 1 / S))


def henry_slope(H: float, P: float) -> float:
    """m = H/P for y = (H/P) x with Henry's constant H in the same pressure units as P."""
    return H / P
