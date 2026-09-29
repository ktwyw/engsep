"""Vapour-liquid equilibrium: an Antoine table (log10 P/mmHg = A - B/(C + T/degC), Lange's Handbook), Raoult's
law and relative volatility, Wilson activity coefficients for common non-ideal binaries (Perry's parameters),
K-values, bubble and dew points, T-x-y and x-y diagrams, and azeotrope location.

>>> from engsep import vle
>>> round(vle.boiling_point("water") - 273.15, 1)
100.0
>>> round(vle.azeotrope("ethanol", "water")["x"], 2)          # ethanol-water azeotrope, 89 mol % ethanol
0.89
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .constants import CAL, MMHG, R

# name: (A, B, C, valid range degC, molar volume cm3/mol at 25 degC)   Antoine in mmHg / degC (Lange's Handbook, Dean)
ANTOINE: dict = {
    "water": (8.07131, 1730.63, 233.426, (1, 100), 18.07),
    "methanol": (8.08097, 1582.271, 239.726, (-16, 91), 40.73),
    "ethanol": (8.20417, 1642.89, 230.300, (-57, 80), 58.68),
    "1-propanol": (7.84767, 1499.21, 204.64, (2, 120), 75.14),
    "2-propanol": (8.11778, 1580.92, 219.61, (0, 113), 76.92),
    "1-butanol": (7.36366, 1305.198, 173.427, (89, 126), 91.97),
    "acetone": (7.11714, 1210.595, 229.664, (-13, 55), 74.05),
    "2-butanone": (7.06356, 1261.34, 221.97, (-6, 80), 90.17),
    "ethyl acetate": (7.10179, 1244.95, 217.88, (-20, 150), 98.49),
    "methyl acetate": (7.06524, 1157.63, 219.73, (-38, 57), 79.84),
    "acetic acid": (7.38782, 1533.313, 222.309, (0, 36), 57.54),
    "diethyl ether": (6.92032, 1064.07, 228.80, (-61, 20), 104.8),
    "1,4-dioxane": (7.43155, 1554.68, 240.34, (20, 105), 85.7),
    "chloroform": (6.4934, 929.44, 196.03, (-35, 61), 80.67),
    "carbon tetrachloride": (6.93390, 1242.43, 230.0, (-20, 77), 97.1),
    "benzene": (6.90565, 1211.033, 220.790, (8, 103), 89.41),
    "toluene": (6.95464, 1344.8, 219.482, (6, 137), 106.85),
    "ethylbenzene": (6.95719, 1424.255, 213.206, (26, 164), 123.1),
    "o-xylene": (6.99891, 1474.679, 213.686, (32, 172), 121.2),
    "p-xylene": (6.99052, 1453.430, 215.307, (27, 166), 123.9),
    "styrene": (7.14016, 1574.51, 224.09, (32, 82), 115.6),
    "cyclohexane": (6.84130, 1201.53, 222.65, (7, 81), 108.75),
    "propane": (6.80338, 803.81, 246.99, (-108, -25), 84.0),
    "n-butane": (6.80896, 935.86, 238.73, (-77, 19), 101.4),
    "isobutane": (6.74808, 882.80, 240.00, (-83, 7), 105.5),
    "n-pentane": (6.87632, 1075.78, 233.205, (-50, 58), 116.1),
    "n-hexane": (6.87601, 1171.17, 224.41, (-25, 92), 131.6),
    "n-heptane": (6.89677, 1264.90, 216.54, (-2, 124), 147.4),
    "n-octane": (6.91868, 1351.99, 209.15, (19, 152), 163.5),
    "n-decane": (6.94365, 1495.17, 193.86, (58, 203), 195.9),
    "ammonia": (7.36050, 926.132, 240.17, (-83, 60), 25.0),
}

# Wilson binary parameters (cal/mol): (a12, a21) for (component 1, component 2), regressed to 1 atm VLE data
# (DECHEMA-style values as tabulated in Perry's and in Seader, Henley & Roper); indicative, +/- 1-2 % on azeotropes
WILSON: dict = {
    ("ethanol", "water"): (325.08, 953.28),
    ("methanol", "water"): (107.38, 469.55),
    ("2-propanol", "water"): (437.98, 1238.00),
    ("acetone", "water"): (291.27, 1448.01),
    ("1-propanol", "water"): (906.53, 1396.64),
    ("acetone", "methanol"): (-161.88, 583.11),
    ("ethyl acetate", "ethanol"): (322.94, 224.65),
    ("benzene", "toluene"): (0.0, 0.0),
}


def antoine_pressure(name: str, T: float) -> float:
    """Vapour pressure in Pa at temperature T (K)."""
    A, B, C, _, _ = ANTOINE[name]
    return 10 ** (A - B / (C + T - 273.15)) * MMHG


def antoine_temperature(name: str, P: float) -> float:
    """Temperature (K) at which the vapour pressure equals P (Pa)."""
    A, B, C, _, _ = ANTOINE[name]
    return B / (A - np.log10(P / MMHG)) - C + 273.15


def boiling_point(name: str, P: float = 101325.0) -> float:
    return float(antoine_temperature(name, P))


def wilson_lambda(pair, T: float):
    """Lambda_12, Lambda_21 from the energy parameters and the molar volumes."""
    c1, c2 = pair
    key = (c1, c2) if (c1, c2) in WILSON else (c2, c1)
    a12, a21 = WILSON[key]
    if key != (c1, c2):
        a12, a21 = a21, a12
    V1, V2 = ANTOINE[c1][4], ANTOINE[c2][4]
    L12 = V2 / V1 * np.exp(-a12 * CAL / (R * T))
    L21 = V1 / V2 * np.exp(-a21 * CAL / (R * T))
    return L12, L21


def wilson_gamma(pair, x1, T: float):
    """Activity coefficients (gamma_1, gamma_2) of a binary from the Wilson equation."""
    x1 = np.asarray(x1, float)
    x2 = 1 - x1
    L12, L21 = wilson_lambda(pair, T)
    s1, s2 = x1 + L12 * x2, x2 + L21 * x1
    term = L12 / s1 - L21 / s2
    g1 = np.exp(-np.log(s1) + x2 * term)
    g2 = np.exp(-np.log(s2) - x1 * term)
    return g1, g2


def has_wilson(pair) -> bool:
    return tuple(pair) in WILSON or tuple(pair)[::-1] in WILSON


def gammas(pair, x1, T: float, ideal: bool = False):
    if ideal or not has_wilson(pair):
        x1 = np.asarray(x1, float)
        return np.ones_like(x1), np.ones_like(x1)
    return wilson_gamma(pair, x1, T)


# ------------------------------------------------------------------ bubble and dew points of a binary
def bubble_T(pair, x1, P: float = 101325.0, ideal: bool = False) -> float:
    """Bubble temperature (K) of a binary liquid of mole fraction x1 (modified Raoult's law)."""
    c1, c2 = pair

    def f(T):
        g1, g2 = gammas(pair, x1, T, ideal)
        return x1 * g1 * antoine_pressure(c1, T) + (1 - x1) * g2 * antoine_pressure(c2, T) - P

    lo, hi = min(boiling_point(c1, P), boiling_point(c2, P)) - 60, max(boiling_point(c1, P), boiling_point(c2, P)) + 5
    return float(optimize.brentq(f, lo, hi))


def bubble_y(pair, x1, P: float = 101325.0, ideal: bool = False) -> tuple:
    """(T, y1) at the bubble point."""
    T = bubble_T(pair, x1, P, ideal)
    g1, _ = gammas(pair, x1, T, ideal)
    return T, float(x1 * g1 * antoine_pressure(pair[0], T) / P)


def dew_T(pair, y1, P: float = 101325.0, ideal: bool = False) -> float:
    """Dew temperature (K) of a binary vapour of mole fraction y1."""
    c1, c2 = pair
    if ideal or not has_wilson(pair):
        f = lambda T: y1 * P / antoine_pressure(c1, T) + (1 - y1) * P / antoine_pressure(c2, T) - 1  # noqa: E731
        lo, hi = (
            min(boiling_point(c1, P), boiling_point(c2, P)) - 60,
            max(boiling_point(c1, P), boiling_point(c2, P)) + 5,
        )
        return float(optimize.brentq(f, lo, hi))
    # non-ideal: find x1 such that bubble_y = y1, T from the bubble point
    x = optimize.brentq(lambda x1: bubble_y(pair, x1, P)[1] - y1, 1e-9, 1 - 1e-9)
    return bubble_T(pair, x, P)


def Txy(pair, P: float = 101325.0, n: int = 51, ideal: bool = False) -> dict:
    x = np.linspace(0, 1, n)
    T, y = zip(*[bubble_y(pair, xi, P, ideal) for xi in x])
    return {"x": x, "y": np.array(y), "T": np.array(T)}


def xy(pair, P: float = 101325.0, n: int = 51, ideal: bool = False) -> dict:
    d = Txy(pair, P, n, ideal)
    return {"x": d["x"], "y": d["y"]}


def equilibrium_y(pair, x1, P: float = 101325.0, ideal: bool = False):
    """y1 in equilibrium with x1 (vectorised over x1)."""
    x1 = np.atleast_1d(np.asarray(x1, float))
    out = np.array([bubble_y(pair, xi, P, ideal)[1] for xi in x1])
    return out if out.size > 1 else float(out[0])


# ------------------------------------------------------------------ relative volatility and constant-alpha equilibrium
def relative_volatility(pair, T: float, x1: float = 0.5, ideal: bool = False) -> float:
    """alpha_12 = (gamma_1 P1sat)/(gamma_2 P2sat) at T (and composition x1 for non-ideal pairs)."""
    g1, g2 = gammas(pair, x1, T, ideal)
    return float(g1 * antoine_pressure(pair[0], T) / (g2 * antoine_pressure(pair[1], T)))


def geometric_mean_alpha(pair, P: float = 101325.0, ideal: bool = False) -> float:
    """Geometric mean of alpha at the two pure-component boiling points (the usual constant-alpha estimate)."""
    T1, T2 = boiling_point(pair[0], P), boiling_point(pair[1], P)
    return float(np.sqrt(relative_volatility(pair, T1, 1.0, ideal) * relative_volatility(pair, T2, 0.0, ideal)))


def y_from_alpha(x, alpha: float):
    """Constant-relative-volatility equilibrium: y = alpha x / (1 + (alpha - 1) x)."""
    x = np.asarray(x, float)
    return alpha * x / (1 + (alpha - 1) * x)


def x_from_alpha(y, alpha: float):
    y = np.asarray(y, float)
    return y / (alpha - (alpha - 1) * y)


# ------------------------------------------------------------------ azeotropes and K-values
def azeotrope(c1: str, c2: str, P: float = 101325.0) -> dict | None:
    """Azeotrope (x = y) of a non-ideal binary, or None."""
    pair = (c1, c2)
    f = lambda x1: bubble_y(pair, x1, P)[1] - x1  # noqa: E731
    xs = np.linspace(0.02, 0.98, 49)
    vals = [f(x) for x in xs]
    for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if fa * fb < 0:
            x_az = float(optimize.brentq(f, a, b))
            return {
                "x": x_az,
                "T": bubble_T(pair, x_az, P),
                "kind": "minimum-boiling"
                if bubble_T(pair, x_az, P) < min(boiling_point(c1, P), boiling_point(c2, P))
                else "maximum-boiling",
            }
    return None


def k_value(name: str, T: float, P: float, gamma: float = 1.0) -> float:
    """K = gamma Psat / P (modified Raoult's law)."""
    return gamma * antoine_pressure(name, T) / P
