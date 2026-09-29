"""Flash calculations: Rachford-Rice isothermal flash for any number of components with given K-values or with
Raoult's-law K-values from the Antoine table, bubble and dew points of mixtures, and an adiabatic flash with
constant heat capacities and latent heats.

>>> from engsep import flash
>>> res = flash.rachford_rice([0.5, 0.5], [2.0, 0.5])              # equimolar, K = 2 and 0.5
>>> round(res["V_over_F"], 4), [round(float(x), 4) for x in res["x"]]
(0.5, [0.3333, 0.6667])
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from . import vle


def rachford_rice(z, K) -> dict:
    """Solve sum z_i (K_i - 1) / (1 + psi (K_i - 1)) = 0 for the vapour fraction psi = V/F; returns psi, x, y."""
    z, K = np.asarray(z, float), np.asarray(K, float)
    f = lambda psi: np.sum(z * (K - 1) / (1 + psi * (K - 1)))  # noqa: E731
    if f(0.0) <= 0:  # below the bubble point
        return {"V_over_F": 0.0, "x": z, "y": z * K / np.sum(z * K), "state": "subcooled liquid"}
    if f(1.0) >= 0:  # above the dew point
        return {"V_over_F": 1.0, "x": z / K / np.sum(z / K), "y": z, "state": "superheated vapour"}
    lo = 1 / (1 - K.max()) + 1e-12 if K.max() > 1 else 0.0
    hi = 1 / (1 - K.min()) - 1e-12 if K.min() < 1 else 1.0
    psi = optimize.brentq(f, max(lo, 0.0), min(hi, 1.0))
    x = z / (1 + psi * (K - 1))
    return {"V_over_F": float(psi), "x": x, "y": K * x, "state": "two-phase"}


def flash_TP(names, z, T: float, P: float) -> dict:
    """Isothermal flash with Raoult's-law K-values."""
    K = np.array([vle.k_value(n, T, P) for n in names])
    out = rachford_rice(z, K)
    out["K"] = K
    return out


def bubble_point_T(names, z, P: float) -> float:
    z = np.asarray(z, float)
    f = lambda T: np.sum(z * np.array([vle.k_value(n, T, P) for n in names])) - 1  # noqa: E731
    Tb = [vle.boiling_point(n, P) for n in names]
    return float(optimize.brentq(f, min(Tb) - 50, max(Tb) + 5))


def dew_point_T(names, z, P: float) -> float:
    z = np.asarray(z, float)
    f = lambda T: np.sum(z / np.array([vle.k_value(n, T, P) for n in names])) - 1  # noqa: E731
    Tb = [vle.boiling_point(n, P) for n in names]
    return float(optimize.brentq(f, min(Tb) - 50, max(Tb) + 5))


def bubble_point_P(names, z, T: float) -> float:
    z = np.asarray(z, float)
    return float(np.sum(z * np.array([vle.antoine_pressure(n, T) for n in names])))


def dew_point_P(names, z, T: float) -> float:
    z = np.asarray(z, float)
    return float(1 / np.sum(z / np.array([vle.antoine_pressure(n, T) for n in names])))


def adiabatic_flash(names, z, T_feed: float, P: float, cp_liquid, dH_vap, T_ref: float | None = None) -> dict:
    """Adiabatic flash of a liquid feed at T_feed to pressure P: find T such that the enthalpy balance holds with
    constant liquid heat capacities cp_liquid (J/(mol K)) and latent heats dH_vap (J/mol):
    F cp_feed (T_feed - T) = V dH_vap(y-weighted)... solved as h_feed = (1 - psi) h_L(T) + psi h_V(T)."""
    z, cp, dH = np.asarray(z, float), np.asarray(cp_liquid, float), np.asarray(dH_vap, float)
    T_ref = T_feed if T_ref is None else T_ref

    def h_feed():
        return np.sum(z * cp) * (T_feed - T_ref)

    def residual(T):
        fl = flash_TP(names, z, T, P)
        psi, x, y = fl["V_over_F"], fl["x"], fl["y"]
        hL = np.sum(x * cp) * (T - T_ref)
        hV = np.sum(y * (cp * (T - T_ref) + dH))
        return (1 - psi) * hL + psi * hV - h_feed()

    T_low = min(vle.boiling_point(n, P) for n in names) - 60
    T = float(optimize.brentq(residual, T_low, T_feed + 1e-6))
    out = flash_TP(names, z, T, P)
    out["T"] = T
    return out
