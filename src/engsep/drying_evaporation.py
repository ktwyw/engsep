"""Drying and evaporation: psychrometric relations for air-water (humidity, saturation, dew point, adiabatic
saturation, wet-bulb approximation), drying-rate curves with constant and falling-rate periods, the time to
dry a batch, and single- and multiple-effect evaporators with boiling-point rise and steam economy.

>>> from engsep import drying_evaporation as de
>>> round(de.humidity_saturation(T_C=30.0), 4)       # kg water per kg dry air at 30 degC, 1 atm (tables: 0.0273)
0.0271
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from . import vle

CP_AIR, CP_VAPOUR, CP_WATER = 1005.0, 1880.0, 4186.0  # J/(kg K)
LATENT_0 = 2.501e6  # J/kg at 0 degC


def latent_heat(T_C: float) -> float:
    """Latent heat of water (J/kg), linear in T (Watson-like fit, 0-100 degC within 0.5 %)."""
    return LATENT_0 - 2.36e3 * T_C


def humidity(p_w: float, P: float = 101325.0) -> float:
    """Absolute humidity (kg water per kg dry air) from the partial pressure of water."""
    return 0.622 * p_w / (P - p_w)


def humidity_saturation(T_C: float, P: float = 101325.0) -> float:
    return humidity(vle.antoine_pressure("water", T_C + 273.15), P)


def relative_humidity(H: float, T_C: float, P: float = 101325.0) -> float:
    p_w = H * P / (0.622 + H)
    return p_w / vle.antoine_pressure("water", T_C + 273.15)


def dew_point(H: float, P: float = 101325.0) -> float:
    p_w = H * P / (0.622 + H)
    return vle.antoine_temperature("water", p_w) - 273.15


def humid_heat(H: float) -> float:
    return CP_AIR + CP_VAPOUR * H


def enthalpy_moist_air(T_C: float, H: float) -> float:
    """J per kg dry air, reference 0 degC."""
    return humid_heat(H) * T_C + H * LATENT_0


def adiabatic_saturation_temperature(T_C: float, H: float, P: float = 101325.0) -> float:
    """T_as from the adiabatic-saturation balance cs (T - T_as) = lambda(T_as) (H_as - H)."""
    f = lambda Ts: humid_heat(H) * (T_C - Ts) - latent_heat(Ts) * (humidity_saturation(Ts, P) - H)  # noqa: E731
    return float(optimize.brentq(f, -20.0, T_C + 1e-9))


def wet_bulb_temperature(T_C: float, H: float, P: float = 101325.0) -> float:
    """For air-water the Lewis relation makes the wet-bulb and adiabatic-saturation temperatures nearly equal."""
    return adiabatic_saturation_temperature(T_C, H, P)


# ------------------------------------------------------------------ drying
def drying_time(X0: float, X_final: float, X_c: float, X_eq: float, R_c: float, A_over_ms: float) -> dict:
    """Batch drying with a constant-rate period at R_c (kg/(m2 s)) down to the critical moisture X_c and a
    linear falling-rate period to the equilibrium moisture X_eq (moisture contents on a dry basis, kg/kg;
    A_over_ms the exposed area per kg of dry solid)."""
    t_c = max(X0 - max(X_c, X_final), 0.0) / (R_c * A_over_ms)
    if X_final >= X_c:
        return {"t_constant": t_c, "t_falling": 0.0, "t_total": t_c}
    t_f = (X_c - X_eq) / (R_c * A_over_ms) * np.log((X_c - X_eq) / (X_final - X_eq))
    return {"t_constant": t_c, "t_falling": t_f, "t_total": t_c + t_f}


def constant_rate(h: float, T_air: float, T_wet: float, T_C_latent: float | None = None) -> float:
    """Constant drying rate from the heat-transfer coefficient h (W/(m2 K)): R_c = h (T_air - T_wb)/lambda."""
    return h * (T_air - T_wet) / latent_heat(T_wet if T_C_latent is None else T_C_latent)


def drying_curve_analysis(t, X, min_points: int = 4, tolerance: float = 3.0, noise: float = 0.015) -> dict:
    """From a measured moisture-time curve: the constant-rate period is the longest leading segment that a straight
    line fits to within `tolerance` standard deviations (the standard deviation is floored at `noise` times the
    initial moisture, the measurement's relative precision); the critical point is where that line meets a line
    through the next points (the start of the falling-rate period); the constant rate is the leading slope.
    Also returns point-wise rates for plotting."""
    t, X = np.asarray(t, float), np.asarray(X, float)
    n = len(t)
    k = min_points
    while k < n:
        slope, intercept = np.polyfit(t[:k], X[:k], 1)
        resid = X[:k] - (slope * t[:k] + intercept)
        sigma = max(np.std(resid, ddof=2), noise * abs(X[0]))
        if abs(X[k] - (slope * t[k] + intercept)) > tolerance * sigma:
            break
        k += 1
    slope, intercept = np.polyfit(t[:k], X[:k], 1)
    j = min(k + 4, n)
    if j - k >= 3:  # intersection of the constant-rate line with the first falling-rate segment
        s2, i2 = np.polyfit(t[k:j], X[k:j], 1)
        t_c = (i2 - intercept) / (slope - s2) if abs(slope - s2) > 1e-15 else t[k - 1]
        t_c = float(np.clip(t_c, t[0], t[k]))
    else:
        t_c = float(t[k - 1])
    rate = -np.gradient(X, t)
    return {
        "rate": rate,
        "R_c": float(-slope),
        "X_critical": float(slope * t_c + intercept),
        "t_critical": t_c,
        "n_constant_points": int(k),
    }


# ------------------------------------------------------------------ evaporation
def single_effect(
    feed_kg_s: float,
    x_feed: float,
    x_product: float,
    T_feed_C: float,
    T_boil_C: float,
    T_steam_C: float,
    U: float,
    cp_feed: float = 4000.0,
    bpr: float = 0.0,
) -> dict:
    """Single-effect evaporator: solvent evaporated, steam required (saturated steam condensing at T_steam), heat
    duty, area and steam economy. The vapour leaves at the boiling point T_boil + BPR."""
    V = feed_kg_s * (1 - x_feed / x_product)
    L = feed_kg_s - V
    T_v = T_boil_C + bpr
    Q = feed_kg_s * cp_feed * (T_v - T_feed_C) + V * latent_heat(T_v)
    steam = Q / latent_heat(T_steam_C)
    dT = T_steam_C - T_v
    return {
        "vapour_kg_s": V,
        "product_kg_s": L,
        "Q_W": Q,
        "steam_kg_s": steam,
        "economy": V / steam,
        "area_m2": Q / (U * dT),
        "dT_driving": dT,
    }


def multiple_effect(
    n_effects: int,
    feed_kg_s: float,
    x_feed: float,
    x_product: float,
    T_feed_C: float,
    T_steam_C: float,
    T_last_C: float,
    U_list,
    cp_feed: float = 4000.0,
    bpr_list=None,
    iters: int = 100,
) -> dict:
    """Forward-feed multiple-effect evaporator with equal areas: distribute the overall temperature drop in
    proportion to 1/U, solve the effect balances, then re-distribute the drops until the areas agree."""
    U = np.asarray(U_list, float)
    bpr = np.zeros(n_effects) if bpr_list is None else np.asarray(bpr_list, float)
    V_total = feed_kg_s * (1 - x_feed / x_product)
    total_dT = T_steam_C - T_last_C - bpr.sum()
    dT = total_dT * (1 / U) / np.sum(1 / U)
    for _ in range(iters):
        T_boil = [T_steam_C - dT[0] - bpr[0]]
        for i in range(1, n_effects):
            T_boil.append(T_boil[-1] - dT[i] - bpr[i])  # boiling temperature of effect i (vapour of i-1 heats it)
        # effect balances: with the vapour split V_i, effect i needs Q_i = L_{i-1} cp (T_v,i - T_{i-1}) + V_i lam_i,
        # supplied by the vapour of the previous effect (V_{i-1} lam_{i-1} = Q_i); iterate the split to consistency
        V = np.full(n_effects, V_total / n_effects)
        for _ in range(iters):
            L_prev, T_prev = feed_kg_s, T_feed_C
            Q = np.zeros(n_effects)
            for i in range(n_effects):
                T_v = T_boil[i] + bpr[i]
                Q[i] = L_prev * cp_feed * (T_v - T_prev) + V[i] * latent_heat(T_v)
                L_prev, T_prev = L_prev - V[i], T_v
            V_upd = V.copy()
            for i in range(1, n_effects):
                V_upd[i - 1] = Q[i] / latent_heat(T_boil[i - 1] + bpr[i - 1])
            V_upd[-1] = V_total - V_upd[:-1].sum()
            if np.max(np.abs(V_upd - V)) < 1e-9:
                V = V_upd
                break
            V = 0.5 * (V + V_upd)
        steam = Q[0] / latent_heat(T_steam_C)
        areas = Q / (U * dT)
        A_mean = areas.mean()
        if np.max(np.abs(areas - A_mean)) / A_mean < 1e-6:
            break
        dT = dT * areas / A_mean  # re-distribute drops in proportion to the area error
        dT = dT * total_dT / dT.sum()
    return {
        "vapour_kg_s": V,
        "steam_kg_s": steam,
        "economy": V.sum() / steam,
        "areas_m2": areas,
        "T_boil_C": np.array(T_boil),
        "dT": dT,
        "Q_W": Q,
    }
