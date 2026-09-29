"""Crystallization: solubility curves (a small table and polynomial fits), yields of cooling and evaporative
crystallisation with hydrates, supersaturation and the metastable limit, nucleation and growth rate laws,
the MSMPR population balance (crystal size distribution, moments, dominant size) and the extraction of
kinetics from a measured CSD.

>>> from engsep import crystallization as cr
>>> round(cr.msmpr_dominant_size(G=1e-8, tau=3600.0) * 1e6, 1)       # L_D = 3 G tau in micrometres
108.0
"""

from __future__ import annotations

import math

import numpy as np

from ._common import trapezoid

# solubility of common salts in water, g per 100 g water at (0, 20, 40, 60, 80, 100) degC (CRC Handbook)
SOLUBILITY: dict = {
    "KNO3": (13.3, 31.6, 63.9, 110.0, 169.0, 246.0),
    "NaCl": (35.7, 36.0, 36.6, 37.3, 38.4, 39.8),
    "KCl": (27.6, 34.0, 40.0, 45.5, 51.1, 56.7),
    "CuSO4": (14.3, 20.7, 28.5, 40.0, 55.0, 75.4),
    "Na2SO4": (4.9, 19.5, 48.8, 45.3, 43.7, 42.7),
    "sucrose": (179.2, 203.9, 238.1, 287.3, 362.1, 487.2),
    "K2Cr2O7": (4.6, 12.5, 26.3, 45.6, 73.0, 102.0),
    "NH4Cl": (29.4, 37.2, 45.8, 55.2, 65.6, 77.3),
}
_T_TAB = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])


def solubility(salt: str, T_C) -> float:
    """g anhydrous solute per 100 g water at T (degC), interpolated in the table."""
    return np.interp(np.asarray(T_C, float), _T_TAB, np.array(SOLUBILITY[salt]))


def cooling_yield(
    salt: str,
    feed_solution_kg: float,
    T1: float,
    T2: float,
    evaporated_water_kg: float = 0.0,
    hydrate_water_ratio: float = 0.0,
) -> dict:
    """Mass of crystals from cooling a saturated solution from T1 to T2 (optionally losing solvent by evaporation),
    for anhydrous crystals or a hydrate with hydrate_water_ratio = kg water of crystallisation per kg anhydrous
    solute in the crystal. Balances: solute = crystals (anhydrous part) + solution at T2; water = free water."""
    c1, c2 = solubility(salt, T1) / 100, solubility(salt, T2) / 100  # kg solute per kg water
    W0 = feed_solution_kg / (1 + c1)  # water in the feed
    S0 = W0 * c1
    # crystals C (anhydrous mass Ca = C/(1 + h)) ; remaining water W0 - E - h Ca ; solute left = c2 (W0 - E - h Ca)
    h = hydrate_water_ratio
    Ca = (S0 - c2 * (W0 - evaporated_water_kg)) / (1 - c2 * h)
    Ca = max(Ca, 0.0)
    return {
        "crystals_kg": Ca * (1 + h),
        "anhydrous_kg": Ca,
        "mother_liquor_kg": feed_solution_kg - evaporated_water_kg - Ca * (1 + h),
        "yield_fraction": Ca / S0,
    }


def supersaturation(c: float, c_star: float) -> dict:
    return {"absolute": c - c_star, "relative": (c - c_star) / c_star, "ratio": c / c_star}


# ------------------------------------------------------------------ kinetics and the MSMPR crystalliser
def growth_rate(k_g: float, dc: float, g: float = 1.0) -> float:
    return k_g * dc**g


def nucleation_rate(k_b: float, dc: float, b: float, M_T: float = 1.0, j: float = 0.0) -> float:
    """B0 = k_b M_T^j dc^b (secondary nucleation with magma density M_T)."""
    return k_b * M_T**j * dc**b


def msmpr_population(L, n0: float, G: float, tau: float):
    """Population density n(L) = n0 exp(-L/(G tau)) for a mixed-suspension mixed-product-removal crystalliser."""
    L = np.asarray(L, float)
    return n0 * np.exp(-L / (G * tau))


def msmpr_moments(n0: float, G: float, tau: float, k_v: float = np.pi / 6, rho_c: float = 2000.0) -> dict:
    """Moments of the MSMPR distribution: number, length, area and mass per unit volume, and the magma density."""
    m = [n0 * math.factorial(j) * (G * tau) ** (j + 1) for j in range(4)]
    return {
        "m0_number": m[0],
        "m1_length": m[1],
        "m2_area": m[2],
        "m3_volume": m[3],
        "magma_density": k_v * rho_c * m[3],
    }


def msmpr_dominant_size(G: float, tau: float) -> float:
    """Mode of the mass distribution: L_D = 3 G tau."""
    return 3 * G * tau


def msmpr_mass_distribution(L, G: float, tau: float):
    """Mass density function (normalised): (L^3/(6 (G tau)^4)) exp(-L/(G tau))."""
    L = np.asarray(L, float)
    return L**3 / (6 * (G * tau) ** 4) * np.exp(-L / (G * tau))


def msmpr_cumulative_mass(L, G: float, tau: float):
    x = np.asarray(L, float) / (G * tau)
    return 1 - np.exp(-x) * (1 + x + x**2 / 2 + x**3 / 6)


def msmpr_from_csd(L, n, tau: float) -> dict:
    """Extract G and n0 (and the nucleation rate B0 = n0 G) from a measured population density: ln n vs L is a
    straight line of slope -1/(G tau)."""
    L, n = np.asarray(L, float), np.asarray(n, float)
    slope, intercept = np.polyfit(L, np.log(n), 1)
    G = -1 / (slope * tau)
    n0 = math.exp(intercept)
    return {"G": float(G), "n0": float(n0), "B0": float(n0 * G)}


def msmpr_design(M_T: float, L_D: float, G: float, k_v: float = np.pi / 6, rho_c: float = 2000.0) -> dict:
    """Residence time and nucleation rate for a target dominant size and magma density."""
    tau = L_D / (3 * G)
    n0 = M_T / (6 * k_v * rho_c * (G * tau) ** 4)
    return {"tau": tau, "n0": n0, "B0": n0 * G}


def sauter_mean(L, n):
    L, n = np.asarray(L, float), np.asarray(n, float)
    return float(trapezoid(n * L**3, L) / trapezoid(n * L**2, L))
