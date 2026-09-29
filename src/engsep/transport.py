"""Mass-transfer fundamentals: diffusion coefficients in gases (Fuller) and liquids (Wilke-Chang), Fick's law and
diffusion through a stagnant film (Stefan), film-theory and penetration-theory coefficients, Sherwood
correlations, and the Chilton-Colburn analogy.

>>> from engsep import transport as tr
>>> round(tr.fuller("water", "air", T=298.15) * 1e4, 3)      # water vapour in air, cm2/s: 0.26 measured
0.253
"""

from __future__ import annotations

import numpy as np

from .constants import R

# Fuller diffusion volumes and molar masses
FULLER_V = {
    "air": 19.7,
    "N2": 17.9,
    "O2": 16.6,
    "H2": 7.07,
    "CO2": 26.9,
    "H2O": 12.7,
    "water": 12.7,
    "NH3": 14.9,
    "CH4": 24.42,
    "CO": 18.9,
    "ethanol": 50.36,
    "benzene": 90.96,
    "toluene": 111.48,
    "acetone": 66.86,
    "SO2": 41.1,
    "He": 2.88,
    "Ar": 16.1,
    "methanol": 29.9,
}
MOLAR_MASS = {
    "air": 28.97,
    "N2": 28.01,
    "O2": 32.0,
    "H2": 2.016,
    "CO2": 44.01,
    "H2O": 18.02,
    "water": 18.02,
    "NH3": 17.03,
    "CH4": 16.04,
    "CO": 28.01,
    "ethanol": 46.07,
    "benzene": 78.11,
    "toluene": 92.14,
    "acetone": 58.08,
    "SO2": 64.07,
    "He": 4.003,
    "Ar": 39.95,
    "methanol": 32.04,
}


def fuller(a: str, b: str, T: float, P: float = 101325.0) -> float:
    """Fuller-Schettler-Giddings gas diffusivity (m2/s):
    D = 1.013e-2 T^1.75 sqrt(1/MA + 1/MB) / (P (VA^1/3 + VB^1/3)^2), with P in Pa."""
    MA, MB = MOLAR_MASS[a], MOLAR_MASS[b]
    VA, VB = FULLER_V[a], FULLER_V[b]
    return float(1.013e-2 * T**1.75 * np.sqrt(1 / MA + 1 / MB) / (P * (VA ** (1 / 3) + VB ** (1 / 3)) ** 2))


def wilke_chang(M_solvent: float, phi: float, T: float, mu_mPa_s: float, V_solute_cm3_mol: float) -> float:
    """Wilke-Chang liquid diffusivity (m2/s): D = 7.4e-8 (phi M)^0.5 T / (mu V^0.6) in cm2/s with mu in cP."""
    return 7.4e-8 * np.sqrt(phi * M_solvent) * T / (mu_mPa_s * V_solute_cm3_mol**0.6) * 1e-4


def fick_flux(D: float, dc: float, dz: float) -> float:
    return D * dc / dz


def stefan_flux(D: float, P: float, T: float, p_A1: float, p_A2: float, dz: float) -> float:
    """Diffusion of A through stagnant B: N_A = (D P/(R T dz)) ln[(P - p_A2)/(P - p_A1)] (mol/(m2 s))."""
    return D * P / (R * T * dz) * np.log((P - p_A2) / (P - p_A1))


def film_coefficient(D: float, delta: float) -> float:
    return D / delta


def penetration_coefficient(D: float, t_contact: float) -> float:
    """Higbie: k = 2 sqrt(D/(pi t))."""
    return 2 * np.sqrt(D / (np.pi * t_contact))


def sherwood_sphere(Re: float, Sc: float) -> float:
    return 2 + 0.6 * Re**0.5 * Sc ** (1 / 3)


def sherwood_flat_plate_laminar(Re: float, Sc: float) -> float:
    return 0.664 * Re**0.5 * Sc ** (1 / 3)


def sherwood_pipe_turbulent(Re: float, Sc: float) -> float:
    """Linton-Sherwood: Sh = 0.023 Re^0.83 Sc^1/3."""
    return 0.023 * Re**0.83 * Sc ** (1 / 3)


def chilton_colburn(f: float, Re: float, Sc: float) -> float:
    """j_D = f/2 = Sh/(Re Sc^(1/3)) -> Sh."""
    return f / 2 * Re * Sc ** (1 / 3)
