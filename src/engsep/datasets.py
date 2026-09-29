"""Bundled data files: literature tables with sources and pseudo-experimental course data whose generating model,
noise and deliberate faults are stated in each file's header.

>>> from engsep import datasets
>>> "vle_ethanol_water.csv" in datasets.available()
True
"""

from __future__ import annotations

from importlib import resources

INFO = {
    "vle_ethanol_water.csv": "Ethanol-water VLE at 1 atm (Carey & Lewis 1932, via Perry's).",
    "gas_permeabilities.csv": "Gas permeabilities (Barrer) of six membrane polymers (Baker; Robeson).",
    "solubility.csv": "Solubility of eight salts in water, 0-100 degC (CRC Handbook).",
    "ternary_acetic_water_ether.csv": "Tie lines of acetic acid - water - isopropyl ether at 20 degC (Treybal / Seader).",
    "isotherm_phenol_carbon.csv": "Phenol on activated carbon: batch isotherm points (Langmuir, 4 % noise).",
    "breakthrough_curve.csv": "Laboratory fixed-bed breakthrough curve (Bohart-Adams, 2 % noise, one contaminated sample).",
    "drying_curve.csv": "Tray-drying curve of a filter cake (constant and falling rate, 1.5 % noise).",
    "csd_msmpr.csv": "Crystal size distribution from an MSMPR crystalliser (5 % noise).",
    "plant_column_benzene_toluene.csv": "30 days of benzene-toluene column data with unit faults and a retrofit target (project).",
}


def path(name: str):
    if name not in INFO:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(sorted(INFO))}")
    return resources.files("engsep.data") / name


def available() -> list[str]:
    return sorted(INFO)


def describe(name: str) -> str:
    return INFO[name]
