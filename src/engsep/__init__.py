"""engsep - separation processes: readable, validated Python with a notebook course.

Modules
-------
constants     physical constants
vle           Antoine table, Raoult's law, Wilson activity coefficients, bubble/dew points, T-x-y and x-y, azeotropes
flash         Rachford-Rice flash, multicomponent bubble and dew points, adiabatic flash
distillation  McCabe-Thiele, minimum reflux and stages, Fenske-Underwood-Gilliland-Kirkbride, batch, efficiencies
absorption    Kremser, stage-by-stage absorbers, packed towers (HTU/NTU, Colburn), two-film theory
extraction    immiscible-solvent stages, ternary systems and Hunter-Nash, leaching
adsorption    isotherms, batch adsorbers, fixed-bed breakthrough (Bohart-Adams, Klinkenberg, LUB), PSA balance
membranes     gas permeation (complete mixing, cross flow), reverse osmosis, ultrafiltration, dialysis
crystallization  solubility, yields, MSMPR population balance and kinetics from a CSD
drying_evaporation  psychrometrics, drying curves and times, single- and multiple-effect evaporators
transport     Fuller and Wilke-Chang diffusivities, Stefan diffusion, film/penetration theory, Sherwood correlations
datasets      bundled literature tables and course data files
"""

__version__ = "0.1.0"

from . import (
    absorption,
    adsorption,
    constants,
    crystallization,
    datasets,
    distillation,
    drying_evaporation,
    extraction,
    flash,
    membranes,
    transport,
    vle,
)

__all__ = [
    "__version__",
    "absorption",
    "adsorption",
    "constants",
    "crystallization",
    "datasets",
    "distillation",
    "drying_evaporation",
    "extraction",
    "flash",
    "membranes",
    "transport",
    "vle",
]
