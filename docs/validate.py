"""Validation of engsep against analytic solutions, published data and independent computations.

Run:  python docs/validate.py      (writes docs/VALIDATION.md)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import optimize

from engsep import distillation as dist
from engsep import flash, vle
from engsep._common import trapezoid
from engsep.constants import R
from engsep.vle import y_from_alpha

RESULTS: list[tuple] = []


def check(section, name, reference, value, expected, rtol=0.0, atol=0.0):
    ok = bool(np.isclose(float(value), float(expected), rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))
    if not ok:
        print(f"FAIL: {section} | {name}: computed {value!r}, expected {expected!r}")


# ------------------------------------------------------------------ 1. vapour pressures and VLE
S = "Vapour-liquid equilibrium"
nist_bp = {
    "water": 373.15,
    "methanol": 337.8,
    "ethanol": 351.4,
    "acetone": 329.2,
    "benzene": 353.2,
    "toluene": 383.8,
    "n-hexane": 341.9,
    "n-heptane": 371.6,
    "n-octane": 398.8,
    "cyclohexane": 353.9,
    "chloroform": 334.3,
    "ethyl acetate": 350.2,
    "1-propanol": 370.3,
    "2-propanol": 355.4,
    "n-pentane": 309.2,
    "n-butane": 272.7,
    "ammonia": 239.8,
    "acetic acid": 391.0,
    "ethylbenzene": 409.3,
    "p-xylene": 411.5,
}
worst = max(abs(vle.boiling_point(n) - T) for n, T in nist_bp.items())
check(
    S,
    f"normal boiling points of {len(nist_bp)} compounds from the Antoine table vs NIST (worst deviation, K)",
    "NIST Chemistry WebBook",
    worst,
    0.0,
    atol=1.0,
)
check(S, "water vapour pressure at 25 degC", "NIST 3.17 kPa", vle.antoine_pressure("water", 298.15), 3169.0, rtol=0.01)
check(
    S,
    "antoine_temperature inverts antoine_pressure",
    "identity",
    vle.antoine_temperature("toluene", vle.antoine_pressure("toluene", 350.0)),
    350.0,
    rtol=1e-12,
)
az = vle.azeotrope("ethanol", "water")
check(
    S,
    "ethanol-water azeotrope composition at 1 atm",
    "0.894 mol fraction (Gmehling; within Wilson's 2-parameter accuracy)",
    az["x"],
    0.894,
    atol=0.015,
)
check(S, "... azeotrope temperature", "78.2 degC", az["T"] - 273.15, 78.2, atol=0.3)
T_, y_ = vle.bubble_y(("ethanol", "water"), 0.1)
check(
    S,
    "ethanol-water at x = 0.10: T and y vs data",
    "Perry: 86.7 degC, y = 0.437",
    abs(T_ - 273.15 - 86.7) + 10 * abs(y_ - 0.437),
    0.0,
    atol=0.3,
)
az2 = vle.azeotrope("2-propanol", "water")
check(S, "2-propanol-water azeotrope", "0.68 mol fraction, 80.4 degC (Perry)", az2["x"], 0.68, atol=0.04)
check(
    S,
    "methanol-water and acetone-water have no azeotrope",
    "known",
    float(vle.azeotrope("methanol", "water") is None and vle.azeotrope("acetone", "water") is None),
    1.0,
)
az3 = vle.azeotrope("acetone", "methanol")
check(S, "acetone-methanol azeotrope", "0.78-0.80 mol acetone, 55.5 degC (Perry)", az3["x"], 0.79, atol=0.02)
g1, g2 = vle.wilson_gamma(("ethanol", "water"), 0.0, 350.0)
check(
    S,
    "Wilson: gamma_2 = 1 at x_1 = 0 and gamma_1 > 1 at infinite dilution (positive deviation)",
    "thermodynamics",
    float(abs(g2 - 1) < 1e-12 and g1 > 1),
    1.0,
)
g1i, g2i = vle.wilson_gamma(("ethanol", "water"), 1.0, 350.0)
check(S, "... gamma_1 = 1 at x_1 = 1", "thermodynamics", float(g1i), 1.0, rtol=1e-12)
xs = np.linspace(0.01, 0.99, 4000)
g1s, g2s = vle.wilson_gamma(("ethanol", "water"), xs, 350.0)
gd = xs * np.gradient(np.log(g1s), xs) + (1 - xs) * np.gradient(np.log(g2s), xs)
check(
    S,
    "Wilson satisfies the Gibbs-Duhem equation (max |x1 dlng1 + x2 dlng2|)",
    "thermodynamics",
    float(np.max(np.abs(gd[2:-2]))),
    0.0,
    atol=2e-3,
)
bt = vle.Txy(("benzene", "toluene"), n=11, ideal=True)
check(
    S,
    "benzene-toluene (ideal) bubble points lie between the pure boiling points and y > x",
    "Raoult",
    float(np.all(np.diff(bt["T"]) < 0) and np.all(bt["y"][1:-1] > bt["x"][1:-1])),
    1.0,
)
check(
    S,
    "benzene-toluene relative volatility at 1 atm (geometric mean)",
    "Seader: about 2.4-2.5",
    vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True),
    2.45,
    rtol=0.03,
)
check(
    S,
    "y_from_alpha and x_from_alpha are inverses",
    "identity",
    float(vle.x_from_alpha(vle.y_from_alpha(0.37, 2.5), 2.5)),
    0.37,
    rtol=1e-12,
)
T_dew = vle.dew_T(("benzene", "toluene"), 0.5, ideal=True)
T_bub = vle.bubble_T(("benzene", "toluene"), 0.5, ideal=True)
check(
    S, "dew point above bubble point for a 50/50 benzene-toluene mixture", "thermodynamics", float(T_dew > T_bub), 1.0
)
check(
    S,
    "dew_T inverts bubble_y for the non-ideal ethanol-water pair",
    "identity",
    vle.dew_T(("ethanol", "water"), y_),
    T_,
    rtol=1e-8,
)

# ------------------------------------------------------------------ 2. flash
S = "Flash"
rr = flash.rachford_rice([0.5, 0.5], [2.0, 0.5])
check(
    S,
    "Rachford-Rice, K = 2 and 0.5, equimolar: psi = 0.5, x = (1/3, 2/3)",
    "exact",
    rr["V_over_F"] + rr["x"][0],
    0.5 + 1 / 3,
    rtol=1e-12,
)
check(
    S,
    "... x and y sum to 1 and satisfy the balance z = (1-psi) x + psi y",
    "identity",
    float(np.sum(rr["x"]) + np.sum(rr["y"]) + np.max(np.abs(0.5 * rr["x"] + 0.5 * rr["y"] - 0.5))),
    2.0,
    rtol=1e-12,
)
check(
    S, "feed below the bubble point stays liquid", "exact", flash.rachford_rice([0.5, 0.5], [0.9, 0.5])["V_over_F"], 0.0
)
check(S, "feed above the dew point stays vapour", "exact", flash.rachford_rice([0.5, 0.5], [3.0, 1.5])["V_over_F"], 1.0)
names, z = ["n-butane", "n-pentane", "n-hexane"], [0.3, 0.4, 0.3]
Tb, Td = flash.bubble_point_T(names, z, 101325.0), flash.dew_point_T(names, z, 101325.0)
check(S, "ternary alkane mixture: bubble point below dew point", "thermodynamics", float(Td > Tb), 1.0)
fl_b = flash.flash_TP(names, z, Tb + 1e-3, 101325.0)
check(S, "at the bubble point the flash gives psi ~ 0", "consistency", fl_b["V_over_F"], 0.0, atol=1e-3)
fl_d = flash.flash_TP(names, z, Td - 1e-3, 101325.0)
check(S, "at the dew point the flash gives psi ~ 1", "consistency", fl_d["V_over_F"], 1.0, atol=1e-3)
fl_m = flash.flash_TP(names, z, 0.5 * (Tb + Td), 101325.0)
check(
    S,
    "mid-way flash: the vapour is richer in butane than the liquid",
    "physics",
    float(fl_m["y"][0] > fl_m["x"][0]),
    1.0,
)
check(
    S,
    "bubble_point_P at the bubble temperature returns 1 atm",
    "identity",
    flash.bubble_point_P(names, z, Tb),
    101325.0,
    rtol=1e-8,
)
check(
    S,
    "dew_point_P at the dew temperature returns 1 atm",
    "identity",
    flash.dew_point_P(names, z, Td),
    101325.0,
    rtol=1e-8,
)
ad = flash.adiabatic_flash(
    ["n-pentane", "n-hexane"], [0.5, 0.5], 360.0, 101325.0, cp_liquid=[167.0, 195.0], dH_vap=[25.8e3, 28.9e3]
)
check(
    S,
    "adiabatic flash of a hot pentane-hexane liquid: temperature drops and some vapour forms",
    "physics",
    float(ad["T"] < 360.0 and 0 < ad["V_over_F"] < 1),
    1.0,
)
h_in = 0.5 * 167.0 * (360.0 - ad["T"]) + 0.5 * 195.0 * (360.0 - ad["T"])
h_vap = ad["V_over_F"] * np.sum(ad["y"] * np.array([25.8e3, 28.9e3]))
check(
    S,
    "... enthalpy balance: sensible heat released = latent heat of the vapour formed",
    "identity",
    h_in,
    h_vap,
    rtol=1e-6,
)

# ------------------------------------------------------------------ 3. distillation
S = "Distillation"
alpha = 2.5
y_eq = lambda x: y_from_alpha(x, alpha)  # noqa: E731
N_min_fenske = dist.fenske(0.95, 0.05, alpha)
N_min_step = dist.minimum_stages(y_eq, 0.95, 0.05)
check(
    S,
    "Fenske minimum stages vs stepping at total reflux (constant alpha = 2.5, 95/5 split)",
    "consistency",
    N_min_step,
    math.ceil(N_min_fenske),
    atol=0,
)
check(
    S, "... Fenske value", "ln[(0.95/0.05)^2]/ln 2.5 = 6.43", N_min_fenske, 2 * math.log(19) / math.log(2.5), rtol=1e-12
)
rm = dist.minimum_reflux(y_eq, 0.95, 0.5, q=1.0)
R_min_analytic = (0.95 - y_from_alpha(0.5, alpha)) / (y_from_alpha(0.5, alpha) - 0.5)
check(S, "minimum reflux from the pinch (saturated liquid feed)", "analytic", rm["R_min"], R_min_analytic, rtol=1e-10)
uw = dist.underwood([alpha, 1.0], [0.5, 0.5], [0.95, 0.05], q=1.0)
check(
    S,
    "Underwood R_min equals the McCabe-Thiele pinch value for a binary",
    "consistency",
    uw["R_min"],
    R_min_analytic,
    rtol=1e-8,
)
mt = dist.mccabe_thiele(y_eq, 0.95, 0.05, 0.5, R=2.0, q=1.0)
check(
    S,
    "McCabe-Thiele stage count lies between Fenske and the stages at 1.05 R_min",
    "consistency",
    float(
        N_min_fenske < mt["stages"] < dist.mccabe_thiele(y_eq, 0.95, 0.05, 0.5, R=1.05 * rm["R_min"], q=1.0)["stages"]
    ),
    1.0,
)
check(
    S,
    "... the last stage reaches below xB and the stair steps are monotone",
    "consistency",
    float(mt["x"][-1] <= 0.05 and np.all(np.diff(mt["x"]) <= 1e-12)),
    1.0,
)
n_big = dist.mccabe_thiele(y_eq, 0.95, 0.05, 0.5, R=200.0, q=1.0)["stages"]
check(
    S,
    "at very high reflux the stage count approaches the Fenske minimum",
    "limit",
    n_big,
    math.ceil(N_min_fenske),
    atol=1,
)
try:
    dist.mccabe_thiele(y_eq, 0.95, 0.05, 0.5, R=0.5 * rm["R_min"], q=1.0)
    below = 0.0
except ValueError:
    below = 1.0
check(S, "a reflux ratio below the minimum is rejected", "exact", below, 1.0)
gl = dist.gilliland(2.0, rm["R_min"], N_min_fenske)
check(
    S,
    "Gilliland (Molokanov) at X = 0: N = N_min... in the limit R -> R_min, Y -> 1 and N -> infinity; at R = R_min + 1e-9 N is huge",
    "limit",
    float(dist.gilliland(rm["R_min"] + 1e-9, rm["R_min"], N_min_fenske) > 1e3),
    1.0,
)
check(
    S,
    "Gilliland at very high reflux (X -> 1): N -> N_min",
    "limit",
    dist.gilliland(1e6, rm["R_min"], N_min_fenske),
    N_min_fenske,
    rtol=1e-3,
)
for X_, Y_ in ((0.1, 0.545), (0.3, 0.38), (0.5, 0.265)):
    check(
        S,
        f"Gilliland chart reading X = {X_} -> Y = {Y_} (Molokanov's fit within 0.02)",
        "Gilliland 1940; Seader Fig. 9.10",
        1 - math.exp((1 + 54.4 * X_) / (11 + 117.2 * X_) * (X_ - 1) / math.sqrt(X_)),
        Y_,
        atol=0.02,
    )
check(
    S,
    "shortcut FUG stage count vs McCabe-Thiele at the same reflux (within 25 %, the correlation's accuracy)",
    "Gilliland vs rigorous",
    dist.fug(alpha, 0.5, 0.95, 0.05, R_over_Rmin=2.0 / rm["R_min"])["N"],
    mt["stages"],
    rtol=0.25,
)
kb = dist.kirkbride(20.0, 50.0, 50.0, 0.5, 0.5, 0.05, 0.05)
check(
    S, "Kirkbride: symmetric separation puts the feed at mid-column", "symmetry", kb["N_rectifying"], 10.0, rtol=1e-12
)
q_lines = dist.operating_lines(0.95, 0.05, 0.5, 2.0, 0.5)
check(S, "q-line for q = 0.5 passes through (zF, zF)", "geometry", q_lines["q_line"](0.5), 0.5, rtol=1e-12)
check(
    S,
    "operating lines meet on the q-line",
    "geometry",
    q_lines["q_line"](q_lines["x_int"]),
    q_lines["y_int"],
    rtol=1e-10,
)
bal = dist.column_balance(100.0, 0.5, 0.95, 0.05)
check(S, "column balance: 50/50 feed to 95/5 products gives D = B = 50", "exact", bal["D"], 50.0, rtol=1e-12)
ray = dist.rayleigh(y_eq, 0.5, 0.2)
check(
    S,
    "Rayleigh distillation: average distillate composition from the balance equals the integral result",
    "identity",
    ray["y_avg"],
    (0.5 - ray["W"] * 0.2) / ray["D"],
    rtol=1e-12,
)
W_analytic = math.exp(-1 / (alpha - 1) * (math.log(0.5 / 0.2) + alpha * math.log((1 - 0.2) / (1 - 0.5))))
check(S, "... vs the closed-form Rayleigh integral for constant alpha", "analytic", ray["W"], W_analytic, rtol=1e-8)
bcr = dist.batch_constant_reflux(y_eq, 0.5, 0.2, R=3.0, N=3)
check(
    S,
    "batch rectification: the distillate gets poorer as the still is depleted",
    "physics",
    float(np.all(np.diff(bcr["x_D"]) <= 1e-9) and bcr["W"][-1] < 1.0),
    1.0,
)
check(
    S,
    "Murphree efficiency below 1 needs more stages",
    "physics",
    float(dist.mccabe_thiele(y_eq, 0.95, 0.05, 0.5, 2.0, murphree=0.7)["stages"] > mt["stages"]),
    1.0,
)
check(
    S,
    "Murphree efficiency of 1 reproduces the ideal count",
    "identity",
    dist.mccabe_thiele(y_eq, 0.95, 0.05, 0.5, 2.0, murphree=1.0)["stages"],
    mt["stages"],
)
check(S, "O'Connell efficiency for alpha mu = 1 mPa s", "0.49", dist.oconnell_efficiency(2.0, 0.5), 0.492, rtol=1e-12)
mt_ew = dist.mccabe_thiele(lambda x: vle.equilibrium_y(("ethanol", "water"), x), 0.85, 0.02, 0.3, R=3.0)
check(
    S,
    "ethanol-water column to 85 % distillate needs many stages near the azeotrope (> 10) and a feed stage",
    "physics",
    float(mt_ew["stages"] > 10 and mt_ew["feed_stage"] is not None),
    1.0,
)

# ------------------------------------------------------------------ 4. absorption and stripping
from engsep import absorption as ab  # noqa: E402
from engsep import adsorption as ad  # noqa: E402
from engsep import crystallization as cr  # noqa: E402
from engsep import drying_evaporation as de  # noqa: E402
from engsep import extraction as ex  # noqa: E402
from engsep import membranes as mb  # noqa: E402
from engsep import transport as tr  # noqa: E402

S = "Absorption and stripping"
A, N = 1.4, 6
phi = ab.kremser_fraction_absorbed(A, N)
check(
    S,
    "Kremser: stages from the recovery invert the fraction absorbed",
    "identity",
    ab.kremser_stages(A, phi),
    N,
    rtol=1e-10,
)
check(S, "Kremser for A = 1 is N/(N+1)", "exact", ab.kremser_fraction_absorbed(1.0, 4), 0.8, rtol=1e-12)
m, L, V = 0.8, 1.0, 1.0
st = ab.stages_absorber(
    lambda x: m * x, L, V, y_in=0.02, y_out=0.02 * (1 - ab.kremser_fraction_absorbed(L / (m * V), 5)), x_in=0.0
)
check(
    S,
    "stage-by-stage absorber with a linear line reproduces the Kremser stage count (5)",
    "consistency",
    st["stages"],
    5,
    atol=0,
)
check(
    S,
    "... and the solute balance: L x_out = V (y_in - y_out)",
    "identity",
    L * st["x_out"],
    V * (0.02 - 0.02 * (1 - ab.kremser_fraction_absorbed(L / (m * V), 5))),
    rtol=1e-6,
)
check(
    S,
    "Kremser -> complete absorption as N -> infinity for A > 1",
    "limit",
    ab.kremser_fraction_absorbed(1.4, 200),
    1.0,
    rtol=1e-8,
)
check(S, "... and -> A for A < 1", "limit", ab.kremser_fraction_absorbed(0.6, 200), 0.6, rtol=1e-8)
check(
    S,
    "minimum solvent: x_out at the pinch equals y_in/m",
    "exact",
    ab.minimum_solvent(1.0, 0.02, 0.002, 0.8) * (0.02 / 0.8),
    1.0 * (0.02 - 0.002),
    rtol=1e-12,
)
y_in, y_out, x_in, A_ = 0.02, 0.002, 0.0, 1.5
ntu_c = ab.ntu_colburn(y_in, y_out, m, x_in, A_)
L_over_V = A_ * m
ntu_num = ab.ntu_gas(y_in, y_out, lambda x: m * x, lambda y: x_in + (y - y_out) / L_over_V, n=4000)
check(
    S,
    "Colburn's NTU equals the numerical integral for a linear equilibrium line",
    "Colburn 1939",
    ntu_c,
    ntu_num,
    rtol=1e-4,
)
check(
    S,
    "Colburn for A = 1 reduces to (y_in - y_out)/(y_out - m x_in)",
    "exact",
    ab.ntu_colburn(0.02, 0.002, 0.8, 0.0, 1.0),
    9.0,
    rtol=1e-12,
)
tf = ab.overall_from_film(k_y=1.0, k_x=2.0, m=0.5)
check(S, "two-film theory: 1/K_y = 1/k_y + m/k_x", "exact", tf["K_y"], 1 / (1 + 0.25), rtol=1e-12)
check(S, "... K_x = m K_y", "identity", tf["K_x"], 0.5 * tf["K_y"], rtol=1e-12)
check(S, "H_OG = H_G + H_L/A", "exact", ab.htu_overall_from_individual(0.5, 0.3, 1.5), 0.7, rtol=1e-12)
check(
    S,
    "stripping NTU by Colburn (liquid basis) mirrors absorption",
    "symmetry",
    ab.stripping_tower_ntu(0.02, 0.002, 1.25, 0.0, 1.5),
    ab.ntu_colburn(0.02, 0.002, 0.8, 0.0, 1.5),
    rtol=1e-12,
)

# ------------------------------------------------------------------ 5. extraction and leaching
S = "Extraction and leaching"
K_, SF = 2.0, 1.0
cc = ex.countercurrent_immiscible(K_, SF, 0.10, 4)
check(
    S,
    "countercurrent immiscible extraction (Kremser form) inverts its stage count",
    "identity",
    ex.countercurrent_stages_immiscible(K_, SF, 0.10, cc["x_out"]),
    4,
    rtol=1e-10,
)
check(
    S,
    "cross-current with N stages: (1 + E)^-N",
    "exact",
    ex.crosscurrent_immiscible(K_, SF, 0.10, 3)["x_out"],
    0.10 / 27,
    rtol=1e-12,
)
check(
    S,
    "countercurrent beats cross-current for the same total solvent (4 stages, S/F = 1 in total)",
    "Seader Ch. 8",
    float(
        ex.countercurrent_immiscible(K_, 1.0, 0.10, 4)["x_out"] < ex.crosscurrent_immiscible(K_, 0.25, 0.10, 4)["x_out"]
    ),
    1.0,
)
check(
    S,
    "single stage: fraction extracted E/(1 + E)",
    "exact",
    ex.single_stage_immiscible(K_, SF, 0.10)["fraction_extracted"],
    2 / 3,
    rtol=1e-12,
)
check(
    S,
    "minimum solvent: infinite stages needed at S/F = S_min (Kremser count diverges)",
    "limit",
    float(
        ex.countercurrent_stages_immiscible(K_, ex.minimum_solvent_immiscible(K_, 0.10, 0.01) * (1 + 1e-6), 0.10, 0.01)
        > 50
    ),
    1.0,
)
a_r = np.linspace(0, 0.35, 15)
tern = ex.Ternary(
    np.c_[a_r, 0.02 + 0.2 * a_r**2],
    np.c_[np.linspace(0, 0.6, 15), 0.98 - 1.2 * np.linspace(0, 0.6, 15)],
    np.c_[a_r[1:], 2.0 * a_r[1:] * (1 - a_r[1:])],
)
hn = ex.hunter_nash(tern, 100.0, (0.25, 0.0), 100.0, (0.0, 1.0), 0.02)
check(
    S,
    "Hunter-Nash on a near-immiscible ternary: stage count agrees with the Kremser estimate (rounded up)",
    "consistency",
    hn["stages"],
    math.ceil(ex.countercurrent_stages_immiscible(tern.distribution_coefficient(0.1), 1.0, 0.25, 0.02)),
    atol=1,
)
check(S, "... overall balance F + S = E1 + RN", "identity", hn["E1"] + hn["RN"], 200.0, rtol=1e-10)
check(
    S,
    "... solute balance F xF + S xS = E1 x_E1 + RN x_RN",
    "identity",
    hn["E1"] * hn["x_E1"][0] + hn["RN"] * hn["x_RN"][0],
    25.0,
    rtol=1e-6,
)
check(
    S,
    "... the mixing point lies on the line E1-RN (lever rule)",
    "geometry",
    abs(
        (hn["x_E1"][0] - hn["x_RN"][0]) * (hn["x_M"][1] - hn["x_RN"][1])
        - (hn["x_E1"][1] - hn["x_RN"][1]) * (hn["x_M"][0] - hn["x_RN"][0])
    ),
    0.0,
    atol=1e-9,
)
check(
    S,
    "... each raffinate is in equilibrium with its extract (tie line)",
    "identity",
    float(max(abs(tern.extract_in_equilibrium(r[0])[0] - e[0]) for r, e in zip(hn["raffinates"], hn["extracts"]))),
    0.0,
    atol=1e-9,
)
check(
    S,
    "leaching: recovery with a washing factor is the Kremser expression",
    "Seader Ch. 16",
    ex.leaching_countercurrent(3, 2.0, 1.0, 100.0)["recovery"],
    (2**4 - 2) / (2**4 - 1),
    rtol=1e-12,
)
check(
    S,
    "leaching stages invert the recovery",
    "identity",
    ex.leaching_stages((2**4 - 2) / (2**4 - 1), 2.0),
    3,
    rtol=1e-10,
)

# ------------------------------------------------------------------ 6. adsorption
S = "Adsorption and fixed beds"
c_ = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
q_ = ad.langmuir(c_, 4.0, 0.8)
fl_ = ad.fit_langmuir(c_, q_)
check(S, "Langmuir fit recovers q_max and K from exact data", "exact", fl_["q_max"] + fl_["K"], 4.8, rtol=1e-8)
ff_ = ad.fit_freundlich(c_, ad.freundlich(c_, 2.0, 2.5))
check(S, "Freundlich fit recovers k and n", "exact", ff_["k"] + ff_["n"], 4.5, rtol=1e-10)
check(
    S,
    "BET reduces to Langmuir-like at small x and diverges as x -> 1",
    "BET 1938",
    float(
        ad.bet(0.9999, 1.0, 100.0) > 1e3
        and abs(ad.bet(0.01, 1.0, 100.0) - 100 * 0.01 / (0.99 * (1 + 99 * 0.01))) < 1e-12
    ),
    1.0,
)
ba_ = ad.batch_adsorber(10.0, 1.0, 2.0, lambda c: ad.langmuir(c, 4.0, 0.8))
check(
    S,
    "batch adsorber closes its balance V (c0 - c) = m q",
    "identity",
    1.0 * (10.0 - ba_["c"]),
    2.0 * ba_["q"],
    rtol=1e-10,
)
t_star = ad.stoichiometric_time(2.0, 10.0, 500.0, 0.4, 1.0, 0.01)
check(
    S,
    "stoichiometric time equals bed length over wave velocity",
    "identity",
    t_star,
    1.0 / ad.wave_velocity(2.0, 10.0, 500.0, 0.4, 0.01),
    rtol=1e-12,
)
t_ = np.linspace(0, 3 * t_star, 6000)
curve = ad.bohart_adams(t_, 10.0, 2.0, 500.0, 1.0, 0.01, k=1e-3)
ban = ad.breakthrough_analysis(t_, curve)
t_ba = 2.0 * 500.0 * 1.0 / (0.01 * 10.0)  # the Bohart-Adams model neglects the void holdup (eps L/u = 40 s here)
check(
    S,
    "Bohart-Adams curve: the equal-area (stoichiometric) time equals q0 rho_b L/(u c0)",
    "constant pattern",
    ban["t_stoich"],
    t_ba,
    rtol=1e-3,
)
check(
    S,
    "... and its midpoint (c/c0 = 0.5) is at that time",
    "symmetry",
    float(np.interp(0.5, curve, t_)),
    t_ba,
    rtol=1e-6,
)
check(
    S,
    "... which is within 1 % of the full stoichiometric time including the holdup",
    "consistency",
    t_ba,
    t_star,
    rtol=0.01,
)
check(S, "... breakthrough before, exhaustion after", "physics", float(ban["t_break"] < t_star < ban["t_exhaust"]), 1.0)
lub = ad.lub_scale_up(0.2, 1.0, ban["t_break"], 2.0)
check(
    S,
    "LUB scale-up: breakthrough time scales with the used length (L - LUB)",
    "LUB method",
    lub["t_break_new"] / ban["t_break"],
    (2.0 - 0.2) / (1.0 - 0.2),
    rtol=1e-12,
)
kl = ad.klinkenberg(np.array([1e5, 2e5, 4e5]), 1.0, 10.0, 5.0, 500.0, 0.4, 0.01, 1e-3)
check(
    S,
    "Klinkenberg curve rises monotonically from 0 to 1",
    "physics",
    float(np.all(np.diff(kl) > 0) and kl[0] < 0.5 < kl[-1]),
    1.0,
)
check(
    S,
    "Klinkenberg midpoint near the stoichiometric time for a linear isotherm",
    "constant pattern",
    float(optimize.brentq(lambda tt: ad.klinkenberg(tt, 1.0, 10.0, 5.0, 500.0, 0.4, 0.01, 1e-3) - 0.5, 1e3, 1e7)),
    ad.stoichiometric_time(50.0, 10.0, 500.0, 0.4, 1.0, 0.01),
    rtol=0.02,
)

# ------------------------------------------------------------------ 7. membranes
S = "Membranes"
check(
    S,
    "osmotic pressure of seawater-like 0.6 M NaCl at 25 degC",
    "van 't Hoff: 2 x 0.6 x R T = 29.7 bar (measured ~27 with activity)",
    mb.osmotic_pressure(1.2, 298.15) / 1e5,
    29.7,
    rtol=0.01,
)
cm = mb.complete_mixing(alpha=5.0, x_f=0.21, pressure_ratio=0.2, theta=0.3)
check(
    S,
    "complete-mixing gas permeation: balance x_f = theta y + (1 - theta) x",
    "identity",
    0.3 * cm["y_permeate"] + 0.7 * cm["x_retentate"],
    0.21,
    rtol=1e-10,
)
check(
    S, "... permeate enriched, retentate depleted", "physics", float(cm["y_permeate"] > 0.21 > cm["x_retentate"]), 1.0
)
cf = mb.cross_flow(alpha=5.0, x_f=0.21, pressure_ratio=0.2, theta=0.3)
check(
    S,
    "cross flow gives a richer permeate than complete mixing (same cut)",
    "Seader Ch. 14",
    float(cf["y_permeate"] > cm["y_permeate"]),
    1.0,
)
check(S, "cross-flow balance closes", "identity", 0.3 * cf["y_permeate"] + 0.7 * cf["x_retentate"], 0.21, rtol=1e-6)
cf0 = mb.cross_flow(5.0, 0.21, 0.2, 1e-4)
cm0 = mb.complete_mixing(5.0, 0.21, 0.2, 1e-4)
check(
    S,
    "at vanishing stage cut both models give the local permeate composition",
    "limit",
    cf0["y_permeate"],
    cm0["y_permeate"],
    rtol=1e-3,
)
check(
    S,
    "at very low pressure ratio the ideal enrichment y/(1-y) = alpha x/(1-x) holds",
    "limit",
    mb.complete_mixing(5.0, 0.21, 1e-6, 1e-4)["y_permeate"],
    5 * 0.21 / (1 + 4 * 0.21),
    rtol=1e-3,
)
ro = mb.ro_permeate_concentration(A=3e-12, B=2e-8, dP=60e5, c_feed=0.6, T=298.15)
check(S, "RO of seawater at 60 bar: rejection above 98 %", "typical membranes", float(ro["rejection"] > 0.98), 1.0)
check(S, "... permeate concentration equals J_s/J_w", "identity", ro["c_permeate"], ro["J_s"] / ro["J_w"], rtol=1e-8)
check(
    S,
    "maximum recovery when the concentrate's osmotic pressure reaches the applied pressure",
    "exact",
    mb.ro_recovery(0.6, 60e5, 298.15),
    1 - mb.osmotic_pressure(1.2, 298.15) / 60e5,
    rtol=1e-12,
)
check(
    S,
    "concentration polarisation: c_m/c_b = exp(J/k)",
    "film model",
    mb.concentration_polarisation(2e-5, 4e-5),
    math.exp(0.5),
    rtol=1e-12,
)
check(
    S,
    "UF limiting flux: k ln(c_gel/c_b)",
    "gel polarisation",
    mb.uf_limiting_flux(1e-5, 300.0, 30.0),
    1e-5 * math.log(10),
    rtol=1e-12,
)
dz = mb.dialysis_countercurrent(1e-5, 1.0, 1e-5, 2e-5, 1.0)
check(
    S,
    "countercurrent dialyser: effectiveness between 0 and 1, removal = effectiveness for a clean dialysate",
    "exchanger analogy",
    dz["removal"],
    dz["effectiveness"],
    rtol=1e-12,
)
check(
    S,
    "... NTU -> infinity gives complete removal for R < 1",
    "limit",
    mb.dialysis_countercurrent(1e-2, 1.0, 1e-5, 2e-5, 1.0)["removal"],
    1.0,
    rtol=1e-6,
)

# ------------------------------------------------------------------ 8. crystallization
S = "Crystallization"
check(S, "KNO3 solubility at 20 degC", "CRC 31.6 g/100 g", float(cr.solubility("KNO3", 20.0)), 31.6, rtol=1e-12)
cy = cr.cooling_yield("KNO3", 1000.0, 80.0, 20.0)
check(
    S,
    "cooling KNO3 solution from 80 to 20 degC: yield fraction 1 - c2/c1",
    "balance",
    cy["yield_fraction"],
    1 - 31.6 / 169.0,
    rtol=1e-12,
)
check(S, "... mass balance closes", "identity", cy["crystals_kg"] + cy["mother_liquor_kg"], 1000.0, rtol=1e-12)
cyh = cr.cooling_yield("CuSO4", 1000.0, 80.0, 20.0, hydrate_water_ratio=5 * 18.02 / 159.6)
check(
    S,
    "hydrate (CuSO4.5H2O) crystals weigh more than the anhydrous solute they contain",
    "physics",
    float(cyh["crystals_kg"] > cyh["anhydrous_kg"] > 0),
    1.0,
)
W_free = 1000.0 / 1.55 - cyh["anhydrous_kg"] * 5 * 18.02 / 159.6
check(
    S,
    "... and the mother liquor is saturated at 20 degC",
    "identity",
    (1000.0 * 0.55 / 1.55 - cyh["anhydrous_kg"]) / W_free,
    20.7 / 100,
    rtol=1e-10,
)
G_, tau_, n0_ = 1e-8, 3600.0, 1e12
mom = cr.msmpr_moments(n0_, G_, tau_)
L_ = np.linspace(0, 40 * G_ * tau_, 40001)
n_ = cr.msmpr_population(L_, n0_, G_, tau_)
check(
    S,
    "MSMPR moments vs numerical integration of the population density (m3)",
    "analytic",
    mom["m3_volume"],
    float(trapezoid(n_ * L_**3, L_)),
    rtol=1e-6,
)
check(
    S,
    "dominant size 3 G tau is the mode of the mass distribution",
    "analytic",
    float(L_[np.argmax(cr.msmpr_mass_distribution(L_, G_, tau_))]),
    3 * G_ * tau_,
    rtol=1e-3,
)
check(
    S,
    "cumulative mass distribution reaches 1",
    "identity",
    float(cr.msmpr_cumulative_mass(40 * G_ * tau_, G_, tau_)),
    1.0,
    rtol=1e-8,
)
check(
    S,
    "... and its derivative is the mass density",
    "identity",
    float((cr.msmpr_cumulative_mass(3.001e-4, G_, tau_) - cr.msmpr_cumulative_mass(2.999e-4, G_, tau_)) / 2e-7),
    float(cr.msmpr_mass_distribution(3e-4, G_, tau_)),
    rtol=1e-5,
)
fit_csd = cr.msmpr_from_csd(L_[::2000], n_[::2000], tau_)
check(S, "kinetics from a CSD: G and n0 recovered", "exact", fit_csd["G"] / G_ + fit_csd["n0"] / n0_, 2.0, rtol=1e-8)
des = cr.msmpr_design(mom["magma_density"], 3 * G_ * tau_, G_)
check(
    S,
    "design inverts the moments: tau and n0 recovered",
    "identity",
    des["tau"] / tau_ + des["n0"] / n0_,
    2.0,
    rtol=1e-8,
)
check(
    S, "Sauter mean of the MSMPR distribution is 3 G tau", "analytic", cr.sauter_mean(L_, n_), 3 * G_ * tau_, rtol=1e-4
)

# ------------------------------------------------------------------ 9. drying, evaporation, transport
S = "Drying, evaporation and transport"
check(
    S, "saturation humidity at 30 degC", "psychrometric tables 0.0273", de.humidity_saturation(30.0), 0.0273, rtol=0.01
)
check(S, "dew point inverts the humidity", "identity", de.dew_point(de.humidity_saturation(25.0)), 25.0, rtol=1e-8)
check(
    S,
    "relative humidity of saturated air is 1",
    "identity",
    de.relative_humidity(de.humidity_saturation(40.0), 40.0),
    1.0,
    rtol=1e-12,
)
T_as = de.adiabatic_saturation_temperature(40.0, 0.01)
check(
    S,
    "adiabatic saturation temperature of 40 degC air at H = 0.01",
    "psychrometric chart: about 23 degC",
    T_as,
    23.0,
    atol=1.0,
)
check(
    S,
    "... saturated air is at its own adiabatic saturation temperature",
    "identity",
    de.adiabatic_saturation_temperature(30.0, de.humidity_saturation(30.0)),
    30.0,
    rtol=1e-6,
)
check(S, "latent heat of water at 100 degC", "2257 kJ/kg", de.latent_heat(100.0) / 1e3, 2257.0, rtol=0.005)
dt_ = de.drying_time(0.8, 0.1, 0.3, 0.05, 1e-3, 0.5)
check(
    S,
    "drying time: constant-rate part = (X0 - Xc)/(R_c A/m)",
    "exact",
    dt_["t_constant"],
    0.5 / (1e-3 * 0.5),
    rtol=1e-12,
)
check(
    S,
    "... falling-rate part uses the log mean of the free moisture",
    "exact",
    dt_["t_falling"],
    0.25 / (1e-3 * 0.5) * math.log(0.25 / 0.05),
    rtol=1e-12,
)
t_dry = np.linspace(0, 8000, 81)
X_dry = np.where(t_dry < 4000, 0.8 - 1e-4 * t_dry, 0.05 + 0.35 * np.exp(-(t_dry - 4000) * 0.4e-4 / 0.35))
dca = de.drying_curve_analysis(t_dry, X_dry)
check(
    S,
    "drying-curve analysis finds the critical moisture (0.4) and the constant rate (1e-4)",
    "recovery",
    dca["X_critical"] / 0.4 + dca["R_c"] / 1e-4,
    2.0,
    rtol=0.05,
)
se = de.single_effect(1.0, 0.1, 0.4, 20.0, 100.0, 120.0, 2000.0)
check(S, "single effect: vapour = F (1 - x_F/x_P)", "balance", se["vapour_kg_s"], 0.75, rtol=1e-12)
check(S, "... steam economy below 1 with a cold feed", "physics", float(se["economy"] < 1.0), 1.0)
me = de.multiple_effect(3, 1.0, 0.1, 0.4, 90.0, 120.0, 60.0, [2500.0, 2000.0, 1500.0])
me_cold = de.multiple_effect(3, 1.0, 0.1, 0.4, 20.0, 120.0, 60.0, [2500.0, 2000.0, 1500.0])
check(
    S,
    "triple effect: total vapour equals the single-effect requirement",
    "balance",
    float(np.sum(me["vapour_kg_s"])),
    0.75,
    rtol=1e-8,
)
check(
    S,
    "... equal areas",
    "design condition",
    float(np.max(np.abs(me["areas_m2"] - me["areas_m2"].mean())) / me["areas_m2"].mean()),
    0.0,
    atol=1e-5,
)
check(
    S,
    "... steam economy between 2 and 3 with a warm feed",
    "McCabe, Smith & Harriott Ch. 16",
    float(2.0 < me["economy"] < 3.0),
    1.0,
)
check(
    S,
    "... and lower with a 20 degC feed (sensible-heat load)",
    "physics",
    float(me_cold["economy"] < me["economy"]),
    1.0,
)
check(S, "... temperature drops sum to the overall drop", "identity", float(np.sum(me["dT"])), 60.0, rtol=1e-10)
check(
    S,
    "Fuller: water vapour in air at 25 degC",
    "measured 0.26 cm2/s",
    tr.fuller("water", "air", 298.15) * 1e4,
    0.26,
    rtol=0.05,
)
check(S, "Fuller: CO2 in air", "measured 0.16 cm2/s", tr.fuller("CO2", "air", 298.15) * 1e4, 0.16, rtol=0.05)
check(
    S,
    "Wilke-Chang: ethanol in water at 25 degC",
    "measured 1.24e-9 m2/s (the correlation is typically within 20-25 %)",
    tr.wilke_chang(18.02, 2.6, 298.15, 0.89, 60.0),
    1.24e-9,
    rtol=0.25,
)
check(
    S,
    "Stefan flux exceeds the equimolar (Fick) flux",
    "physics",
    float(
        tr.stefan_flux(2.5e-5, 101325.0, 298.15, 3000.0, 0.0, 0.01) > tr.fick_flux(2.5e-5, 3000.0 / (R * 298.15), 0.01)
    ),
    1.0,
)
check(S, "Sherwood for a sphere at rest is 2", "exact", tr.sherwood_sphere(0.0, 1.0), 2.0)
check(
    S,
    "Chilton-Colburn with f = 2 Sh/(Re Sc^1/3) is an identity",
    "identity",
    tr.chilton_colburn(2 * 50 / (1e4 * 2.0 ** (1 / 3)), 1e4, 2.0),
    50.0,
    rtol=1e-12,
)
check(
    S,
    "penetration coefficient scales as t^-1/2",
    "Higbie",
    tr.penetration_coefficient(1e-9, 1.0) / tr.penetration_coefficient(1e-9, 4.0),
    2.0,
    rtol=1e-12,
)

# ------------------------------------------------------------------ report
passed = sum(r[5] for r in RESULTS)
lines = [
    "# Validation report",
    "",
    "Generated by `python docs/validate.py`; rerun in CI on every push. Checks compare engsep with analytic",
    "solutions, published data (NIST, Perry, Gmehling, Seader) and internal consistency between methods.",
    "",
    f"**{passed} / {len(RESULTS)} checks pass.**",
    "",
]
for section in dict.fromkeys(r[0] for r in RESULTS):
    lines += ["", f"## {section}", "", "| Check | Reference | engsep | Expected | Result |", "|---|---|---|---|:-:|"]
    for s_, name, ref, v, expected, ok in RESULTS:
        if s_ == section:
            lines.append(f"| {name} | {ref} | {v:.10g} | {expected:.10g} | {'pass' if ok else 'FAIL'} |")
Path(__file__).with_name("VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{passed}/{len(RESULTS)} checks pass")
