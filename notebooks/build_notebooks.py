"""Generate the course notebooks (single source of truth) and execute them.

    python notebooks/build_notebooks.py            # write and execute all notebooks
    python notebooks/build_notebooks.py 03 15      # only notebooks whose names start with 03 or 15
    python notebooks/build_notebooks.py --no-run   # write without executing
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS: dict[str, list] = {}
EXTRAS: dict[str, dict] = {}

COLAB = '''try:                      # on Google Colab (or anywhere engsep is missing): install it from GitHub
    import engsep
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ktwyw/engsep"], check=True)'''

STYLE = '''plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

import inspect
from IPython.display import Code

def show_source(obj):
    """Display the source code of a library function or class."""
    return Code(inspect.getsource(obj), language="python")

def heading(text):
    print(f"\\n{text}\\n" + "=" * len(text))'''


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def setup_cell(extra_imports=()):
    lines = ["import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"]
    for imp in extra_imports:
        if imp not in lines:
            lines.append(imp)
    return code(COLAB + "\n" + "\n".join(lines) + "\n" + STYLE)


def read_data(name):
    """Code snippet reading a bundled CSV file (comment lines start with #)."""
    return f'pd.read_csv(datasets.path("{name}"), comment="#")'



# =====================================================================================================





NOTEBOOKS["00_python_for_separations"] = [
    md(r"""
# 00 · Python for separation processes

Separations are decided by equilibrium and by rates: where a mixture wants to go, and how fast it gets there.
This notebook sets up the tools - vapour pressures from an Antoine table, mole fractions and material
balances, root finding and quadrature from `scipy`, plotting on the diagrams the field uses - with the pair
that will recur through the course, benzene and toluene, and the one that will keep it honest, ethanol and
water.
"""),
    setup_cell(["from scipy import optimize, integrate", "from engsep import datasets, vle"]),
    md("## Vapour pressures: the Antoine equation"),
    code(r"""for name in ("benzene", "toluene", "ethanol", "water"):
    A, B, C, rng_, V = vle.ANTOINE[name]
    print(f"{name:<9} log10 P/mmHg = {A} - {B}/({C} + T/degC)   valid {rng_[0]}..{rng_[1]} degC   boiling point {vle.boiling_point(name)-273.15:.1f} degC")
T = np.linspace(280, 400, 200)
for name in ("benzene", "toluene"):
    plt.semilogy(T - 273.15, vle.antoine_pressure(name, T) / 1e3, label=name)
plt.axhline(101.325, color="k", ls=":", label="1 atm"); plt.xlabel("T (degC)"); plt.ylabel("vapour pressure (kPa)"); plt.legend(); plt.show()"""),
    md(r"""
Three constants per compound reproduce the vapour pressure over a hundred degrees to better than a percent,
and the temperature at which the curve crosses 1 atm is the normal boiling point. The library's table checks
its boiling points against NIST to within a kelvin - the first thing one should verify before trusting any
constants copied from a handbook.

## Mole fractions, balances and the lever rule
"""),
    code(r"""F, zF = 100.0, 0.45                                                 # kmol/h of feed, benzene mole fraction
xD, xB = 0.95, 0.05
D = F * (zF - xB) / (xD - xB)
print(f"benzene balance: D = F (zF - xB)/(xD - xB) = {D:.1f} kmol/h, B = {F - D:.1f} kmol/h; benzene recovery in the distillate {D*xD/(F*zF):.1%}")
xs = np.array([xB, zF, xD])
plt.plot(xs, [0, 0, 0], "k-"); plt.plot(xs, [0, 0, 0], "o"); plt.text(xB, 0.02, "B"); plt.text(zF, 0.02, "F"); plt.text(xD, 0.02, "D"); plt.yticks([]); plt.xlabel("benzene mole fraction"); plt.title("the lever rule: F sits between B and D, closer to the larger product"); plt.show()"""),
    md(r"""
Every separation is a material balance first: the feed splits into products whose compositions bracket it,
and the amounts follow from the lever rule. The recovery - what fraction of a component ends up where it
should - is the number a plant is paid for.

## Root finding and quadrature
"""),
    code(r"""P = 101325.0
# bubble point of a 50/50 benzene-toluene liquid: sum x_i Psat_i = P
T_b = optimize.brentq(lambda T: 0.5 * vle.antoine_pressure("benzene", T) + 0.5 * vle.antoine_pressure("toluene", T) - P, 300, 400)
print(f"bubble point of 50/50 benzene-toluene: {T_b - 273.15:.1f} degC (library: {vle.bubble_T(('benzene', 'toluene'), 0.5, ideal=True) - 273.15:.1f})")
# a PFR-like integral the course will meet as the number of transfer units: int dy/(y - y*)
ntu, _ = integrate.quad(lambda y: 1 / (y - 0.5 * y), 0.002, 0.02)
print(f"a transfer-unit integral by quadrature: {ntu:.3f} (analytic 2 ln 10 = {2*np.log(10):.3f})")"""),
    md(r"""
`brentq` finds the temperature at which a sum of vapour pressures equals the total pressure - the bubble
point - and `quad` evaluates the integrals that size packed towers. The library uses the same two tools
behind every function, so a hand calculation with them is always available as a check.

## Diagrams: the x-y plot
"""),
    code(r"""d = pd.read_csv(datasets.path("vle_ethanol_water.csv"), comment="#")
xy_bt = vle.xy(("benzene", "toluene"), n=41, ideal=True)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(xy_bt["x"], xy_bt["y"], label="benzene-toluene (Raoult)")
ax.plot(d.x_ethanol, d.y_ethanol, "o-", ms=4, label="ethanol-water (data)")
ax.plot([0, 1], [0, 1], "k:"); ax.set(xlabel="x (liquid mole fraction)", ylabel="y (vapour mole fraction)", xlim=(0, 1), ylim=(0, 1)); ax.legend(); plt.show()
print(datasets.describe("vle_ethanol_water.csv"))"""),
    md(r"""
The x-y diagram is the working drawing of distillation: the further the curve from the diagonal, the easier
the separation. Benzene-toluene bows uniformly; ethanol-water bows strongly at low ethanol and collapses onto
the diagonal at 89 mol % - the azeotrope, where distillation stops. Everything in notebooks 01-05 is built on
this picture.

## The course data
"""),
    code(r"""for name in datasets.available():
    print(f"{name:<36} {datasets.describe(name)[:88]}")"""),
    md(r"""
Literature tables with their sources (VLE data, permeabilities, solubilities, tie lines) and pseudo-experimental
files generated from the library's own models with stated noise and faults; the project of notebook 17
works from a month of column data, faults included, to a retrofit proposal.
"""),
]

NOTEBOOKS["01_phase_equilibrium_and_vle"] = [
    md(r"""
# 01 · Phase equilibrium and vapour-liquid equilibrium

Distillation, flash and absorption all rest on one question: at a given temperature and pressure, what
vapour is in equilibrium with a given liquid? For ideal mixtures Raoult's law answers with vapour pressures
alone; for the mixtures that matter - alcohols and water, ketones and hydrocarbons - activity coefficients
bend the answer, sometimes into an azeotrope. This notebook builds T-x-y and x-y diagrams for both cases,
fits the Wilson model against data, and locates azeotropes.
"""),
    setup_cell(["from engsep import datasets, vle"]),
    md("## Raoult's law and the ideal binary"),
    code(r"""pair = ("benzene", "toluene")
d = vle.Txy(pair, n=41, ideal=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(d["x"], d["T"] - 273.15, label="bubble (liquid)"); ax[0].plot(d["y"], d["T"] - 273.15, label="dew (vapour)"); ax[0].set(xlabel="x, y benzene", ylabel="T (degC)", title="T-x-y at 1 atm"); ax[0].legend()
ax[1].plot(d["x"], d["y"]); ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set(xlabel="x", ylabel="y", title="x-y"); plt.show()
for x1 in (0.2, 0.5, 0.8):
    T, y = vle.bubble_y(pair, x1, ideal=True)
    print(f"x = {x1}: bubble point {T-273.15:.1f} degC, y = {y:.3f}, alpha = {vle.relative_volatility(pair, T, ideal=True):.2f}")"""),
    md(r"""
For an ideal pair the bubble and dew curves form a lens between the two boiling points, and the relative
volatility - the ratio of vapour pressures - is nearly constant (2.3 to 2.6 across the range). A constant α
makes the equilibrium curve a one-parameter formula, $y = \alpha x/(1 + (\alpha - 1)x)$, which is why so much
of distillation theory is written in terms of it.

## Non-ideal mixtures: the Wilson model
"""),
    code(r"""pair = ("ethanol", "water")
data = pd.read_csv(datasets.path("vle_ethanol_water.csv"), comment="#")
model = vle.Txy(pair, n=61)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(model["x"], model["T"] - 273.15, label="Wilson bubble"); ax[0].plot(model["y"], model["T"] - 273.15, label="Wilson dew"); ax[0].plot(data.x_ethanol, data.T_C, "ko", ms=4, label="data (liquid)"); ax[0].set(xlabel="x, y ethanol", ylabel="T (degC)"); ax[0].legend()
ax[1].plot(model["x"], model["y"], label="Wilson"); ax[1].plot(data.x_ethanol, data.y_ethanol, "ko", ms=4, label="data"); ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set(xlabel="x", ylabel="y"); ax[1].legend(); plt.show()
err = np.abs(np.interp(data.x_ethanol, model["x"], model["y"]) - data.y_ethanol)
print(f"Wilson vs data: mean |dy| = {err.mean():.4f}, max {err.max():.4f}")"""),
    md(r"""
Ethanol-water is far from ideal: the vapour is much richer in ethanol than Raoult's law predicts at low
concentration, and the curves cross the diagonal at the azeotrope. The Wilson equation, with two parameters
fitted to the data, reproduces the whole curve within a few thousandths in $y$ - good enough for column
design, and a demonstration that the activity coefficient is the whole story of non-ideality here.

## Activity coefficients
"""),
    code(r"""x = np.linspace(0, 1, 101)
g1, g2 = vle.wilson_gamma(pair, x, 350.0)
plt.plot(x, g1, label="gamma ethanol"); plt.plot(x, g2, label="gamma water"); plt.axhline(1, color="k", ls=":"); plt.xlabel("x ethanol"); plt.ylabel("activity coefficient at 350 K"); plt.legend(); plt.show()
print(f"infinite-dilution activity coefficients: ethanol in water {g1[0]:.2f}, water in ethanol {g2[-1]:.2f}")"""),
    md(r"""
Each component behaves ideally in its own pure liquid ($\gamma$ = 1) and most non-ideally at infinite
dilution, where an ethanol molecule is surrounded by water it would rather not be in ($\gamma^\infty \approx 6$).
Positive deviations from Raoult's law - $\gamma > 1$ - raise the vapour pressure of the mixture and, if
large enough, produce a minimum-boiling azeotrope.

## Azeotropes
"""),
    code(r"""for c1, c2 in (("ethanol", "water"), ("2-propanol", "water"), ("acetone", "methanol"), ("methanol", "water"), ("acetone", "water"), ("benzene", "toluene")):
    az = vle.azeotrope(c1, c2)
    print(f"{c1}-{c2:<10}: " + (f"{az['kind']} azeotrope at x = {az['x']:.3f}, {az['T']-273.15:.1f} degC" if az else "no azeotrope"))"""),
    md(r"""
The azeotrope is where the equilibrium curve crosses the diagonal: the vapour has the liquid's composition
and boiling changes nothing. Ethanol cannot be distilled past 89 mol % (96 wt %) at 1 atm - the reason for
molecular sieves and entrainers in fuel-ethanol plants. The library locates azeotropes by finding the
crossing, and reports none for methanol-water, which is non-ideal but not enough to cross.

## Pressure changes everything
"""),
    code(r"""for P_atm in (0.2, 1.0, 5.0):
    az = vle.azeotrope("ethanol", "water", P=P_atm * 101325.0)
    a_bt = vle.geometric_mean_alpha(("benzene", "toluene"), P=P_atm * 101325.0, ideal=True)
    print(f"{P_atm:4.1f} atm: ethanol-water azeotrope at x = {az['x']:.3f} ({az['T']-273.15:.0f} degC); benzene-toluene alpha = {a_bt:.2f}")"""),
    md(r"""
Lower pressure lowers the boiling points, raises the relative volatility (vapour pressures diverge at low
temperature) and shifts the ethanol azeotrope towards pure ethanol - which is why vacuum columns separate
better and why pressure-swing distillation can break some azeotropes. Pressure is the designer's free
variable in VLE.
"""),
]

NOTEBOOKS["02_flash_separations"] = [
    md(r"""
# 02 · Flash separations

The simplest separator is a vessel: heat a liquid, drop the pressure, and let vapour and liquid part company
once. The flash sets the vocabulary - bubble and dew points, K-values, the Rachford-Rice equation - that
every stage of every column reuses. This notebook solves isothermal and adiabatic flashes for binary and
multicomponent mixtures, and shows what one stage can and cannot do.
"""),
    setup_cell(["from engsep import flash, vle"]),
    md("## Bubble and dew points of a mixture"),
    code(r"""names, z = ["n-butane", "n-pentane", "n-hexane"], [0.3, 0.4, 0.3]
P = 2e5
Tb, Td = flash.bubble_point_T(names, z, P), flash.dew_point_T(names, z, P)
print(f"C4/C5/C6 30/40/30 at 2 bar: bubble point {Tb-273.15:.1f} degC, dew point {Td-273.15:.1f} degC")
print("K-values at the bubble point:", {n: round(vle.k_value(n, Tb, P), 3) for n in names})"""),
    md(r"""
Between the bubble point (the first bubble of vapour) and the dew point (the last drop of liquid) the
mixture is two-phase. K-values $y_i/x_i$ - vapour pressure over total pressure for an ideal mixture - are
above one for the lights and below one for the heavies; a component with K = 1 does not separate at all.

## The Rachford-Rice equation
"""),
    code(r"""T = 0.5 * (Tb + Td)
res = flash.flash_TP(names, z, T, P)
print(f"flash at {T-273.15:.1f} degC: V/F = {res['V_over_F']:.3f}")
print(pd.DataFrame({"component": names, "z": z, "K": np.round(res["K"], 3), "x (liquid)": np.round(res["x"], 3), "y (vapour)": np.round(res["y"], 3)}).to_string(index=False))
psi = np.linspace(0, 1, 200)
f = [np.sum(np.array(z) * (res["K"] - 1) / (1 + p * (res["K"] - 1))) for p in psi]
plt.plot(psi, f); plt.axhline(0, color="k", ls=":"); plt.axvline(res["V_over_F"], color="C3", ls="--"); plt.xlabel("V/F"); plt.ylabel("Rachford-Rice function"); plt.title("monotone: one root"); plt.show()"""),
    md(r"""
Rachford and Rice combined the component balances into one equation in the vapour fraction, monotone
between the poles, so a root finder cannot miss. One flash enriches the vapour in butane from 30 to 45 %:
useful, but nowhere near a separation. Columns exist because one stage is not enough.

## Flash curve and the T-V/F diagram
"""),
    code(r"""Ts = np.linspace(Tb, Td, 50)
VF = [flash.flash_TP(names, z, T_, P)["V_over_F"] for T_ in Ts]
yC4 = [flash.flash_TP(names, z, T_, P)["y"][0] for T_ in Ts]
fig, ax = plt.subplots(); ax.plot(Ts - 273.15, VF, label="V/F"); ax.plot(Ts - 273.15, yC4, label="y butane"); ax.set(xlabel="T (degC)", ylabel="fraction"); ax.legend(); plt.show()"""),
    md(r"""
As the temperature rises through the two-phase region the vapour fraction goes from 0 to 1 and the vapour
gets *leaner* in the light component: the first bubble is the richest vapour the mixture can produce, the last
drop the leanest liquid. A flash trades recovery for purity, and the flash curve is the trade.

## Adiabatic flash: the pressure drop pays for the vapour
"""),
    code(r"""cp = [140.0, 167.0, 195.0]                                          # J/(mol K), liquid
dH = [22.4e3, 25.8e3, 28.9e3]                                        # J/mol
for P_flash in (1.5e5, 1.0e5, 0.5e5):
    ad = flash.adiabatic_flash(names, z, T_feed=Tb, P=P_flash, cp_liquid=cp, dH_vap=dH)
    print(f"liquid at its 2 bar bubble point ({Tb-273.15:.1f} degC) flashed to {P_flash/1e5:.1f} bar: T = {ad['T']-273.15:.1f} degC, V/F = {ad['V_over_F']:.3f}")"""),
    md(r"""
Throttling a hot liquid through a valve makes vapour at the expense of the liquid's own sensible heat: the
temperature falls until the enthalpy balance closes. The adiabatic flash couples the energy balance to the
phase split - a root find within a root find - and it is how refinery feeds, LNG and refrigerants are
partially vaporised without a heater.

## Binary flash on the x-y diagram
"""),
    code(r"""pair, z1 = ("benzene", "toluene"), 0.5
xy = vle.xy(pair, n=41, ideal=True)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(xy["x"], xy["y"]); ax.plot([0, 1], [0, 1], "k:")
for VF_ in (0.2, 0.5, 0.8):
    T_ = __import__("scipy").optimize.brentq(lambda T: flash.flash_TP(list(pair), [z1, 1 - z1], T, 101325.0)["V_over_F"] - VF_, vle.bubble_T(pair, z1, ideal=True), vle.dew_T(pair, z1, ideal=True))
    r = flash.flash_TP(list(pair), [z1, 1 - z1], T_, 101325.0)
    xline = np.linspace(0, 1, 2); ax.plot(xline, -(1 - VF_) / VF_ * (xline - z1) + z1, "--", lw=1, label=f"V/F = {VF_}: x = {r['x'][0]:.2f}, y = {r['y'][0]:.2f}")
    ax.plot(r["x"][0], r["y"][0], "o")
ax.set(xlabel="x benzene", ylabel="y benzene", title="flash operating lines of slope -(1 - V/F)/(V/F)"); ax.legend(fontsize=8); plt.show()"""),
    md(r"""
On the x-y diagram a flash is a straight line through $(z, z)$ with slope $-(1 - \psi)/\psi$: vertical for
$\psi$ = 0 (bubble point), horizontal for $\psi$ = 1 (dew point); its intersection with the equilibrium curve
is the product pair. The same geometry - operating line meets equilibrium curve - is the McCabe-Thiele method
of the next notebook, applied stage after stage.
"""),
]

NOTEBOOKS["03_binary_distillation_mccabe_thiele"] = [
    md(r"""
# 03 · Binary distillation: the McCabe-Thiele method

A distillation column is a stack of flashes with the vapour of each feeding the stage above and the liquid the
stage below, and reflux at the top to keep the vapour climbing towards purity. McCabe and Thiele (1925)
showed that with constant molar overflow the whole column is a staircase on the x-y diagram. This notebook
draws it: operating lines, the q-line, stage stepping, feed location, minimum reflux, minimum stages and the
reflux-versus-stages trade-off.
"""),
    setup_cell(["from engsep import distillation as dist", "from engsep import vle", "from engsep.vle import y_from_alpha"]),
    md("## Operating lines and the staircase"),
    code(r"""alpha = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
y_eq = lambda x: y_from_alpha(x, alpha)
xD, xB, zF, R, q = 0.95, 0.05, 0.45, 2.0, 1.0
mt = dist.mccabe_thiele(y_eq, xD, xB, zF, R, q)
def draw(mt, title):
    xs = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, y_eq(xs), "k", label="equilibrium"); ax.plot([0, 1], [0, 1], "k:")
    ol = mt["lines"]
    ax.plot([ol["x_int"], xD], [ol["y_int"], xD], "C0", label="rectifying line"); ax.plot([xB, ol["x_int"]], [xB, ol["y_int"]], "C1", label="stripping line")
    if ol["q_line"] is not None:
        ax.plot([zF, ol["x_int"]], [zF, ol["y_int"]], "C2", label="q-line")
    else:
        ax.plot([zF, zF], [zF, ol["y_int"]], "C2", label="q-line (saturated liquid)")
    ax.step(mt["x"], mt["y"], "C3", where="post", lw=1)
    ax.set(xlabel="x benzene", ylabel="y benzene", xlim=(0, 1), ylim=(0, 1), title=title); ax.legend(loc="lower right"); plt.show()
draw(mt, f"benzene-toluene, R = {R}, q = {q}: {mt['stages']} stages, feed on stage {mt['feed_stage']}")
print(f"{mt['stages']} theoretical stages including the reboiler; feed on stage {mt['feed_stage']} from the top")"""),
    md(r"""
The rectifying line comes from a balance around the top of the column, slope $R/(R+1)$ through $(x_D, x_D)$;
the stripping line from a balance around the bottom, through $(x_B, x_B)$; they meet on the q-line, which
encodes the feed's thermal condition. Stepping between the operating lines and the equilibrium curve counts
the stages, and the step that crosses the intersection is the feed stage.

## The feed condition: the q-line
"""),
    code(r"""R_cmp = 3.0                                                          # above the minimum for every feed condition below
for q_, label in ((1.3, "subcooled liquid"), (1.0, "saturated liquid"), (0.5, "half vaporised"), (0.0, "saturated vapour")):
    mt_ = dist.mccabe_thiele(y_eq, xD, xB, zF, R_cmp, q_)
    print(f"q = {q_:4.1f} ({label:<17}): at R = {R_cmp}: {mt_['stages']:2d} stages, feed stage {mt_['feed_stage']}; minimum reflux {dist.minimum_reflux(y_eq, xD, zF, q_)['R_min']:.3f}")"""),
    md(r"""
The q-line has slope $q/(q-1)$: vertical for a saturated liquid, horizontal for a saturated vapour. A
vapour feed needs almost twice the minimum reflux of a liquid feed (the rectifying section must carry the
extra vapour) and, at the same reflux, more stages; a subcooled feed condenses vapour in the stripping
section and helps - at R = 2 the vapour feed cannot even be separated, which is why the comparison is run
at R = 3. Feed preheating is a lever on the
whole column, decided by this geometry and the utility costs.

## Minimum reflux and minimum stages
"""),
    code(r"""rm = dist.minimum_reflux(y_eq, xD, zF, q)
N_min = dist.minimum_stages(y_eq, xD, xB)
print(f"minimum reflux R_min = {rm['R_min']:.3f} (pinch at x = {rm['x_pinch']:.2f}); Fenske: N_min = {dist.fenske(xD, xB, alpha):.2f} -> {N_min} stages at total reflux")
Rs = np.linspace(1.05 * rm["R_min"], 8, 40)
Ns = [dist.mccabe_thiele(y_eq, xD, xB, zF, R_, q)["stages"] for R_ in Rs]
fig, ax = plt.subplots(); ax.plot(Rs / rm["R_min"], Ns, "o-", ms=3); ax.axhline(N_min, color="k", ls=":", label="N_min (total reflux)"); ax.axvline(1.0, color="C3", ls="--", label="R_min"); ax.set(xlabel="R / R_min", ylabel="theoretical stages", ylim=(0, 30)); ax.legend(); plt.show()"""),
    md(r"""
At minimum reflux the operating lines touch the equilibrium curve and the staircase needs infinitely many
steps at the pinch; at total reflux the operating lines coincide with the diagonal and the stages are fewest.
Real columns run at 1.2-1.5 times $R_{min}$: the steep part of the curve is where stages are expensive and
the flat part where reflux (reboiler steam) is. The optimum is an economic balance, computed in notebook 04.

## Plate efficiency
"""),
    code(r"""for E in (1.0, 0.8, 0.6):
    mt_ = dist.mccabe_thiele(y_eq, xD, xB, zF, R, q, murphree=E)
    print(f"Murphree vapour efficiency {E:.0%}: {mt_['stages']} stages (real trays {mt_['stages'] - 1} + reboiler)")
print(f"O'Connell overall efficiency for alpha = {alpha:.2f} and mu = 0.3 mPa s: {dist.oconnell_efficiency(alpha, 0.3):.2f}")"""),
    md(r"""
Real trays do not reach equilibrium: with a Murphree efficiency of 60 % the staircase takes shorter steps
and the column needs half again as many trays. O'Connell's correlation predicts overall efficiencies from
viscosity and relative volatility - about 0.5-0.6 for benzene-toluene at its 0.3 mPa s - and the plant column
of notebook 17 will be judged against it.

## A hard separation: ethanol-water
"""),
    code(r"""y_ew = lambda x: vle.equilibrium_y(("ethanol", "water"), x)
mt_ew = dist.mccabe_thiele(y_ew, 0.85, 0.02, 0.30, R=3.0)
xs = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 6)); ax.plot(xs, [y_ew(x) for x in xs], "k"); ax.plot([0, 1], [0, 1], "k:"); ax.step(mt_ew["x"], mt_ew["y"], "C3", where="post", lw=1); ax.set(xlabel="x ethanol", ylabel="y ethanol", title=f"ethanol-water to 85 mol %: {mt_ew['stages']} stages at R = 3"); plt.show()
rm_ew = dist.minimum_reflux(y_ew, 0.85, 0.30)
print(f"minimum reflux {rm_ew['R_min']:.2f}; tangent pinch: {rm_ew['tangent_pinch']}; stages at R = 3: {mt_ew['stages']}")"""),
    md(r"""
Near the azeotrope the equilibrium curve hugs the diagonal and each stage gains almost nothing: the last
few percent of purity cost as many stages as the first eighty. The library's equilibrium curve here is the
Wilson model, checked against the data in notebook 01 - McCabe-Thiele works with any curve, which is its
enduring advantage over shortcut formulas that assume constant α.
"""),
]

NOTEBOOKS["04_shortcut_and_multicomponent_distillation"] = [
    md(r"""
# 04 · Shortcut design and multicomponent distillation

Before a rigorous simulation, a designer wants the size of a column in minutes: the Fenske equation for the
minimum stages, Underwood's for the minimum reflux, Gilliland's correlation for the stages at the chosen
reflux, Kirkbride's for the feed location. The same tools handle multicomponent feeds through the light and
heavy keys. This notebook runs the shortcut, compares it with McCabe-Thiele, optimises the reflux ratio
against cost, and designs a depropaniser.
"""),
    setup_cell(["from engsep import distillation as dist", "from engsep import vle", "from engsep.vle import y_from_alpha"]),
    md("## Fenske, Underwood, Gilliland for the benzene-toluene column"),
    code(r"""alpha = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
xD, xB, zF = 0.95, 0.05, 0.45
sc = dist.fug(alpha, zF, xD, xB, R_over_Rmin=1.3)
print(f"alpha = {alpha:.2f}: N_min = {sc['N_min']:.2f} (Fenske), R_min = {sc['R_min']:.3f} (Underwood), at R = 1.3 R_min = {sc['R']:.2f}: N = {sc['N']:.1f} stages (Gilliland)")
y_eq = lambda x: y_from_alpha(x, alpha)
mt = dist.mccabe_thiele(y_eq, xD, xB, zF, sc["R"])
print(f"McCabe-Thiele at the same reflux: {mt['stages']} stages, feed stage {mt['feed_stage']}")
kb = dist.kirkbride(sc["N"], B=55.0, D=45.0, zF_hk=1 - zF, zF_lk=zF, xB_lk=xB, xD_hk=1 - xD)
print(f"Kirkbride feed stage: {kb['feed_stage_from_top']:.1f} from the top")"""),
    md(r"""
The shortcut lands within a stage or two of the rigorous count - Gilliland's correlation is an empirical fit
to hundreds of columns, good to perhaps 10-20 % - and Kirkbride puts the feed where McCabe-Thiele does. For a
first estimate, a cost comparison, or a sanity check on a simulator, the three equations are all one needs.

## Reflux ratio versus cost
"""),
    code(r"""RR = np.linspace(1.05, 3.0, 40)
rows = []
for r_ in RR:
    R = r_ * sc["R_min"]
    N = dist.gilliland(R, sc["R_min"], sc["N_min"])
    capital = 80e3 * N / 0.7                                             # $ per real tray (efficiency 0.7), annualised over 10 y below
    D = 45.0                                                             # kmol/h
    steam = (R + 1) * D * 30e3 / 3600 * 8000 * 0.02 / 1e3                # kW of reboiler duty x hours x $/kWh... in k$/y
    rows.append((r_, N, capital / 10 / 1e3, steam, capital / 10 / 1e3 + steam))
rows = np.array(rows)
i = np.argmin(rows[:, 4])
fig, ax = plt.subplots(); ax.plot(rows[:, 0], rows[:, 2], label="annualised capital (trays)"); ax.plot(rows[:, 0], rows[:, 3], label="steam"); ax.plot(rows[:, 0], rows[:, 4], "k", lw=2, label="total"); ax.axvline(rows[i, 0], color="C3", ls="--"); ax.set(xlabel="R / R_min", ylabel="k$/year"); ax.legend(); plt.show()
print(f"the total cost is minimised at R/R_min = {rows[i, 0]:.2f} ({rows[i, 1]:.1f} stages)")"""),
    md(r"""
Stages cost capital once; reflux costs steam every hour. The total has a shallow minimum near 1.2-1.5 times
the minimum reflux - the textbook rule of thumb, derived rather than remembered - and the flatness of the
minimum is why the rule is safe: being 20 % off the optimum costs a few percent.

## Multicomponent: keys and non-keys
"""),
    code(r"""names = ["propane", "isobutane", "n-butane", "n-pentane"]
z = np.array([0.30, 0.15, 0.35, 0.20])
T_ref = 320.0                                                       # a representative temperature, K
alphas = np.array([vle.antoine_pressure(n, T_ref) for n in names]) / vle.antoine_pressure("n-butane", T_ref)   # relative to the heavy key
print("relative volatilities (to n-butane):", {n: round(float(a), 2) for n, a in zip(names, alphas)})
# depropaniser: light key propane (99 % to the distillate), heavy key n-butane (1 % to the distillate); isobutane distributes, pentane all to the bottoms
d_frac = np.array([0.99, 0.6, 0.01, 0.0])
D_i = z * d_frac; B_i = z - D_i
xD, xB = D_i / D_i.sum(), B_i / B_i.sum()
N_min = dist.fenske(xD[0], xB[0], alphas[0], xD_hk=xD[2], xB_hk=xB[2])
uw = dist.underwood(alphas, z, xD, q=1.0)
R = 1.3 * uw["R_min"]
N = dist.gilliland(R, uw["R_min"], N_min)
print(f"N_min = {N_min:.2f}, R_min = {uw['R_min']:.3f} (Underwood roots {np.round(uw['theta'], 3)}), N = {N:.1f} at R = {R:.2f}")
print(pd.DataFrame({"component": names, "feed": z, "distillate": np.round(xD, 4), "bottoms": np.round(xB, 4)}).to_string(index=False))"""),
    md(r"""
With more than two components the designer chooses two adjacent *keys* and specifies their split; the
lighter non-keys go almost entirely overhead, the heavier to the bottoms, and a component between the keys
(isobutane here) distributes. Fenske and Underwood then work as for a binary, with the caveat that the
assumed non-key splits should be checked with Fenske at the end.

## Checking the non-key distribution with Fenske
"""),
    code(r"""def fenske_splits(d_frac):
    D_i = z * d_frac; B_i = z - D_i
    xD, xB = D_i / D_i.sum(), B_i / B_i.sum()
    N_min = dist.fenske(xD[0], xB[0], alphas[0], xD_hk=xD[2], xB_hk=xB[2])
    db = (D_i[2] / B_i[2]) * alphas ** N_min                       # (d_i/b_i) = (d_hk/b_hk) alpha_i^N_min at total reflux
    return N_min, db / (1 + db)
for it in range(4):
    N_min, splits = fenske_splits(d_frac)
    print(f"iteration {it}: assumed isobutane split {d_frac[1]:.3f} -> Fenske gives {splits[1]:.3f}  (N_min = {N_min:.2f}; propane {splits[0]:.3f}, pentane {splits[3]:.4f})")
    d_frac = np.array([0.99, splits[1], 0.01, 0.0])
D_i = z * d_frac; B_i = z - D_i
xD, xB = D_i / D_i.sum(), B_i / B_i.sum()
uw = dist.underwood(alphas, z, xD, q=1.0)
print(f"\nconverged design: N_min = {N_min:.2f}, R_min = {uw['R_min']:.3f}, N = {dist.gilliland(1.3 * uw['R_min'], uw['R_min'], N_min):.1f} stages at 1.3 R_min")"""),
    md(r"""
Fenske's equation applies to every component, not just the keys: at total reflux each component's
distillate-to-bottoms ratio scales as $\alpha_i^{N_{min}}$. Propane and pentane split essentially completely,
as assumed - but the isobutane guess of 60 % overhead was wrong: with a volatility only 1.4 times the heavy
key's, it goes 10 % overhead. Replacing the guess converges in one iteration: $N_{min}$ does not change (only
the key ratios enter Fenske), but the distillate is much smaller and lighter than assumed and $R_{min}$ falls
from 2.07 to 1.14. The consistency check is what separates a design from a guess.
"""),
]

NOTEBOOKS["05_batch_distillation_and_column_efficiency"] = [
    md(r"""
# 05 · Batch distillation and column efficiency

Small-volume and multipurpose plants - pharmaceuticals, fine chemicals, solvent recovery - distil in batches:
a still is charged, heated, and the distillate collected as the charge grows poorer. This notebook follows
the Rayleigh equation for simple distillation and batch rectification at constant reflux, then turns to
what limits real columns: tray efficiency, packing HETP, and the flooding that sets a column's diameter.
"""),
    setup_cell(["from engsep import distillation as dist", "from engsep import vle", "from engsep.vle import y_from_alpha"]),
    md("## Simple (Rayleigh) distillation"),
    code(r"""alpha = 2.4
y_eq = lambda x: y_from_alpha(x, alpha)
x0 = 0.5
xs = np.linspace(0.5, 0.05, 46)
ray = [dist.rayleigh(y_eq, x0, x_) for x_ in xs]
fig, ax = plt.subplots(); ax.plot(xs, [r["W"] for r in ray], label="charge remaining W/W0"); ax.plot(xs, [r["y_avg"] for r in ray], label="average distillate composition"); ax.plot(xs, [y_eq(x_) for x_ in xs], "k:", label="instantaneous distillate y(x)")
ax.invert_xaxis(); ax.set(xlabel="still composition x (decreasing with time)", ylabel="fraction"); ax.legend(); plt.show()
r = dist.rayleigh(y_eq, x0, 0.1)
print(f"boiling a 50 % charge down to 10 %: {r['D']:.0%} distilled, average distillate {r['y_avg']:.3f} - hardly a separation")"""),
    md(r"""
Rayleigh's equation integrates the equilibrium curve as the still is depleted: the distillate is richest at
the start and poorer with every drop, and the accumulated product is a blend. Simple distillation cannot
produce a pure product; it can strip a light impurity, which is what it is used for.

## Batch rectification at constant reflux
"""),
    code(r"""bcr = dist.batch_constant_reflux(y_eq, x0=0.5, x_final=0.1, R=4.0, N=4)
fig, ax = plt.subplots(); ax.plot(bcr["x_still"], bcr["x_D"], label="instantaneous distillate x_D"); ax.plot(bcr["x_still"], bcr["y_avg"], label="accumulated distillate"); ax.plot(bcr["x_still"], bcr["W"], "k:", label="charge remaining")
ax.invert_xaxis(); ax.set(xlabel="still composition", ylabel="fraction", title="4 stages, R = 4"); ax.legend(); plt.show()
i = np.searchsorted(-bcr["x_D"], -0.8)
print(f"the distillate stays above 80 % until the still reaches x = {bcr['x_still'][i]:.2f}, having collected {bcr['D'][i]:.0%} of the charge at an average of {bcr['y_avg'][i]:.3f}")"""),
    md(r"""
A column above the still, run at constant reflux, keeps the distillate pure while the still is rich, then
lets it slide; the operator stops when the distillate falls below specification and the rest is a second
cut. The alternative - raising the reflux continuously to hold the distillate constant - collects less per
hour but keeps every drop on spec. Batch distillation is scheduling as much as separation.

## Tray efficiency: from theoretical stages to real trays
"""),
    code(r"""rows = []
for mu in (0.1, 0.3, 1.0, 3.0):
    for a in (1.5, 2.4, 5.0):
        rows.append((mu, a, dist.oconnell_efficiency(a, mu)))
print(pd.DataFrame(rows, columns=["viscosity (mPa s)", "alpha", "O'Connell efficiency"]).round(2).pivot(index="viscosity (mPa s)", columns="alpha", values="O'Connell efficiency").to_string())"""),
    md(r"""
O'Connell's correlation captures the two things that hurt tray efficiency: viscous liquids (slow diffusion)
and high relative volatility (the vapour changes composition fast, so a tray's mixing falls behind). Typical
hydrocarbon columns reach 60-80 %; viscous or wide-boiling ones 30-50 %. The real tray count is the
theoretical count divided by this.

## Packed columns: HETP
"""),
    code(r"""for packing, hetp in (("25 mm Pall rings", 0.5), ("50 mm Pall rings", 0.75), ("structured (Mellapak 250Y)", 0.35), ("structured (Mellapak 500Y)", 0.2)):
    print(f"{packing:<28}: HETP {hetp} m -> 12 theoretical stages need {dist.packed_height(12, hetp):.1f} m of packing")"""),
    md(r"""
Packing replaces trays with a continuous surface; its height equivalent to a theoretical plate (HETP) is
measured, not derived - a fraction of a metre for structured packings, which is why they dominate vacuum
and revamp applications where pressure drop and height matter.

## Flooding and column diameter
"""),
    code(r"""def flooding_velocity(rho_L, rho_V, sigma, C_sb=0.08):
    # Souders-Brown: u_f = C_sb sqrt((rho_L - rho_V)/rho_V) with a capacity factor C_sb (m/s) including the surface-tension correction
    return C_sb * (sigma / 0.02) ** 0.2 * np.sqrt((rho_L - rho_V) / rho_V)
rho_L, rho_V, sigma = 800.0, 2.8, 0.021                              # benzene-toluene near the top at 1 atm
V = (2.0 + 1) * 45.0 / 3600 * 29.0                                   # vapour (R+1) D in kmol/s times the molar volume at 80 degC (29 m3/kmol)
u_f = flooding_velocity(rho_L, rho_V, sigma)
u = 0.75 * u_f
area = V / u
print(f"flooding velocity {u_f:.2f} m/s; at 75 % of flooding u = {u:.2f} m/s; vapour {V:.3f} m3/s -> area {area:.2f} m2, diameter {np.sqrt(4*area/np.pi):.2f} m")"""),
    md(r"""
Stages set the height; vapour velocity sets the diameter. Too fast and the vapour carries liquid up the
column (entrainment) or holds it back on the trays (flooding); the Souders-Brown capacity factor gives the
limit, and design runs at 70-80 % of it. A column is two calculations - stages and hydraulics - and this
notebook has now done both.
"""),
]

NOTEBOOKS["06_absorption_and_stripping"] = [
    md(r"""
# 06 · Absorption and stripping

Gas absorption removes a component from a gas by dissolving it in a liquid; stripping does the reverse.
Sweetening natural gas, capturing CO$_2$ from flue gas, scrubbing SO$_2$, recovering solvents: the equipment
is a countercurrent column of trays or packing, and for dilute systems with a linear equilibrium line the
Kremser equation gives the answer in closed form. This notebook designs a tray absorber and a stripper, finds
the minimum solvent rate, and shows the absorption factor as the single number that controls both.
"""),
    setup_cell(["from engsep import absorption as ab"]),
    md("## The absorption factor and the Kremser equation"),
    code(r"""m = 0.8                                                            # equilibrium y = m x (Henry's law in mole-fraction form)
V, y_in = 100.0, 0.02                                               # kmol/h of carrier gas, 2 % solute
fig, ax = plt.subplots()
Ns = np.arange(1, 13)
for A in (0.7, 1.0, 1.4, 2.0):
    ax.plot(Ns, [ab.kremser_fraction_absorbed(A, N) for N in Ns], "o-", ms=3, label=f"A = {A}")
ax.set(xlabel="equilibrium stages", ylabel="fraction of solute absorbed"); ax.legend(); plt.show()
for A in (0.7, 1.4):
    print(f"A = {A}: fraction absorbed with 6 stages {ab.kremser_fraction_absorbed(A, 6):.3f}, with 100 stages {ab.kremser_fraction_absorbed(A, 100):.3f}")"""),
    md(r"""
The absorption factor $A = L/(mV)$ compares the solvent's capacity with the gas load. Below 1 the solvent
cannot absorb everything however many stages are added (the limit is $A$ itself); above 1 recovery approaches
100 % with enough stages; at $A \approx 1.4$ the curve bends where more stages stop paying - Kremser's
equation, and the whole economics of absorption, in one plot.

## Designing a tray absorber
"""),
    code(r"""recovery = 0.95
y_out = y_in * (1 - recovery)
L_min = ab.minimum_solvent(V, y_in, y_out, m)
print(f"minimum solvent for {recovery:.0%} recovery: L_min = {L_min:.1f} kmol/h (A = {ab.absorption_factor(L_min, V, m):.3f})")
for factor in (1.2, 1.5, 2.0):
    L = factor * L_min
    A = ab.absorption_factor(L, V, m)
    N = ab.kremser_stages(A, recovery)
    print(f"L = {factor} L_min = {L:6.1f} kmol/h: A = {A:.2f}, stages = {N:.2f} -> {int(np.ceil(N))} trays")
st = ab.stages_absorber(lambda x: m * x, 1.5 * L_min, V, y_in, y_out)
print(f"\nstage-by-stage at 1.5 L_min: {st['stages']} stages, rich solvent x_out = {st['x_out']:.4f}")"""),
    md(r"""
The minimum solvent rate makes the operating line touch the equilibrium line at the rich end (a pinch:
infinite stages); 1.2-2 times the minimum is the usual design range, and each increase in solvent trades
trays for pumping and regeneration cost. Stepping stages by hand (equilibrium, then operating line, like
McCabe-Thiele) reproduces the Kremser count.

## The operating diagram
"""),
    code(r"""fig, ax = plt.subplots(figsize=(5.5, 5.5))
xs = np.linspace(0, 0.03, 50)
ax.plot(xs, m * xs, "k", label="equilibrium y = m x")
L = 1.5 * L_min
ax.plot([0, st["x_out"]], [y_out, y_in], "C0", label=f"operating line, slope L/V = {L/V:.2f}")
ax.step(st["x"], st["y"], "C3", where="post", lw=1, label="stages")
ax.set(xlabel="x (liquid mole fraction)", ylabel="y (gas mole fraction)"); ax.legend(); plt.show()"""),
    md(r"""
For absorption the operating line lies *above* the equilibrium line (the gas is richer than equilibrium with
the liquid, so solute moves gas to liquid); for stripping it lies below. The stages step between them from
the lean end, and the closer the lines the more stages: the pinch again.

## Stripping: the same equation upside down
"""),
    code(r"""x_in, x_out_target = 0.01, 0.0005                                    # rich solvent in, 95 % stripped
L = 100.0
V_min = L / m * 0.95                                                  # the stripping gas must carry S > phi: V_min = phi L/m... for 95 % stripping S must exceed 0.95
print(f"stripping 95 % of the solute from {L} kmol/h of solvent (m = {m}) needs S = m V/L > 0.95, i.e. V > {V_min:.0f} kmol/h of gas")
for V_strip in (150.0, 200.0, 400.0):
    S = ab.stripping_factor(L, V_strip, m)
    print(f"V = {V_strip:.0f} kmol/h: S = {S:.2f}, stages = {ab.kremser_stages(S, 1 - x_out_target / x_in):.1f}")
try:
    ab.kremser_stages(ab.stripping_factor(L, 30.0, m), 0.95)
except ValueError as e:
    print("V = 30 kmol/h:", e)"""),
    md(r"""
The stripping factor $S = mV/L$ is the absorption factor inverted, and the Kremser equation applies with
the roles of gas and liquid exchanged - including the limit: with $S < 1$ the gas cannot carry the solute
away however many stages there are, and the library says so rather than returning nonsense. Absorber and stripper are usually a pair - the solvent circulates
between them - and the overall design balances solvent rate, stages in each, and the energy to regenerate,
which is what makes amine CO$_2$ capture expensive.

## Temperature and pressure: where the slope m comes from
"""),
    code(r"""H = {"CO2": 1640.0, "H2S": 550.0, "SO2": 40.0, "NH3": 0.9}         # Henry's constants in water at 25 degC, bar (mole-fraction basis, approximate)
for P_bar in (1.0, 10.0, 50.0):
    print(f"P = {P_bar:4.0f} bar: " + ", ".join(f"{g}: m = {ab.henry_slope(Hc, P_bar):8.1f}" for g, Hc in H.items()))"""),
    md(r"""
The equilibrium slope $m = H/P$ falls with pressure, so absorbing at high pressure raises the absorption
factor and cuts the solvent rate - which is why gas sweetening runs at pipeline pressure. CO$_2$ in water has
an enormous Henry's constant (physical absorption is hopeless at 1 bar); ammonia's is tiny (water scrubs it
easily); SO$_2$ sits between. Chemical solvents like amines change the picture by reacting with the solute,
lowering the effective $m$ by orders of magnitude.
"""),
]

NOTEBOOKS["07_packed_towers_and_mass_transfer"] = [
    md(r"""
# 07 · Packed towers and mass transfer

Packed columns have no stages: gas and liquid meet continuously over the packing, and their height is set by
rates, not equilibrium alone. The transfer-unit method splits the height into a number of transfer units
(how hard the separation is) and a height of a transfer unit (how good the packing and the mass-transfer
coefficients are). This notebook computes both, relates overall to film coefficients through the two-film
theory, and estimates diffusivities and mass-transfer coefficients from correlations.
"""),
    setup_cell(["from engsep import absorption as ab", "from engsep import transport as tr"]),
    md("## NTU: the integral"),
    code(r"""m, y_in, y_out, x_in = 0.8, 0.02, 0.002, 0.0
for A in (1.2, 1.5, 2.0, 3.0):
    L_over_V = A * m
    x_of_y = lambda y, L_over_V=L_over_V: x_in + (y - y_out) / L_over_V
    N_num = ab.ntu_gas(y_in, y_out, lambda x: m * x, x_of_y)
    N_col = ab.ntu_colburn(y_in, y_out, m, x_in, A)
    print(f"A = {A}: N_OG = {N_num:.3f} (integral), {N_col:.3f} (Colburn)")"""),
    md(r"""
The number of overall gas-phase transfer units is $\int dy/(y - y^*)$ along the column: the driving force
$y - y^*$ shrinks as the operating and equilibrium lines approach, and the integral grows. For a linear
equilibrium line Colburn's closed form agrees with the integral to four figures - and, like the Kremser
equation, it says that more solvent (a larger $A$) means fewer transfer units.

## HTU: the packing and the coefficients
"""),
    code(r"""G_M = 0.05                                                           # kmol/(m2 s) of gas
for packing, K_y_a in (("25 mm Raschig rings", 0.03), ("25 mm Pall rings", 0.05), ("structured packing", 0.08)):
    H_OG = ab.htu_gas(G_M, K_y_a)
    print(f"{packing:<22}: K_y a = {K_y_a} kmol/(m3 s) -> H_OG = {H_OG:.2f} m -> height for N_OG = {ab.ntu_colburn(y_in, y_out, m, x_in, 1.5):.2f}: {ab.packed_height(H_OG, ab.ntu_colburn(y_in, y_out, m, x_in, 1.5)):.1f} m")"""),
    md(r"""
The height of a transfer unit is the gas flux over the volumetric mass-transfer coefficient: a property of
the packing and the flows, measured by the packing vendor, typically 0.3-1 m. Height = HTU x NTU separates
what the chemistry demands from what the hardware delivers.

## Two-film theory
"""),
    code(r"""for m_ in (0.05, 0.5, 5.0, 50.0):
    tf = ab.overall_from_film(k_y=0.1, k_x=1.0, m=m_)
    print(f"m = {m_:5.2f}: K_y = {tf['K_y']:.4f}, fraction of the resistance in the gas film {tf['gas_resistance_fraction']:.0%}")"""),
    md(r"""
The overall resistance is the sum of the two film resistances, $1/K_y = 1/k_y + m/k_x$, with the equilibrium
slope weighting the liquid side. A very soluble gas (small $m$: ammonia in water) is gas-film controlled;
a sparingly soluble one (large $m$: oxygen, CO$_2$) is liquid-film controlled. Knowing which film controls
tells you what to improve - gas velocity or liquid turbulence.

## Diffusivities from correlations
"""),
    code(r"""rows = []
for gas in ("water", "CO2", "NH3", "SO2", "benzene"):
    rows.append((gas, tr.fuller(gas, "air", 298.15) * 1e4))
print("gas diffusivities in air at 25 degC (Fuller), cm2/s:", {g: round(d, 3) for g, d in rows}, "- measured: water 0.26, CO2 0.16, NH3 0.28")
D_l = tr.wilke_chang(18.02, 2.6, 298.15, 0.89, 34.0)                # CO2 in water (V = 34 cm3/mol)
print(f"CO2 in water at 25 degC (Wilke-Chang): {D_l:.2e} m2/s (measured 1.9e-9)")"""),
    md(r"""
Gas diffusivities are around 0.1-0.3 cm$^2$/s and liquid ones ten thousand times smaller - the reason
liquid films control for sparingly soluble gases and why liquid-side turbulence is worth engineering. The
correlations are good to 5-10 % in gases and 10-25 % in liquids: adequate for design, not for science.

## Mass-transfer coefficients from correlations
"""),
    code(r"""D_g, nu_g, d_p = tr.fuller("CO2", "air", 298.15), 1.6e-5, 0.025
Sc = nu_g / D_g
for u in (0.5, 1.0, 2.0):
    Re = u * d_p / nu_g
    Sh = tr.sherwood_sphere(Re, Sc)
    print(f"u = {u} m/s over 25 mm packing: Re = {Re:.0f}, Sc = {Sc:.2f}, Sh = {Sh:.1f}, k_c = {Sh * D_g / d_p * 100:.2f} cm/s")
print(f"penetration theory for a liquid film exposed 0.1 s: k_L = {tr.penetration_coefficient(1.9e-9, 0.1) * 100:.3f} cm/s")"""),
    md(r"""
Sherwood correlations of the form $Sh = a + b\,Re^{1/2}Sc^{1/3}$ give the gas-side coefficient from the flow;
Higbie's penetration theory gives the liquid side from the time a liquid element is exposed as it trickles
over a packing piece. Both feed $K_y a$, and hence the HTU, from first principles when vendor data are
missing.
"""),
]

NOTEBOOKS["08_liquid_liquid_extraction"] = [
    md(r"""
# 08 · Liquid-liquid extraction

When distillation fails - heat-sensitive products, close boiling points, azeotropes, dilute aqueous streams -
a second liquid can pull the solute out. Extraction is distillation's stage-wise cousin with a solvent in
place of heat, and the same ideas: distribution coefficient in place of relative volatility, extraction factor
in place of the absorption factor, and stages stepped on a diagram. This notebook designs immiscible-solvent
extractions with the Kremser equation and partially miscible ones on triangular coordinates.
"""),
    setup_cell(["from engsep import datasets", "from engsep import extraction as ex"]),
    md("## Immiscible solvent: one stage, many stages, countercurrent"),
    code(r"""K, x_in = 2.5, 0.05                                                 # distribution coefficient (extract/raffinate), feed solute fraction
for S_over_F in (0.5, 1.0, 2.0):
    one = ex.single_stage_immiscible(K, S_over_F, x_in)
    cross = ex.crosscurrent_immiscible(K, S_over_F / 4, x_in, 4)
    counter = ex.countercurrent_immiscible(K, S_over_F, x_in, 4)
    print(f"S/F = {S_over_F}: single stage {one['fraction_extracted']:.3f}; 4 cross-current stages (same total solvent) {cross['fraction_extracted']:.3f}; 4 countercurrent {counter['fraction_extracted']:.3f}")"""),
    md(r"""
With the same total solvent, splitting it over four cross-current stages beats one stage, and running the
four countercurrently beats both - the solvent meets the leanest raffinate last, when it is still fresh.
Countercurrent contact is the rule for the same reason as in absorption and distillation: it keeps the
driving force alive along the whole cascade.

## The extraction factor and the number of stages
"""),
    code(r"""x_out = 0.001                                                         # 98 % recovery
S_min = ex.minimum_solvent_immiscible(K, x_in, x_out)
print(f"minimum solvent S/F = {S_min:.3f} (E = {ex.extraction_factor(K, S_min):.3f})")
for f in (1.2, 1.5, 2.0, 3.0):
    E = ex.extraction_factor(K, f * S_min)
    print(f"S/F = {f} S_min: E = {E:.2f}, stages = {ex.countercurrent_stages_immiscible(K, f * S_min, x_in, x_out):.2f}")"""),
    md(r"""
The extraction factor $E = KS/F$ plays the role of the absorption factor, and the stage count follows the
Kremser equation: infinite at the minimum solvent, falling steeply as $E$ passes 1.3-2. Solvent costs
recovery (usually by distillation), so the design point is again a compromise, typically $E$ between 1.3
and 2.

## Partially miscible systems: the triangular diagram
"""),
    code(r"""d = pd.read_csv(datasets.path("ternary_acetic_water_ether.csv"), comment="#")
tern = ex.Ternary(np.c_[d.xA_raffinate, d.xS_raffinate], np.c_[d.yA_extract, d.yS_extract], np.c_[d.xA_raffinate, d.yA_extract])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(d.xA_raffinate, d.xS_raffinate, "o-", ms=4, label="raffinate branch (water-rich)"); ax.plot(d.yA_extract, d.yS_extract, "s-", ms=4, label="extract branch (ether-rich)")
for r, e in zip(np.c_[d.xA_raffinate, d.xS_raffinate], np.c_[d.yA_extract, d.yS_extract]):
    ax.plot([r[0], e[0]], [r[1], e[1]], "k-", lw=0.6)
ax.set(xlabel="mass fraction acetic acid", ylabel="mass fraction isopropyl ether", title="acetic acid - water - isopropyl ether at 20 degC (right-triangle coordinates)"); ax.legend(); plt.show()
print("distribution coefficient K = yA/xA along the tie lines:", np.round((d.yA_extract / d.xA_raffinate).to_numpy(), 2))"""),
    md(r"""
Water and isopropyl ether are almost immiscible, and acetic acid distributes between them - poorly, with a
distribution coefficient of 0.3-0.8 that changes along the tie lines: ether is a weak solvent for acetic acid.
On triangular (here right-triangle) coordinates the two branches of the binodal curve and the tie lines
carry all the equilibrium information; the lever rule does the balances.

## The Hunter-Nash construction
"""),
    code(r"""F, xF = 100.0, (0.30, 0.0)                                            # kg/h of 30 % acetic acid in water
xS = (0.0, 1.0)                                                       # pure ether
for S in (150.0, 250.0, 400.0, 600.0):
    try:
        hn = ex.hunter_nash(tern, F, xF, S, xS, xR_final=0.02)
        print(f"S = {S:.0f} kg/h: {hn['stages']} stages; extract {hn['E1']:.1f} kg/h at {hn['x_E1'][0]:.3f} acid; raffinate {hn['RN']:.1f} kg/h at {hn['x_RN'][0]:.3f}")
    except ValueError as e:
        print(f"S = {S:.0f} kg/h: {e}")
S_min = ex.minimum_solvent_ternary(tern, F, xF, xS, 0.02)
print(f"minimum solvent about {S_min:.0f} kg/h")"""),
    md(r"""
The construction mixes feed and solvent to a point M, splits M along the line through the final raffinate
into extract and raffinate, and defines an operating point P from which every operating line radiates; the
stages alternate between tie lines and lines through P. With a weak solvent like ether the solvent rate is
several times the feed and the stage count is sensitive to it - the reason acetic acid recovery usually
uses a better solvent.

## Plotting the stages
"""),
    code(r"""hn = ex.hunter_nash(tern, F, xF, 250.0, xS, xR_final=0.02)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(d.xA_raffinate, d.xS_raffinate, "C0-"); ax.plot(d.yA_extract, d.yS_extract, "C1-")
for r, e in zip(hn["raffinates"], hn["extracts"]):
    ax.plot([r[0], e[0]], [r[1], e[1]], "k-", lw=0.8)
ax.plot(*hn["x_M"], "ks", label="M"); ax.plot(*xF, "C2o", label="F"); ax.plot(*xS, "C2^", label="S"); ax.plot(*hn["x_E1"], "C3o", label="E1"); ax.plot(*hn["x_RN"], "C3s", label="R_N")
ax.set(xlabel="mass fraction acetic acid", ylabel="mass fraction ether", title=f"{hn['stages']} stages at S = 250 kg/h"); ax.legend(); plt.show()"""),
    md(r"""
Each tie line drawn is a stage. The diagram shows why partially miscible systems need the full construction:
the extract carries water and the raffinate ether, the flows change from stage to stage, and no single
distribution coefficient describes the system - the library's Kremser functions are for the immiscible
limit, its `Ternary` and `hunter_nash` for the real thing.
"""),
]

NOTEBOOKS["09_leaching_and_washing"] = [
    md(r"""
# 09 · Leaching and washing

Sugar from beet, oil from soybeans, copper from ore, caffeine from coffee: leaching dissolves a solute out
of a solid with a solvent, and the solid carries solution with it from stage to stage. With constant underflow
the countercurrent cascade obeys a Kremser-type equation with a washing factor in place of the absorption
factor; with variable underflow the stages are stepped on a diagram. This notebook designs a countercurrent
leaching train, sizes it for a recovery target, and compares it with cross-current washing.
"""),
    setup_cell(["from engsep import extraction as ex"]),
    md("## Countercurrent leaching with constant underflow"),
    code(r"""solute_in_feed = 100.0                                              # kg of solute per batch of solid
for W in (0.8, 1.0, 1.5, 2.0, 3.0):
    row = [f"{ex.leaching_countercurrent(N, W * 1.0, 1.0, solute_in_feed)['recovery']:.3f}" for N in (1, 2, 4, 8)]
    print(f"washing factor W = {W}: recovery with 1, 2, 4, 8 stages = {row}")"""),
    md(r"""
The washing factor is the ratio of overflow solution to the solution retained in the underflow: below 1 the
solvent cannot carry the solute away however many stages; above 1 recovery climbs towards complete. The
numbers are the Kremser equation again - leaching, absorption and extraction are one calculation with
different names for the ratio that matters.

## Stages for a recovery target
"""),
    code(r"""for W in (1.5, 2.0, 3.0):
    for rec in (0.95, 0.99, 0.999):
        print(f"W = {W}, recovery {rec}: {ex.leaching_stages(rec, W):.1f} stages")"""),
    md(r"""
Each extra "nine" of recovery costs about the same number of stages, and more solvent per stage buys fewer
stages: the same trade as in extraction, with the solvent's recovery (evaporation of the extract) as the
cost on the other side. Sugar-beet diffusers run 12-16 stages; oilseed extractors 6-10.

## Cross-current washing: a filter cake
"""),
    code(r"""retained = 0.5                                                      # kg solution per kg dry solid in the cake
for n_washes, wash_each in ((1, 2.0), (2, 1.0), (4, 0.5), (8, 0.25)):
    frac_left = 1.0
    for _ in range(n_washes):
        frac_left *= retained / (retained + wash_each)                 # each wash dilutes the retained solution and displaces it
    print(f"{n_washes} washes of {wash_each} kg/kg (2 kg/kg total): {1 - frac_left:.3f} of the solute removed")
print(f"countercurrent with 2 kg/kg in 4 stages: {ex.leaching_countercurrent(4, 2.0, 0.5, 1.0)['recovery']:.3f}")"""),
    md(r"""
Washing a cake on a filter is cross-current leaching: fresh wash each time, so splitting the water into
more washes removes more, and countercurrent contact removes more still. The ideal displacement washing of a
real filter does better than this well-mixed model - the wash pushes the mother liquor out as a front - but
the ranking is the same.

## Variable underflow: a stepped construction
"""),
    code(r"""# underflow retains solution in proportion to the solution's solute concentration (viscous solutions cling): r = 0.4 + 0.6 c
def underflow_ratio(c):
    return 0.4 + 0.6 * c
def countercurrent_variable(N, solvent, solute_feed, solid=1.0):
    # solve stage by stage with iteration on the overflow concentration profile
    c = np.full(N, 0.5 * solute_feed / (solvent + solute_feed))
    for _ in range(500):
        c_new = c.copy()
        # stage n receives underflow from n-1 (retained r(c_{n-1}) solid at c_{n-1}) and overflow from n+1 (at c_{n+1}); it leaves at c_n both ways
        for n in range(N):
            r_in = solid * underflow_ratio(c[n - 1]) if n > 0 else 0.0
            sol_in_under = solute_feed if n == 0 else r_in * c[n - 1]
            liq_in_under = (solid * underflow_ratio(c[n - 1]) if n > 0 else 0.0)
            over_in = solvent if n == N - 1 else (solvent - 0.0)
            # solute balance: sol_in_under + over_in * c_{n+1} = r(c_n) c_n + over_out c_n  with over_out = liq_in_under + over_in - r(c_n)
            c_next = c[n + 1] if n < N - 1 else 0.0
            def f(cn):
                r_out = solid * underflow_ratio(cn)
                over_out = liq_in_under + over_in - r_out + (0 if n > 0 else 0)
                return sol_in_under + over_in * c_next - r_out * cn - over_out * cn
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            c_new[n] = mid
        if np.max(np.abs(c_new - c)) < 1e-10:
            break
        c = c_new
    r_last = solid * underflow_ratio(c[-1])
    return 1 - r_last * c[-1] / solute_feed, c
for N in (2, 4, 6):
    rec, prof = countercurrent_variable(N, 3.0, 1.0)
    print(f"{N} stages, 3 kg solvent per kg solid: recovery {rec:.3f}; concentrations {np.round(prof, 3)}")"""),
    md(r"""
When the underflow retains more solution at high concentration, the first stages carry more solute in the
solid and the cascade needs a stage-by-stage balance; the library's Kremser functions do not apply, and a
short iteration on the concentration profile does. The result is a lower recovery than the constant-underflow
estimate - the direction real leaching data usually take.
"""),
]

NOTEBOOKS["10_adsorption_and_fixed_beds"] = [
    md(r"""
# 10 · Adsorption and fixed beds

A porous solid can pull a component out of a fluid onto its surface: activated carbon for organics, zeolites
for water and CO$_2$, silica for chromatography. Adsorption runs in cycles - a bed loads until the solute
breaks through, then is regenerated - so its design is about isotherms (capacity) and breakthrough curves
(how sharply the bed fills). This notebook fits isotherms to the course data, sizes a batch adsorber,
predicts breakthrough with the constant-pattern model, and scales a laboratory column to a plant bed.
"""),
    setup_cell(["from engsep import adsorption as ad", "from engsep import datasets"]),
    md("## Isotherms from batch data"),
    code(r"""d = pd.read_csv(datasets.path("isotherm_phenol_carbon.csv"), comment="#")
lf, ff = ad.fit_langmuir(d.c_mol_m3, d.q_mol_kg), ad.fit_freundlich(d.c_mol_m3, d.q_mol_kg)
c = np.linspace(0.1, 60, 200)
fig, ax = plt.subplots(); ax.plot(d.c_mol_m3, d.q_mol_kg, "o", label="data"); ax.plot(c, ad.langmuir(c, lf["q_max"], lf["K"]), label=f"Langmuir: q_max = {lf['q_max']:.2f}, K = {lf['K']:.2f}"); ax.plot(c, ad.freundlich(c, ff["k"], ff["n"]), "--", label=f"Freundlich: k = {ff['k']:.2f}, n = {ff['n']:.2f}")
ax.set(xlabel="c (mol/m3)", ylabel="q (mol/kg)"); ax.legend(); plt.show()
print(datasets.describe("isotherm_phenol_carbon.csv"))"""),
    md(r"""
Langmuir's isotherm saturates at a monolayer; Freundlich's keeps rising as a power law. Over the measured
range both fit - the data cannot tell them apart - but they extrapolate differently, and the Langmuir
capacity $q_{max}$ is what a bed's saturation loading depends on. A *favourable* isotherm (concave, like
these) is what makes fixed-bed adsorption sharp.

## A batch adsorber
"""),
    code(r"""iso = lambda cc: ad.langmuir(cc, lf["q_max"], lf["K"])
for m_carbon in (5.0, 20.0, 50.0):
    b = ad.batch_adsorber(c0=20.0, V=1.0, m=m_carbon, isotherm=iso)
    print(f"{m_carbon:4.0f} kg carbon per m3 of 20 mol/m3 phenol: equilibrium c = {b['c']:.2f} mol/m3, removal {b['removal']:.1%}, loading {b['q']:.2f} mol/kg")"""),
    md(r"""
A batch adsorber equilibrates along a straight operating line ($V(c_0 - c) = mq$) that meets the isotherm:
more carbon, lower residual, but at a falling loading - the last percent of removal wastes capacity, which
is why batch adsorption is for small volumes and the fixed bed for everything else.

## The fixed bed: the ideal wave and the real breakthrough
"""),
    code(r"""q0, c0, rho_b, eps, L, u = float(iso(10.0)), 10.0, 500.0, 0.4, 1.0, 0.01
t_star = ad.stoichiometric_time(q0, c0, rho_b, eps, L, u)
print(f"a 1 m bed at 1 cm/s: the concentration wave moves at {ad.wave_velocity(q0, c0, rho_b, eps, u)*1e6:.1f} um/s and the ideal breakthrough time is {t_star/3600:.2f} h")
b = pd.read_csv(datasets.path("breakthrough_curve.csv"), comment="#")
t = np.linspace(0, 20000, 400)
fig, ax = plt.subplots(); ax.plot(b.t_s / 3600, b.c_over_c0, "o", ms=4, label="laboratory data"); ax.plot(t / 3600, ad.bohart_adams(t, c0, 2.0, rho_b, L, u, 6e-4), label="Bohart-Adams, k = 6e-4"); ax.axvline(t_star / 3600, color="k", ls=":", label="ideal (shock) breakthrough")
ax.set(xlabel="t (h)", ylabel="c / c0"); ax.legend(); plt.show()"""),
    md(r"""
With no mass-transfer resistance the bed would fill as a sharp front (a shock wave) arriving at the
stoichiometric time. Real beds show an S-shaped breakthrough curve spread by mass transfer; for a
favourable isotherm the spread reaches a constant pattern that travels unchanged - the Bohart-Adams curve
through the data, apart from one sample that a careful eye rejects.

## Analysing the curve: used bed and LUB
"""),
    code(r"""mask = np.ones(len(b), bool); mask[12] = False                     # the contaminated sample
an = ad.breakthrough_analysis(b.t_s[mask], b.c_over_c0[mask], c_break=0.05, c_exhaust=0.95)
print(f"breakthrough at {an['t_break']/3600:.2f} h, exhaustion at {an['t_exhaust']/3600:.2f} h, stoichiometric time {an['t_stoich']/3600:.2f} h")
print(f"fraction of the bed used at breakthrough {an['used_fraction']:.2f}; length of unused bed {an['LUB_fraction'] * L:.2f} m")
for L_new in (2.0, 4.0):
    s = ad.lub_scale_up(an["LUB_fraction"] * L, L, an["t_break"], L_new)
    print(f"a {L_new:.0f} m bed of the same packing: breakthrough after {s['t_break_new']/3600:.1f} h, using {s['used_fraction_new']:.0%} of its capacity")"""),
    md(r"""
At breakthrough (5 % of the feed concentration leaving) the bed is only partly used: the mass-transfer zone
occupies the unused length, and that length is a property of the packing and flow, not of the bed's length.
That is the LUB scale-up: a longer plant bed wastes the same absolute length, so its fractional use improves.
A one-metre laboratory column sizes a four-metre plant bed with one number.

## Pressure-swing adsorption: a cycle balance
"""),
    code(r"""q_high, q_low = ad.langmuir(6.0 * 40.0, 3.0, 0.02), ad.langmuir(1.0 * 40.0, 3.0, 0.02)          # loadings at 6 bar and 1 bar (c in mol/m3 ~ 40 x P/bar for an ideal gas at 300 K)
cyc = ad.psa_cycle(q_high, q_low, rho_b=650.0, V_bed=2.0, y_feed=0.2, c_feed=0.2 * 240.0)
print(f"working capacity per cycle {cyc['working_capacity_mol']:.0f} mol; feed processed per cycle {cyc['feed_volume_per_cycle']:.1f} m3 at 6 bar")"""),
    md(r"""
Pressure-swing adsorption regenerates the bed by depressurising instead of heating: the working capacity is
the difference between the loadings at the two pressures, and two beds alternate every few minutes. Air
separation on zeolites (nitrogen adsorbs more strongly) and hydrogen purification run this way; the isotherm's
curvature between the two pressures is the whole design.
"""),
]

NOTEBOOKS["11_ion_exchange_and_chromatography"] = [
    md(r"""
# 11 · Ion exchange and chromatography

Two more fixed-bed separations: ion exchange swaps ions on a resin (water softening, deionisation, metal
recovery) and chromatography separates solutes by how strongly they adsorb as a pulse travels down a column
(the workhorse of biotechnology purification and of analysis). Both are adsorption with a twist: an
equilibrium written as a selectivity, and a column run as pulses rather than to breakthrough. This notebook
models both with the tools of notebook 10 and the plate theory of chromatography.
"""),
    setup_cell(["from engsep import adsorption as ad", "from scipy import special"]),
    md("## Ion-exchange equilibrium: selectivity"),
    code(r"""K_sel = 2.5                                                          # selectivity of the resin for Ca2+ over Na+ (equivalents, simplified binary exchange)
x = np.linspace(0, 1, 100)                                           # equivalent fraction of Ca in solution
y = K_sel * x / (1 + (K_sel - 1) * x)                                # equivalent fraction on the resin (constant-selectivity form)
fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(x, y, label=f"K = {K_sel}"); ax.plot([0, 1], [0, 1], "k:"); ax.set(xlabel="Ca fraction in solution", ylabel="Ca fraction on resin"); ax.legend(); plt.show()
print("the resin prefers calcium: a hard water with 10 % Ca equivalents loads the resin to", round(float(K_sel * 0.1 / (1 + (K_sel - 1) * 0.1)), 3))"""),
    md(r"""
Ion exchange is stoichiometric - one calcium replaces two sodiums - and its equilibrium looks exactly like
the constant-relative-volatility curve of distillation, with the selectivity as α. A resin that prefers the
ion to be removed gives a favourable isotherm and a sharp exchange front; regeneration with a concentrated
brine reverses the preference by mass action.

## Softening a water: capacity and cycle
"""),
    code(r"""capacity = 1.8                                                       # eq/L of resin
hardness = 4.0e-3                                                    # eq/L of Ca + Mg in the feed
flow = 10.0                                                          # m3/h
V_resin = 0.5                                                        # m3
t_cycle = V_resin * capacity * 1e3 / (hardness * 1e3 * flow)          # equivalents in the bed / equivalents fed per hour
print(f"a 0.5 m3 bed treats hard water (4 meq/L) at 10 m3/h for {t_cycle:.1f} h before exhaustion (ideal); with 70 % of the bed usable at breakthrough: {0.7 * t_cycle:.1f} h")
print(f"regeneration with 10 % NaCl (1.7 eq/L) at 150 % of stoichiometry needs {1.5 * V_resin * capacity / 1.7:.2f} m3 of brine per cycle")"""),
    md(r"""
The cycle time is the bed's capacity over the ionic load, reduced by the mass-transfer zone exactly as in
adsorption; the regenerant demand is the same equivalents plus an excess, which is where the salt and the
waste come from. A softener is a fixed-bed adsorber with an equivalents balance.

## Chromatography: the plate model
"""),
    code(r"""def peak(t, t_R, N_plates):
    sigma = t_R / np.sqrt(N_plates)
    return np.exp(-0.5 * ((t - t_R) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
t0 = 60.0                                                            # s, the column's dead time
t = np.linspace(0, 600, 2000)
fig, ax = plt.subplots()
for N_plates in (100, 1000, 10000):
    for k_ret, lbl in ((2.0, "A"), (2.4, "B")):
        t_R = t0 * (1 + k_ret)
        ax.plot(t, peak(t, t_R, N_plates) / peak(t, t_R, 10000).max() + (0 if N_plates == 100 else 1.2 if N_plates == 1000 else 2.4), "C0" if lbl == "A" else "C1")
ax.text(300, 0.9, "N = 100"); ax.text(300, 2.1, "N = 1000"); ax.text(300, 3.3, "N = 10000"); ax.set(xlabel="t (s)", ylabel="detector (offset)", title="two solutes with retention factors 2.0 and 2.4"); plt.show()
for N_plates in (100, 1000, 10000):
    tA, tB = t0 * 3.0, t0 * 3.4
    Rs = (tB - tA) / (2 * (tA + tB) / np.sqrt(N_plates))
    print(f"N = {N_plates:5d} plates: resolution R_s = {Rs:.2f}" + ("  (baseline separated)" if Rs >= 1.5 else ""))"""),
    md(r"""
A solute travels at the mobile-phase velocity divided by $1 + k$, where $k$ is the retention factor (how much
of it sits on the stationary phase); different $k$ separate the peaks and band broadening merges them. In
the plate model each peak is a Gaussian whose width falls as $1/\sqrt{N}$: two solutes differing by 20 % in
retention need ten thousand plates for baseline resolution - the reason HPLC columns are packed with
micron particles.

## Van Deemter: why there is an optimum velocity
"""),
    code(r"""u = np.linspace(0.05, 5, 200)                                        # mm/s
A_, B_, C_ = 0.01, 0.002, 0.004                                      # mm, mm2/s, s (illustrative)
H = A_ + B_ / u + C_ * u
u_opt = np.sqrt(B_ / C_)
plt.plot(u, H); plt.axvline(u_opt, color="C3", ls="--", label=f"optimum u = {u_opt:.2f} mm/s, H = {A_ + 2*np.sqrt(B_*C_):.4f} mm"); plt.xlabel("velocity (mm/s)"); plt.ylabel("plate height H (mm)"); plt.legend(); plt.show()"""),
    md(r"""
The plate height has three contributions: eddy diffusion (independent of velocity), longitudinal diffusion
(worse when the pulse lingers) and mass-transfer lag (worse when it rushes); their sum has a minimum at
$u = \sqrt{B/C}$. Chromatographers run near it, or faster with smaller particles that shrink $C$ -
the same competition between diffusion and residence time as in every column in this course.
"""),
]

NOTEBOOKS["12_membrane_gas_separation"] = [
    md(r"""
# 12 · Membrane gas separation

A polymer film lets some gases through faster than others: hydrogen recovery, nitrogen from air, CO$_2$
from natural gas and biogas all run on the solution-diffusion mechanism with no phase change and no heat.
The design questions are the membrane's selectivity and permeance, the pressure ratio, the stage cut, and
how the flow pattern on the two sides changes the answer. This notebook computes permeate and retentate
compositions for the complete-mixing and cross-flow models, sizes the membrane area, and reads Robeson's
trade-off from the permeability table.
"""),
    setup_cell(["from engsep import datasets", "from engsep import membranes as mb"]),
    md("## Permeabilities and the selectivity trade-off"),
    code(r"""d = pd.read_csv(datasets.path("gas_permeabilities.csv"), comment="#")
d["O2/N2"] = d.O2 / d.N2
d["CO2/CH4"] = d.CO2 / d.CH4
print(d.round(2).to_string(index=False))
fig, ax = plt.subplots(); ax.loglog(d.O2, d["O2/N2"], "o")
for _, r in d.iterrows():
    ax.annotate(r.polymer.split(" ")[0], (r.O2, r["O2/N2"]), fontsize=8)
ax.set(xlabel="O2 permeability (Barrer)", ylabel="O2/N2 selectivity", title="permeable or selective: Robeson's trade-off"); plt.show()"""),
    md(r"""
The most permeable polymer (silicone rubber) is the least selective; the glassy polymers separate well but
slowly. Robeson's upper bound is the frontier of that trade-off, and every commercial membrane is a
compromise on it - thin skins (a fraction of a micron on a porous support) make the glassy polymers fast
enough to use.

## One membrane, complete mixing
"""),
    code(r"""alpha = 5.6                                                          # O2/N2 selectivity of polysulfone
x_f = 0.21
for r in (0.5, 0.2, 0.1, 0.05):
    row = [mb.complete_mixing(alpha, x_f, r, theta)["y_permeate"] for theta in (0.05, 0.2, 0.5)]
    print(f"pressure ratio p_perm/p_feed = {r:4.2f}: permeate O2 at stage cut 5 / 20 / 50 %: {row[0]:.3f} / {row[1]:.3f} / {row[2]:.3f}")"""),
    md(r"""
Two levers: a lower pressure ratio (higher feed pressure or vacuum on the permeate) raises the enrichment
towards the selectivity limit, and a lower stage cut keeps the permeate rich (the first permeate is the
purest; taking more dilutes it with what comes later). Air enriched to 35-45 % oxygen with a single membrane
is the usual outcome, which is exactly what medical oxygen concentrators and enriched-air combustion use.

## Cross flow versus complete mixing
"""),
    code(r"""thetas = np.linspace(0.02, 0.8, 30)
fig, ax = plt.subplots()
for r in (0.2, 0.05):
    ax.plot(thetas, [mb.complete_mixing(alpha, x_f, r, t)["y_permeate"] for t in thetas], "--", label=f"complete mixing, r = {r}")
    ax.plot(thetas, [mb.cross_flow(alpha, x_f, r, t)["y_permeate"] for t in thetas], label=f"cross flow, r = {r}")
ax.set(xlabel="stage cut theta", ylabel="permeate O2 mole fraction"); ax.legend(); plt.show()
cm, cf = mb.complete_mixing(alpha, x_f, 0.1, 0.3), mb.cross_flow(alpha, x_f, 0.1, 0.3)
print(f"at r = 0.1 and theta = 0.3: complete mixing y = {cm['y_permeate']:.3f} (retentate {cm['x_retentate']:.3f}); cross flow y = {cf['y_permeate']:.3f} (retentate {cf['x_retentate']:.3f})")"""),
    md(r"""
In cross flow the feed depletes along the membrane and the local permeate is collected as it forms, so the
early, rich permeate is not mixed back with the feed side's average: the permeate is richer and the retentate
leaner than with complete mixing. Countercurrent modules do better still; complete mixing is the conservative
bound a designer sizes against.

## Membrane area and a nitrogen generator
"""),
    code(r"""feed = 1.0                                                           # mol/s of air
Q_O2 = 1.4 * 3.35e-16 / 0.5e-6                                        # polysulfone, 0.5 um skin: permeance in mol/(m2 s Pa)
p_feed, p_perm = 8e5, 1e5
for theta in (0.3, 0.5, 0.7):
    res = mb.cross_flow(alpha, x_f, p_perm / p_feed, theta)
    x_avg = 0.5 * (x_f + res["x_retentate"])
    A = mb.membrane_area(feed, theta, res["y_permeate"], Q_O2, p_feed, x_avg, p_perm)
    print(f"stage cut {theta}: retentate {1 - res['x_retentate']:.1%} N2 ({(1 - theta):.0%} of the feed), permeate {res['y_permeate']:.1%} O2; area {A:.0f} m2")"""),
    md(r"""
A nitrogen generator keeps the retentate: at a high stage cut the oxygen has mostly permeated and what is
left is 95-99 % nitrogen, at the cost of most of the feed and a large area. Membrane area is permeate flow over
permeance times driving force - and with a permeance from a half-micron skin, tens of square metres per
mol/s, which hollow-fibre modules pack into a cylinder the size of a scuba tank.

## Choosing the membrane for CO2 removal from biogas
"""),
    code(r"""for _, r in d.iterrows():
    res = mb.complete_mixing(r["CO2/CH4"], 0.40, 0.1, 0.4)
    print(f"{r.polymer:<24} CO2/CH4 selectivity {r['CO2/CH4']:5.1f}: permeate {res['y_permeate']:.2f} CO2, retentate {1 - res['x_retentate']:.2f} CH4 (methane lost with the permeate: {0.4 * (1 - res['y_permeate']) / 0.6:.0%})")"""),
    md(r"""
For upgrading biogas the retentate is the product (methane) and the permeate the waste, so the methane lost
with the CO$_2$ is the cost: only the selective glassy polymers keep it below ten percent, and real plants
use two stages with permeate recycle to get below two. The selectivity table and a flow model are enough to
rank the candidates before any module is bought.
"""),
]

NOTEBOOKS["13_reverse_osmosis_and_ultrafiltration"] = [
    md(r"""
# 13 · Reverse osmosis and ultrafiltration

Pressure-driven membranes separate liquids by size and by solution-diffusion: reverse osmosis desalts
seawater against an osmotic pressure of tens of bar, nanofiltration softens, ultrafiltration retains proteins
and colloids, microfiltration clarifies. This notebook computes osmotic pressures, water and salt fluxes,
rejection and recovery for an RO plant, the gel-polarisation limit of ultrafiltration, and a countercurrent
dialyser.
"""),
    setup_cell(["from engsep import membranes as mb"]),
    md("## Osmotic pressure: the price of desalination"),
    code(r"""T = 298.15
for name, c_salt in (("brackish water (2 g/L NaCl)", 2 / 58.44), ("seawater (35 g/L)", 35 / 58.44), ("brine at 50 % recovery (70 g/L)", 70 / 58.44)):
    pi = mb.osmotic_pressure(2 * c_salt, T)
    print(f"{name:<32}: {c_salt:.3f} mol/L -> osmotic pressure {pi/1e5:.1f} bar (ideal van 't Hoff)")
print("the minimum work of desalination is the osmotic pressure times the volume: about 0.8 kWh/m3 for seawater at 50 % recovery; real plants use 3-4")"""),
    md(r"""
Van 't Hoff's law - osmotic pressure like an ideal-gas pressure of the dissolved ions - puts seawater near
30 bar and its concentrate at 50 % recovery near 60; the applied pressure must exceed the *concentrate's*
osmotic pressure at the membrane, which is why seawater RO runs at 55-70 bar and why recovery is capped.

## Water and salt fluxes, rejection and recovery
"""),
    code(r"""A_w, B_s = 3e-12, 2e-8                                              # water permeability m/(s Pa), salt permeability m/s (typical seawater membrane)
c_feed = 35 / 58.44
rows = []
for dP_bar in (40, 50, 60, 70, 80):
    ro = mb.ro_permeate_concentration(A_w, B_s, dP_bar * 1e5, c_feed, T)
    rows.append((dP_bar, ro["J_w"] * 3600 * 1000, ro["c_permeate"] * 58.44 * 1000, ro["rejection"], mb.ro_recovery(c_feed, dP_bar * 1e5, T)))
print(pd.DataFrame(rows, columns=["dP (bar)", "water flux (L/(m2 h))", "permeate (mg/L NaCl)", "rejection", "max recovery"]).round(3).to_string(index=False))"""),
    md(r"""
Water flux grows with the net driving pressure ($\Delta P - \Delta\pi$) while salt flux depends only on the
concentration difference, so raising the pressure dilutes the salt in more water: rejection improves with
pressure. The permeate meets the 500 mg/L drinking-water limit above 50 bar; the maximum recovery - where the
concentrate's osmotic pressure catches the applied pressure - sets the plant's water yield.

## Concentration polarisation
"""),
    code(r"""J_w = mb.ro_permeate_concentration(A_w, B_s, 60e5, c_feed, T)["J_w"]
for k in (2e-5, 5e-5, 1e-4):
    cp = mb.concentration_polarisation(J_w, k)
    pi_wall = mb.osmotic_pressure(2 * c_feed * cp, T)
    print(f"mass-transfer coefficient {k:.0e} m/s: wall concentration {cp:.2f} x bulk -> osmotic pressure at the wall {pi_wall/1e5:.1f} bar instead of {mb.osmotic_pressure(2*c_feed, T)/1e5:.1f}")"""),
    md(r"""
Rejected salt piles up at the membrane surface until back-diffusion balances the convective flux: the wall
concentration exceeds the bulk by the factor $e^{J/k}$, raising the osmotic pressure the membrane sees and
the salt passage. Cross-flow velocity and spacers keep $k$ high; polarisation is why RO modules are
thin channels with turbulence promoters, and it is worse in ultrafiltration.

## Ultrafiltration: the gel layer
"""),
    code(r"""c_gel, k_uf = 300.0, 1.5e-6                                          # g/L (protein gel), m/s
cs = np.array([1, 3, 10, 30, 100.0])
J = mb.uf_limiting_flux(k_uf, c_gel, cs)
plt.semilogx(cs, J * 3600 * 1000, "o-"); plt.xlabel("bulk protein concentration (g/L)"); plt.ylabel("limiting flux (L/(m2 h))"); plt.title("gel polarisation: flux independent of pressure"); plt.show()
print("flux at 10 g/L:", round(float(J[2] * 3600 * 1000), 1), "L/(m2 h); it falls to zero as the bulk approaches the gel concentration")"""),
    md(r"""
Ultrafiltration of macromolecules polarises so strongly that a gel forms at the wall; beyond that point
more pressure only thickens the gel and the flux is set by mass transfer alone, $J = k\ln(c_{gel}/c_b)$. The
flux falls logarithmically as the retentate concentrates - which is why protein concentration by UF slows
towards the end and why diafiltration (adding water) is used to wash out small solutes.

## Dialysis
"""),
    code(r"""for area in (1.0, 2.0, 4.0):
    dz = mb.dialysis_countercurrent(P_m=1e-5, area=area, Q1=5e-6, Q2=1e-5, c1_in=1.0)
    print(f"membrane area {area} m2: NTU = {dz['NTU']:.1f}, removal of the small solute {dz['removal']:.1%}")"""),
    md(r"""
Dialysis needs no pressure: solutes diffuse down their concentration gradients across a membrane that
retains the large ones. A countercurrent dialyser is a heat exchanger in disguise - NTU, effectiveness and a
flow-ratio - and the haemodialyser that keeps a kidney patient alive is designed with exactly these two
lines of arithmetic.
"""),
]

NOTEBOOKS["14_crystallization"] = [
    md(r"""
# 14 · Crystallization

Most solid products - salts, sugars, pharmaceuticals - are made by crystallisation, which purifies and
shapes at the same time. The yield comes from solubility; the crystal size distribution, which decides
filtration, drying, flow and dissolution, comes from the competition between nucleation and growth. This
notebook computes yields from a solubility table, follows supersaturation in the metastable zone, and applies
the MSMPR population balance to extract kinetics from a measured size distribution and to design for a
target crystal size.
"""),
    setup_cell(["from engsep import crystallization as cr", "from engsep import datasets"]),
    md("## Solubility and yield"),
    code(r"""T = np.linspace(0, 100, 101)
fig, ax = plt.subplots()
for salt in ("KNO3", "NaCl", "KCl", "CuSO4", "Na2SO4"):
    ax.plot(T, cr.solubility(salt, T), label=salt)
ax.set(xlabel="T (degC)", ylabel="solubility (g / 100 g water)"); ax.legend(); plt.show()
for salt in ("KNO3", "NaCl"):
    cy = cr.cooling_yield(salt, 1000.0, 80.0, 20.0)
    print(f"cooling 1000 kg of saturated {salt} solution from 80 to 20 degC: {cy['crystals_kg']:.0f} kg of crystals ({cy['yield_fraction']:.0%} of the solute)")
cy_e = cr.cooling_yield("NaCl", 1000.0, 80.0, 80.0, evaporated_water_kg=300.0)
print(f"evaporating 300 kg of water from the NaCl solution at 80 degC instead: {cy_e['crystals_kg']:.0f} kg of crystals")"""),
    md(r"""
Potassium nitrate's solubility rises fivefold between 20 and 80 degC, so cooling crystallisation recovers
80 % of it; sodium chloride's barely changes, so salt is crystallised by evaporation instead. The choice
between cooling, evaporation and antisolvent crystallisation is read off the solubility curve.

## Hydrates
"""),
    code(r"""h = 5 * 18.02 / 159.6                                                # kg of water of crystallisation per kg CuSO4 in CuSO4.5H2O
cy = cr.cooling_yield("CuSO4", 1000.0, 80.0, 20.0, hydrate_water_ratio=h)
print(f"CuSO4.5H2O from 1000 kg of solution saturated at 80 degC, cooled to 20: {cy['crystals_kg']:.0f} kg of pentahydrate ({cy['anhydrous_kg']:.0f} kg CuSO4), yield {cy['yield_fraction']:.0%}")"""),
    md(r"""
When the crystal takes water with it, the mother liquor loses solvent as well as solute and the balance must
account for both; the yield of the hydrate is higher than the anhydrous solubility difference suggests,
and the crystals weigh more. Blue vitriol is a quarter water by mass.

## Supersaturation and the metastable zone
"""),
    code(r"""c_star = float(cr.solubility("KNO3", 40.0))
for c in (c_star * 1.02, c_star * 1.05, c_star * 1.15):
    s = cr.supersaturation(c, c_star)
    print(f"c = {c:.1f} g/100 g at 40 degC: relative supersaturation {s['relative']:.2f} -> " + ("growth only (metastable)" if s["relative"] < 0.08 else "spontaneous nucleation likely"))"""),
    md(r"""
Between saturation and the metastable limit crystals grow but do not spontaneously nucleate; beyond it a
shower of nuclei appears and the product is fine dust. Industrial crystallisers keep supersaturation inside
the zone - a few percent - by controlling cooling or evaporation rate, and seed to avoid waiting for
nucleation.

## The MSMPR crystalliser and its size distribution
"""),
    code(r"""d = pd.read_csv(datasets.path("csd_msmpr.csv"), comment="#")
fit = cr.msmpr_from_csd(d.L_m, d.n_per_m3_m, tau=2400.0)
print(f"from the population density plot: G = {fit['G']*1e9:.2f} nm/s, n0 = {fit['n0']:.2e} 1/(m3 m), nucleation rate B0 = {fit['B0']:.2e} 1/(m3 s)   (generated with G = 8 nm/s, n0 = 2e17)")
L = np.linspace(0, 1e-3, 300)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(d.L_m * 1e6, d.n_per_m3_m, "o", label="sieve data"); ax[0].semilogy(L * 1e6, cr.msmpr_population(L, fit["n0"], fit["G"], 2400.0), label="n0 exp(-L/G tau)"); ax[0].set(xlabel="L (um)", ylabel="population density (1/(m3 m))"); ax[0].legend()
ax[1].plot(L * 1e6, cr.msmpr_mass_distribution(L, fit["G"], 2400.0) * 1e-6, label="mass density"); ax[1].axvline(cr.msmpr_dominant_size(fit["G"], 2400.0) * 1e6, color="k", ls=":", label=f"dominant size {cr.msmpr_dominant_size(fit['G'], 2400.0)*1e6:.0f} um"); ax[1].set(xlabel="L (um)", ylabel="mass fraction per um"); ax[1].legend(); plt.show()"""),
    md(r"""
In a mixed-suspension mixed-product-removal crystalliser the population density is exponential in size,
$n = n^0 e^{-L/G\tau}$: the slope of the semi-log plot gives the growth rate, the intercept the nuclei
density, and their product the nucleation rate. Two numbers from one sieve analysis - and the mass
distribution peaks at $3G\tau$, the dominant size that customers specify.

## Designing for a crystal size
"""),
    code(r"""M_T = 200.0                                                          # kg/m3 magma density
for L_D in (200e-6, 400e-6, 800e-6):
    des = cr.msmpr_design(M_T, L_D, fit["G"])
    print(f"dominant size {L_D*1e6:.0f} um needs tau = {des['tau']/60:.0f} min at G = {fit['G']*1e9:.1f} nm/s, with a nucleation rate of {des['B0']:.2e} 1/(m3 s)")
print("\ndoubling the crystal size needs double the residence time and cuts the allowable nucleation rate sixteenfold: nucleation control (seeding, gentle stirring) is the whole game")"""),
    md(r"""
At fixed growth rate the dominant size is proportional to residence time, but the magma density constrains
the nuclei: $n^0 \propto M_T/(G\tau)^4$. Bigger crystals need a longer, gentler crystalliser with far fewer
nuclei - which is why fines destruction and classified product removal exist, and why the population balance
is the designer's tool rather than the yield alone.
"""),
]

NOTEBOOKS["15_drying_and_evaporation"] = [
    md(r"""
# 15 · Drying and evaporation

Removing water is the most common and most energy-hungry separation: evaporators concentrate solutions
before crystallisation, dryers finish solids. Both are governed by heat transfer and by the properties of
moist air and steam. This notebook works through psychrometry, drying-rate curves and drying times, then
single- and multiple-effect evaporators with their steam economy - and shows why reusing vapour matters.
"""),
    setup_cell(["from engsep import datasets", "from engsep import drying_evaporation as de"]),
    md("## Psychrometry: the air that does the drying"),
    code(r"""for T_C, RH in ((20.0, 0.6), (40.0, 0.2), (60.0, 0.05), (80.0, 0.02)):
    H = RH * de.humidity_saturation(T_C)
    print(f"air at {T_C:.0f} degC, {RH:.0%} RH: H = {H:.4f} kg/kg, dew point {de.dew_point(H):.1f} degC, wet-bulb {de.wet_bulb_temperature(T_C, H):.1f} degC, enthalpy {de.enthalpy_moist_air(T_C, H)/1e3:.0f} kJ/kg dry air")"""),
    md(r"""
Hot dry air has a low wet-bulb temperature: the surface of a wet solid sits at the wet-bulb temperature while
water evaporates freely, whatever the air temperature, so heating the air raises the driving force
($T - T_{wb}$) without cooking the product. That is the whole thermodynamic reason for hot-air drying.

## The drying-rate curve
"""),
    code(r"""d = pd.read_csv(datasets.path("drying_curve.csv"), comment="#")
an = de.drying_curve_analysis(d.t_s, d.X_dry_basis)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(d.t_s / 60, d.X_dry_basis, "o-", ms=3); ax[0].axvline(an["t_critical"] / 60, color="k", ls=":"); ax[0].set(xlabel="t (min)", ylabel="X (kg/kg dry)")
ax[1].plot(d.X_dry_basis, an["rate"] * 3600, "o-", ms=3); ax[1].axvline(an["X_critical"], color="k", ls=":", label=f"X_c = {an['X_critical']:.2f}"); ax[1].set(xlabel="X (kg/kg dry)", ylabel="drying rate (kg/(kg h))"); ax[1].invert_xaxis(); ax[1].legend(); plt.show()
print(f"constant rate {an['R_c']*3600:.3f} kg/(kg h) until the critical moisture X_c = {an['X_critical']:.2f}; then the rate falls")"""),
    md(r"""
While the surface is wet the rate is constant and set by the air (heat transfer to a wet-bulb surface);
once the surface dries out - at the critical moisture content - water must diffuse from inside and the rate
falls, roughly in proportion to the free moisture. The two regimes need different scale-up: the first
depends on air velocity and temperature, the second on the solid's thickness and structure.

## Drying time
"""),
    code(r"""X0, X_final, X_c, X_eq = 0.9, 0.10, an["X_critical"], 0.06
R_c = an["R_c"]
dt = de.drying_time(X0, X_final, X_c, X_eq, R_c, A_over_ms=1.0)
print(f"from X = {X0} to {X_final}: constant-rate period {dt['t_constant']/60:.0f} min, falling-rate period {dt['t_falling']/60:.0f} min, total {dt['t_total']/60:.0f} min")
for X_f in (0.2, 0.1, 0.07):
    print(f"drying to X = {X_f}: {de.drying_time(X0, X_f, X_c, X_eq, R_c, 1.0)['t_total']/60:.0f} min")"""),
    md(r"""
The falling-rate period takes longer than the constant one even though it removes less water, and the last
few percent towards the equilibrium moisture take forever: the log-mean dependence. Specifying a product
drier than it needs to be is the commonest waste in a drying operation.

## Single-effect evaporation
"""),
    code(r"""se = de.single_effect(feed_kg_s=2.0, x_feed=0.10, x_product=0.40, T_feed_C=25.0, T_boil_C=100.0, T_steam_C=120.0, U=2000.0, bpr=3.0)
print(f"2 kg/s of 10 % solution to 40 %: evaporates {se['vapour_kg_s']:.2f} kg/s, needs {se['steam_kg_s']:.2f} kg/s of steam (economy {se['economy']:.2f}), duty {se['Q_W']/1e6:.2f} MW, area {se['area_m2']:.0f} m2 at dT = {se['dT_driving']:.0f} K")"""),
    md(r"""
A single effect evaporates less than a kilogram of water per kilogram of steam - the latent heat is simply
passed through, and heating a cold feed costs extra. The boiling-point rise of the concentrated solution eats
into the temperature difference, and the area follows from $Q = UA\,\Delta T$.

## Multiple effects: reusing the vapour
"""),
    code(r"""rows = []
for n in (1, 2, 3, 4):
    me = de.multiple_effect(n, 2.0, 0.10, 0.40, 95.0, 120.0, 55.0, [2500.0 - 300 * i for i in range(n)], bpr_list=[1.0 + 0.7 * i for i in range(n)])
    rows.append((n, me["steam_kg_s"], me["economy"], me["areas_m2"].sum(), np.round(me["T_boil_C"], 1)))
print(pd.DataFrame(rows, columns=["effects", "steam (kg/s)", "economy", "total area (m2)", "boiling temperatures (degC)"]).round(2).to_string(index=False))"""),
    md(r"""
The vapour from one effect boils the next at a lower pressure, so one kilogram of steam evaporates nearly
$n$ kilograms in $n$ effects: the economy climbs and the steam bill falls. The price is area - the same total
temperature drop is split among the effects, so each works with a smaller $\Delta T$ and the total area grows
roughly in proportion to $n$. Sugar mills run five or six effects; the optimum number is where the saved
steam stops paying for the extra steel.

## The boiling-point rise
"""),
    code(r"""for bpr in (0.0, 5.0, 15.0):
    me = de.multiple_effect(3, 2.0, 0.10, 0.40, 95.0, 120.0, 55.0, [2500.0, 2200.0, 1900.0], bpr_list=[bpr / 3, bpr / 2, bpr])
    print(f"boiling-point rise up to {bpr:.0f} K: economy {me['economy']:.2f}, total area {me['areas_m2'].sum():.0f} m2")"""),
    md(r"""
A concentrated solution boils above pure water, and that rise is lost driving force in every effect: with a
large rise (caustic soda, some sugar liquors) the effects run out of temperature difference and the area
balloons. Boiling-point rise is the reason some solutions are concentrated in two or three effects when
economics alone would ask for six.
"""),
]

NOTEBOOKS["16_humidification_and_cooling_towers"] = [
    md(r"""
# 16 · Humidification and cooling towers

Every large plant rejects heat to the atmosphere through a cooling tower: warm water trickles down packing
against rising air, and a small fraction evaporates, cooling the rest. The process is simultaneous heat and
mass transfer, and Merkel's method reduces it to one integral in enthalpy - the same transfer-unit form as
absorption. This notebook computes the enthalpy driving force, the number of transfer units of a tower, the
approach to wet-bulb, and the water lost to evaporation.
"""),
    setup_cell(["from engsep import drying_evaporation as de", "from scipy import integrate"]),
    md("## Enthalpy of saturated air: the equilibrium curve"),
    code(r"""T = np.linspace(10, 50, 200)
H_sat = de.enthalpy_moist_air(T, de.humidity_saturation(T)) / 1e3
plt.plot(T, H_sat); plt.xlabel("water temperature (degC)"); plt.ylabel("enthalpy of saturated air (kJ/kg dry air)"); plt.title("the equilibrium curve of a cooling tower"); plt.show()
T_air, RH = 30.0, 0.4
H_in = de.enthalpy_moist_air(T_air, RH * de.humidity_saturation(T_air)) / 1e3
print(f"ambient air at {T_air} degC, {RH:.0%} RH: enthalpy {H_in:.1f} kJ/kg, wet-bulb {de.wet_bulb_temperature(T_air, RH * de.humidity_saturation(T_air)):.1f} degC")"""),
    md(r"""
The driving force for cooling is the difference between the enthalpy of air saturated at the water
temperature and the enthalpy of the bulk air - Merkel's insight that heat and mass transfer combine into an
enthalpy difference. The saturation curve rises steeply, so warm water has a large driving force and cold
water almost none: no tower cools below the ambient wet-bulb temperature.

## Merkel's integral: the tower's NTU
"""),
    code(r"""T_hot, T_cold = 40.0, 28.0                                           # water in and out
L_over_G = 1.2                                                       # kg water per kg dry air
def air_enthalpy(Tw):                                                # operating line: H = H_in + (L/G) cp_w (T_w - T_cold)
    return H_in + L_over_G * 4.186 * (Tw - T_cold)
def integrand(Tw):
    return 4.186 / (de.enthalpy_moist_air(Tw, de.humidity_saturation(Tw)) / 1e3 - air_enthalpy(Tw))
NTU, _ = integrate.quad(integrand, T_cold, T_hot)
Tw = np.linspace(T_cold, T_hot, 100)
plt.plot(T, H_sat, label="saturated air"); plt.plot(Tw, air_enthalpy(Tw), label=f"operating line, L/G = {L_over_G}"); plt.xlabel("water temperature (degC)"); plt.ylabel("enthalpy (kJ/kg)"); plt.legend(); plt.show()
print(f"cooling from {T_hot} to {T_cold} degC (approach {T_cold - de.wet_bulb_temperature(T_air, RH * de.humidity_saturation(T_air)):.1f} K to wet-bulb): NTU = KaV/L = {NTU:.2f}")"""),
    md(r"""
The operating line is a straight line in the enthalpy-temperature plane (an energy balance), the equilibrium
curve the saturated-air enthalpy, and the tower's demand $KaV/L = \int c_p\,dT/(H^* - H)$ is the area between
them in reciprocal form. A packing's characteristic $KaV/L$ is measured by the manufacturer as a function
of L/G; matching demand to characteristic sizes the tower.

## The approach and the cost of cold water
"""),
    code(r"""T_wb = de.wet_bulb_temperature(T_air, RH * de.humidity_saturation(T_air))
for T_cold_ in (32.0, 28.0, 25.0, 23.0):
    if T_cold_ <= T_wb + 0.5:
        print(f"cold water at {T_cold_} degC: approach below 0.5 K - not achievable")
        continue
    H_in_ = H_in
    f = lambda Tw, Tc=T_cold_: 4.186 / (de.enthalpy_moist_air(Tw, de.humidity_saturation(Tw)) / 1e3 - (H_in_ + L_over_G * 4.186 * (Tw - Tc)))
    ntu_, _ = integrate.quad(f, T_cold_, T_hot)
    print(f"cold water at {T_cold_} degC (approach {T_cold_ - T_wb:.1f} K): NTU = {ntu_:.2f}")"""),
    md(r"""
The approach - how close the cold water gets to the wet-bulb temperature - is the tower's figure of merit,
and the transfer units needed grow without bound as it shrinks: a 3 K approach costs three times the packing
of an 8 K approach. Designers accept 5-8 K, and the wet-bulb temperature of the hottest summer week, not the
average, sets the design.

## Water lost: evaporation, drift and blowdown
"""),
    code(r"""H_out = air_enthalpy(T_hot) * 1e3
# the outlet air is nearly saturated at a temperature such that its enthalpy equals H_out
from scipy import optimize
T_out = optimize.brentq(lambda t: de.enthalpy_moist_air(t, de.humidity_saturation(t)) - H_out, 10.0, 60.0)
dH = de.humidity_saturation(T_out) - RH * de.humidity_saturation(T_air)
evap = dH / L_over_G                                                  # kg water evaporated per kg of circulating water
print(f"air leaves saturated at about {T_out:.1f} degC; evaporation {evap:.1%} of the circulating water; latent heat {evap * de.latent_heat(T_out)/1e3:.0f} kJ/kg of water vs sensible cooling {4.186 * (T_hot - T_cold):.0f} kJ/kg")
cycles = 4
print(f"at {cycles} cycles of concentration the blowdown is {evap/(cycles-1):.1%} and the make-up {evap * cycles/(cycles-1):.1%} of the circulation")"""),
    md(r"""
About 80 % of the cooling is latent heat: each kilogram of water cooled by 12 K loses 2 % of itself to the
air. The dissolved solids stay behind, so a bleed (blowdown) keeps them from scaling the packing; the make-up
water is evaporation plus blowdown, and its treatment is the operating cost of an otherwise free heat sink.
"""),
]

NOTEBOOKS["17_project_column_data_to_retrofit"] = [
    md(r"""
# 17 · Project: from a month of column data to a retrofit proposal

**Problem.** A benzene-toluene column (18 sieve trays, total condenser, kettle reboiler, 1 atm) has produced a
month of daily averages: feed rate and composition, reflux ratio, product purities, temperatures. The plant
wants the distillate at 98.5 % and the bottoms at 1.5 % benzene instead of the current 95/5. The task: clean
the data, determine how many theoretical stages the column actually delivers and its tray efficiency, check
the temperatures against thermodynamics, and decide whether the target can be met by reflux alone, by adding
trays, or by replacing the trays with packing - with a quantified recommendation.
"""),
    setup_cell(["from engsep import datasets, vle", "from engsep import distillation as dist", "from engsep.vle import y_from_alpha"]),
    md("## 1. Look at the data"),
    code(r"""p = pd.read_csv(datasets.path("plant_column_benzene_toluene.csv"), comment="#")
print(p.describe().round(3).to_string())
print("\nrows with notes or impossible values:\n", p[(p.note != "ok") | (p.xD > 1) | (p.zF == 0) | (p.T_top_C > 200)].to_string(index=False))"""),
    md("## 2. Clean, with a log"),
    code(r"""log, q = [], p[p.note != "target for the retrofit"].copy()
target = p[p.note == "target for the retrofit"].iloc[0]
i = q.index[q.xD > 1]; q.loc[i, "xD"] /= 100; log.append(f"day {int(q.loc[i, 'day'].iloc[0])}: distillate purity was logged in percent")
i = q.index[q.T_top_C > 200]; q.loc[i, "T_top_C"] -= 273.15; log.append(f"day {int(q.loc[i, 'day'].iloc[0])}: top temperature was logged in K")
i = q.index[q.zF == 0]; q = q.drop(i); log.append(f"day {int(p.loc[i, 'day'].iloc[0])}: feed analyser offline, day excluded")
print("\n".join(log)); print(f"\n{len(q)} usable days")"""),
    md("## 3. Check the temperatures against thermodynamics"),
    code(r"""pair = ("benzene", "toluene")
q["T_top_pred"] = [vle.dew_T(pair, xd, ideal=True) - 273.15 for xd in q.xD]
q["T_bot_pred"] = [vle.bubble_T(pair, xb, ideal=True) - 273.15 for xb in q.xB]
dev_top, dev_bot = (q.T_top_C - q.T_top_pred).abs().mean(), (q.T_bottom_C - q.T_bot_pred).abs().mean()
print(f"top: measured {q.T_top_C.mean():.1f} +/- {q.T_top_C.std():.1f} degC vs dew point of the distillate {q.T_top_pred.mean():.1f} (mean |deviation| {dev_top:.2f} K)")
print(f"bottom: measured {q.T_bottom_C.mean():.1f} +/- {q.T_bottom_C.std():.1f} degC vs bubble point of the bottoms {q.T_bot_pred.mean():.1f} (mean |deviation| {dev_bot:.2f} K)")"""),
    md(r"""
The measured temperatures match the dew point of the distillate and the bubble point of the bottoms within
the thermometers' scatter: the products are what the analysers say, the pressure is 1 atm as recorded, and
Raoult's law with the Antoine table describes the system. Temperatures are the cheapest analyser a column
has, and this check is the first thing to do with them.

## 4. How many theoretical stages does the column deliver?
"""),
    code(r"""alpha = vle.geometric_mean_alpha(pair, ideal=True)
y_eq = lambda x: y_from_alpha(x, alpha)
def stages_for(row):
    return dist.mccabe_thiele(y_eq, row.xD, row.xB, row.zF, row.R, q=1.0)["stages"]
q["N_theo"] = q.apply(stages_for, axis=1)
print(q.N_theo.describe().round(2).to_string())
N_theo = int(round(q.N_theo.median()))
E_overall = (N_theo - 1) / 18                                        # stages minus the reboiler, over real trays
print(f"\nthe column delivers about {N_theo} theoretical stages (incl. reboiler): overall tray efficiency {E_overall:.0%}; O'Connell predicts {dist.oconnell_efficiency(alpha, 0.3):.0%}")"""),
    md(r"""
Each day's operating point (feed, reflux, both purities) is a McCabe-Thiele problem whose answer is the number
of theoretical stages the column must have to have produced it; the values cluster around 11 with a
reboiler, so 10 theoretical trays out of 18 real ones - an efficiency of about 55 %, in line with O'Connell's
correlation for this system. The column is neither damaged nor exceptional.

## 5. Can reflux alone reach the target?
"""),
    code(r"""zF_avg = q.zF.mean()
rm = dist.minimum_reflux(y_eq, target.xD, zF_avg)
N_min = dist.minimum_stages(y_eq, target.xD, target.xB)
print(f"target 98.5/1.5: minimum stages {N_min} (Fenske {dist.fenske(target.xD, target.xB, alpha):.1f}), minimum reflux {rm['R_min']:.2f}")
rows = []
for R in (2.5, 3.0, 4.0, 6.0, 10.0):
    n = dist.mccabe_thiele(y_eq, target.xD, target.xB, zF_avg, R)["stages"]
    rows.append((R, n, "yes" if n <= N_theo else "no"))
print(pd.DataFrame(rows, columns=["reflux ratio", "theoretical stages needed", f"fits in {N_theo}?"]).to_string(index=False))"""),
    md(r"""
The target needs more theoretical stages than the column has at any reasonable reflux ratio: only at R = 10,
more than four times the present reboiler duty, does the count fall to the eleven available, with the Fenske
minimum at total reflux just below. Reflux alone would meet the specification at a steam bill no plant would
pay. The column needs more stages - real ones, or better ones.

## 6. Retrofit options
"""),
    code(r"""R_design = 1.3 * rm["R_min"]
N_needed = dist.mccabe_thiele(y_eq, target.xD, target.xB, zF_avg, R_design)["stages"]
print(f"at R = 1.3 R_min = {R_design:.2f}: {N_needed} theoretical stages needed")
options = [
    ("more trays, same efficiency", int(np.ceil((N_needed - 1) / E_overall)), f"{int(np.ceil((N_needed - 1) / E_overall)) - 18} extra trays: a taller shell"),
    ("high-efficiency trays (E = 0.75)", int(np.ceil((N_needed - 1) / 0.75)), "replace tray internals in the existing shell if it fits"),
    ("structured packing (HETP 0.4 m)", None, f"{dist.packed_height(N_needed - 1, 0.4):.1f} m of packing"),
]
for name, n_trays, note in options:
    print(f"{name:<34}: " + (f"{n_trays} trays; " if n_trays else "") + note)
print(f"\nexisting shell: 18 trays at 0.6 m spacing = 10.8 m; the retrofit must fit in it or the shell grows")"""),
    md(r"""
Three ways to get the missing stages: a taller column with more of the same trays (a new shell), better
trays in the same shell (high-efficiency valve trays can reach 75 %), or structured packing, whose height
equivalent to a theoretical plate of 0.4 m puts the needed stages in less height than the trays occupy now.
Packing in the existing shell is the usual retrofit answer, and the calculation says it fits with room to
spare.

## 7. The report
"""),
    code(r"""report = f'''# Benzene-toluene column: retrofit basis

Data: 30 daily averages; {'; '.join(log)}. {len(q)} days used.
Thermodynamics: Raoult's law with the Antoine table; measured top/bottom temperatures agree with the product dew/bubble points to {dev_top:.1f} and {dev_bot:.1f} K on average.
Column performance: {N_theo} theoretical stages incl. reboiler from McCabe-Thiele on each day's operating point (median; interquartile range {q.N_theo.quantile(0.25):.0f}-{q.N_theo.quantile(0.75):.0f}); overall efficiency {E_overall:.0%} (O'Connell {dist.oconnell_efficiency(alpha, 0.3):.0%}).
Target 98.5 / 1.5 mol % benzene: N_min = {N_min}, R_min = {rm['R_min']:.2f}; not reachable by reflux with the present stages.
Recommendation: replace the trays with structured packing (HETP 0.4 m), {dist.packed_height(N_needed - 1, 0.4):.1f} m for {N_needed - 1} theoretical stages, operated at R = {R_design:.2f} ({(R_design + 1)/(2.2 + 1):.2f} x the present vapour rate and reboiler duty); fits the existing 10.8 m shell.
Checks before commitment: vendor HETP at the design vapour load, flooding at R = {R_design:.2f} (notebook 05), and a rigorous simulation with non-constant relative volatility.
'''
print(report)"""),
    md(r"""
Every number in the report traces to a file, a method and a check; the unit faults are logged, the
thermodynamics were verified against the plant's own thermometers, and the recommendation is quantitative
with its remaining uncertainties named. The tools were McCabe-Thiele, Fenske, O'Connell and an HETP - the
content of notebooks 03 to 05, applied to real-looking data.
"""),
]

# =====================================================================================================
# Teaching layer
def _x(objectives, prereq, time, inside, exercises=(), implement=""):
    return dict(objectives=objectives, prereq=prereq, time=time, inside=inside, exercises=list(exercises),
                implement=implement)






EXTRAS["00_python_for_separations"] = _x(
    ["use the Antoine table and check it against boiling points", "write material balances and the lever rule",
     "use brentq and quad for bubble points and transfer-unit integrals", "read x-y diagrams and the course data"],
    "basic Python", "45 min",
    [md(r"""
## Inside the algorithm: the Antoine equation
"""), code(r"""A, B, C, _, _ = vle.ANTOINE["benzene"]
T_ = 353.25
print("by hand:", 10 ** (A - B / (C + T_ - 273.15)) * 133.322, "Pa; library:", vle.antoine_pressure("benzene", T_))""")],
    ["Compute the boiling points of every compound in the Antoine table and rank them. Which pairs of adjacent boilers would be "
     "the hardest to separate by distillation, and why does a boiling-point difference not tell the whole story?",
     "A feed of 200 kmol/h with 0.6 benzene is split into a distillate of 0.98 and a bottoms of 0.02. Compute D, B and the "
     "recovery of each component in its product, and repeat for a distillate of 0.999 - how much more benzene is lost to the "
     "bottoms when the top is made purer at fixed D?"],
    "**Implement it yourself:** write a bubble-point function for any binary with Raoult's law using brentq and compare with "
    "`vle.bubble_T` for three compositions of benzene-toluene.")

EXTRAS["01_phase_equilibrium_and_vle"] = _x(
    ["construct T-x-y and x-y diagrams for ideal and non-ideal binaries", "use activity coefficients and the Wilson model",
     "locate azeotropes and understand their origin", "see the effect of pressure on VLE"],
    "notebook 00", "60 min",
    [md(r"""
## Inside the algorithm: modified Raoult's law
"""), code(r"""x1_, T_ = 0.3, 355.0
g1_, g2_ = vle.wilson_gamma(("ethanol", "water"), x1_, T_)
P_ = x1_ * g1_ * vle.antoine_pressure("ethanol", T_) + (1 - x1_) * g2_ * vle.antoine_pressure("water", T_)
print("by hand: bubble pressure", P_, "Pa; y1 =", x1_ * g1_ * vle.antoine_pressure("ethanol", T_) / P_)""")],
    ["Fit a one-parameter Margules model (ln gamma_1 = A x_2^2, ln gamma_2 = A x_1^2) to the ethanol-water data by least "
     "squares on y and compare its x-y curve and azeotrope with Wilson's. Where does the simpler model fail?",
     "For acetone-methanol (a minimum-boiling azeotrope), compute the azeotrope composition at 0.5, 1, 2 and 5 atm and "
     "explain how pressure-swing distillation could separate the pair."],
    "**Implement it yourself:** write the Wilson activity-coefficient equations from the Lambda parameters and reproduce "
    "`vle.wilson_gamma` for ethanol-water at x = 0.2 and 350 K.")

EXTRAS["02_flash_separations"] = _x(
    ["compute bubble and dew points of multicomponent mixtures", "solve the Rachford-Rice equation",
     "understand the recovery-purity trade-off of a single stage", "solve an adiabatic flash"],
    "notebook 01", "60 min",
    [md(r"""
## Inside the algorithm: Rachford-Rice by bisection
"""), code(r"""zz, KK = np.array(z), res["K"]
lo, hi = 0.0, 1.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if np.sum(zz * (KK - 1) / (1 + mid * (KK - 1))) > 0:
        lo = mid
    else:
        hi = mid
print("by hand: V/F =", round(mid, 6), "; library:", round(res["V_over_F"], 6))""")],
    ["A natural-gas liquid (methane... use propane 0.5, n-butane 0.3, n-pentane 0.2) is flashed at 25 degC. Find the pressure "
     "range over which it is two-phase and the vapour fraction at 5 bar. What pressure recovers 90 % of the propane in the vapour?",
     "Two flashes in series (the liquid from the first flashed again at a lower pressure) versus one flash to the final "
     "pressure: compare the purity of the combined vapour for the C4/C5/C6 feed. Why does staging help?"],
    "**Implement it yourself:** write the dew-point temperature equation (sum y_i/K_i = 1) and solve it with brentq for "
    "the C4/C5/C6 feed at 2 bar; compare with `flash.dew_point_T`.")

EXTRAS["03_binary_distillation_mccabe_thiele"] = _x(
    ["derive and draw the operating lines and the q-line", "step off stages and locate the feed stage",
     "find the minimum reflux and minimum stages", "account for Murphree efficiency and handle any equilibrium curve"],
    "notebooks 01 and 02", "75 min",
    [md(r"""
## Inside the algorithm: one stage of the staircase
"""), code(r"""from scipy import optimize
x_, y_ = xD, xD
x_ = optimize.brentq(lambda xx: y_eq(xx) - y_, 0, 1)         # horizontal step to the equilibrium curve
y_ = R / (R + 1) * x_ + xD / (R + 1)                          # vertical step to the operating line
print("by hand: after stage 1, x =", round(x_, 4), "y =", round(y_, 4), "; library:", np.round(mt["x"][1], 4), np.round(mt["y"][2], 4))""")],
    ["Design the benzene-toluene column for a 40 % vapour feed (q = 0.6) and for a feed preheated to its bubble point; compare "
     "stages, feed location and R_min. Estimate which is cheaper if a preheater costs 20 % of a reboiler.",
     "For the ethanol-water column, plot the number of stages against the distillate purity from 0.70 to 0.88 at R = 3 and "
     "R = 6. Explain the shape from the equilibrium curve, and what happens at 0.89."],
    "**Implement it yourself:** write your own McCabe-Thiele stepping function for a saturated-liquid feed and reproduce the "
    "stage count and feed stage of the benzene-toluene example.")

EXTRAS["04_shortcut_and_multicomponent_distillation"] = _x(
    ["apply Fenske, Underwood, Gilliland and Kirkbride", "optimise the reflux ratio against cost",
     "choose keys and estimate non-key distribution for multicomponent feeds", "check a shortcut design for consistency"],
    "notebook 03", "75 min",
    [md(r"""
## Inside the algorithm: the Fenske equation
"""), code(r"""print("by hand: N_min =", np.log((xD / (1 - xD)) * ((1 - xB) / xB)) / np.log(alpha) if np.ndim(xD) == 0 else np.log((xD[0] / xD[2]) * (xB[2] / xB[0])) / np.log(alphas[0]), "; library:", N_min)""")],
    ["Repeat the depropaniser design with the heavy-key recovery in the distillate at 0.1 % and 5 % instead of 1 %. How do "
     "N_min, R_min and the number of stages respond? Which specification is the expensive one?",
     "The cost model uses a fixed tray efficiency. Replace it with O'Connell's correlation at three viscosities (0.2, 0.5, "
     "1.0 mPa s) and find how the optimum reflux ratio shifts when trays are less efficient."],
    "**Implement it yourself:** write Underwood's equations for a binary (find theta between the two volatilities, then "
    "R_min) and compare with the McCabe-Thiele pinch value for benzene-toluene with q = 1.")

EXTRAS["05_batch_distillation_and_column_efficiency"] = _x(
    ["apply the Rayleigh equation and batch rectification", "convert theoretical stages to real trays with efficiencies",
     "size packed columns by HETP", "estimate the column diameter from flooding"],
    "notebook 03", "60 min",
    [md(r"""
## Inside the algorithm: the Rayleigh integral
"""), code(r"""from scipy import integrate
val, _ = integrate.quad(lambda x: 1 / (y_eq(x) - x), 0.1, 0.5)
print("by hand: W/W0 =", np.exp(-val), "; library:", dist.rayleigh(y_eq, 0.5, 0.1)["W"])""")],
    ["A 1000 L charge of 40 mol % methanol in water is distilled in a simple still until the distillate averages 70 %. Use the "
     "Wilson equilibrium curve. How much is collected, and what is left in the still?",
     "For the batch rectification example, compare constant-reflux operation with constant-distillate-composition operation "
     "(raise R to hold x_D = 0.9): which collects more distillate above 0.9 from the same charge, and which takes longer "
     "(total vapour boiled)?"],
    "**Implement it yourself:** compute the O'Connell efficiency yourself and the real tray count for the benzene-toluene "
    "column of notebook 03 at three viscosities.")

EXTRAS["06_absorption_and_stripping"] = _x(
    ["use the absorption factor and the Kremser equation", "find the minimum solvent rate and choose a design rate",
     "step stages on the operating diagram", "design a stripper and relate m to Henry's law and pressure"],
    "notebook 03", "60 min",
    [md(r"""
## Inside the algorithm: the Kremser equation
"""), code(r"""A_, N_ = 1.4, 6
print("by hand:", (A_ ** (N_ + 1) - A_) / (A_ ** (N_ + 1) - 1), "; library:", ab.kremser_fraction_absorbed(A_, N_))""")],
    ["An absorber must remove 99 % of the SO2 (1.5 mol %) from 200 kmol/h of flue gas with water at 1 atm (m = 40 at 25 degC). "
     "Find L_min, the stages at 1.5 L_min, and then repeat at 10 degC (m = 25) and 40 degC (m = 70). Why do scrubbers run cold?",
     "The rich solvent (x = 0.01) is stripped with air. Plot the stages needed against the stripping factor from 1.2 to 5 for "
     "95 and 99 % stripping. Where does adding stripping gas stop paying?"],
    "**Implement it yourself:** write a stage-by-stage absorber (equilibrium, then operating line) for a linear equilibrium "
    "and reproduce the Kremser stage count for A = 1.4 and 95 % recovery.")

EXTRAS["07_packed_towers_and_mass_transfer"] = _x(
    ["compute NTU by integration and by Colburn's equation", "size a packed tower from HTU and NTU",
     "use two-film theory to identify the controlling resistance", "estimate diffusivities and coefficients from correlations"],
    "notebook 06", "60 min",
    [md(r"""
## Inside the algorithm: the transfer-unit integral
"""), code(r"""from scipy import integrate
A_ = 1.5
val, _ = integrate.quad(lambda y: 1 / (y - m * (x_in + (y - y_out) / (A_ * m))), y_out, y_in)
print("by hand: N_OG =", val, "; Colburn:", ab.ntu_colburn(y_in, y_out, m, x_in, A_))""")],
    ["A packed tower absorbs acetone (2 %) from air into water (m = 2.0) with 95 % recovery at A = 1.6; G_M = 0.04 kmol/(m2 s), "
     "K_y a = 0.06 kmol/(m3 s). Find the height. Then halve the gas flux (a wider tower): how does the height change, and why "
     "less than proportionally (assume K_y a ~ G^0.7)?",
     "For a gas with m = 5 and k_y = 0.1, k_x = 0.5, what fraction of the resistance is in the liquid? By how much must k_x be "
     "raised (more liquid turbulence) to halve the total resistance?"],
    "**Implement it yourself:** compute Fuller's diffusivity for ammonia in air from the diffusion volumes and molar masses "
    "in `transport.FULLER_V` and `transport.MOLAR_MASS`, and compare with the library and with the measured 0.28 cm2/s.")

EXTRAS["08_liquid_liquid_extraction"] = _x(
    ["compare single-stage, cross-current and countercurrent extraction", "use the extraction factor and find minimum solvent",
     "read triangular diagrams and tie lines", "apply the Hunter-Nash construction"],
    "notebooks 03 and 06", "75 min",
    [md(r"""
## Inside the algorithm: the lever rule
"""), code(r"""M_, xM_ = ex.mixing_point(F, xF, 250.0, xS)
print("by hand: M =", M_, "at", np.round(xM_, 4), "; the mixing point lies at", round((250.0 / M_), 3), "of the way from F to S")""")],
    ["Recover 98 % of the acetic acid from the 30 % feed with an immiscible solvent of K = 3.0 (a better solvent than ether). "
     "Find S_min and the stages at 1.5 S_min, and compare with the ether case from the Hunter-Nash construction.",
     "Cross-current extraction of the 30 % acid feed with ether: mix the feed with 100 kg/h of ether, separate, and repeat "
     "with fresh ether three times. Use the tie-line data (lever rule each time) and compare the final raffinate with the "
     "countercurrent result at the same total solvent."],
    "**Implement it yourself:** write the Kremser form of countercurrent extraction from the stage balances (x_n = (x_{n-1} + "
    "E y_{n+1}/K)/(1+E)...) as an iterative stage-by-stage calculation and confirm the closed form for E = 1.5, N = 4.")

EXTRAS["09_leaching_and_washing"] = _x(
    ["design countercurrent leaching with constant underflow", "size the cascade for a recovery target",
     "compare cross-current and countercurrent washing", "handle variable underflow stage by stage"],
    "notebook 08", "45 min",
    [md(r"""
## Inside the algorithm: the washing factor
"""), code(r"""W_, N_ = 2.0, 4
print("by hand: recovery =", (W_ ** (N_ + 1) - W_) / (W_ ** (N_ + 1) - 1), "; library:", ex.leaching_countercurrent(N_, 2.0, 1.0, 1.0)["recovery"])""")],
    ["Sugar beet cossettes (17 % sugar, retaining 1.2 kg solution per kg dry pulp) are leached countercurrently with water to "
     "99 % recovery. Find the stages for water-to-pulp ratios of 1.5, 2 and 3, and the sugar concentration of the juice. "
     "Which ratio gives the juice needing least evaporation?",
     "Add a solute loss in the underflow moisture after pressing (the pulp is pressed to 0.6 kg/kg after the last stage): how "
     "does the recovery change, and how many extra stages compensate?"],
    "**Implement it yourself:** simulate a 4-stage countercurrent leaching train by iterating the stage balances until the "
    "concentrations converge (constant underflow), and confirm the closed-form recovery.")

EXTRAS["10_adsorption_and_fixed_beds"] = _x(
    ["fit Langmuir and Freundlich isotherms", "size a batch adsorber", "predict fixed-bed breakthrough with the constant-pattern model",
     "analyse breakthrough curves and scale up with the LUB method"],
    "notebook 06", "60 min",
    [md(r"""
## Inside the algorithm: the stoichiometric time
"""), code(r"""print("by hand:", L * (eps + rho_b * q0 / c0) / u, "s; library:", ad.stoichiometric_time(q0, c0, rho_b, eps, L, u))""")],
    ["The phenol bed is fed at 5 mol/m3 instead of 10. Recompute the ideal breakthrough time and the Bohart-Adams curve (same "
     "k). Why does a lower feed concentration give a *longer* time but a *broader* curve, and how does the LUB change?",
     "Design a two-bed system: one bed adsorbs while the other regenerates for 4 h. For the 10 mol/m3 feed at 1 cm/s, what bed "
     "length gives a breakthrough time of at least 4 h using the LUB from the laboratory column?"],
    "**Implement it yourself:** compute the stoichiometric time of the laboratory breakthrough curve as the area above the "
    "curve (trapezoidal rule) and compare with `adsorption.breakthrough_analysis`.")

EXTRAS["11_ion_exchange_and_chromatography"] = _x(
    ["write ion-exchange equilibria as selectivities", "size a softener cycle and its regeneration",
     "use the plate model of chromatography and compute resolution", "explain the van Deemter optimum"],
    "notebook 10", "45 min",
    [md(r"""
## Inside the algorithm: resolution from plates
"""), code(r"""N_ = 1000
tA_, tB_ = t0 * 3.0, t0 * 3.4
print("by hand: R_s =", (tB_ - tA_) / (2 * (tA_ + tB_) / np.sqrt(N_)))""")],
    ["A deioniser uses a cation resin (1.8 eq/L) and an anion resin (1.2 eq/L) in series to treat water with 3 meq/L of "
     "cations and 3 meq/L of anions at 5 m3/h. Size each bed for a 24 h cycle at 75 % utilisation, and compute the acid and "
     "base needed for regeneration at 200 % of stoichiometry.",
     "Two proteins have retention factors 1.5 and 1.8 on a 20 cm column with a plate height of 20 um. Find the resolution. "
     "How long must the column be for R_s = 1.5, and how does the run time scale?"],
    "**Implement it yourself:** simulate the two chromatographic peaks as Gaussians for N = 500 and compute the resolution "
    "from the peak widths at half height (w = 2.355 sigma) instead of the formula.")

EXTRAS["12_membrane_gas_separation"] = _x(
    ["read permeability tables and the selectivity trade-off", "compute permeate and retentate for complete mixing and cross flow",
     "size membrane area", "choose membranes for nitrogen generation and biogas upgrading"],
    "notebook 02", "60 min",
    [md(r"""
## Inside the algorithm: the complete-mixing flux ratio
"""), code(r"""r_, th_ = 0.1, 0.3
res_ = mb.complete_mixing(alpha, x_f, r_, th_)
y_, x_ = res_["y_permeate"], res_["x_retentate"]
print("flux ratio check: y/(1-y) =", round(y_ / (1 - y_), 4), "; alpha (x - r y)/((1-x) - r(1-y)) =", round(alpha * (x_ - r_ * y_) / ((1 - x_) - r_ * (1 - y_)), 4))""")],
    ["Hydrogen recovery from a refinery purge (60 % H2, 40 % CH4) with polysulfone (H2/CH4 selectivity from the table) at "
     "a pressure ratio of 0.1: plot permeate H2 purity and H2 recovery against stage cut (cross flow). Where would you operate "
     "for 95 % purity, and what recovery does that give?",
     "Two-stage biogas upgrading: the permeate of the first stage (polyimide) is recompressed and fed to a second stage, "
     "whose retentate is recycled to the feed. Estimate the methane loss with and without the second stage."],
    "**Implement it yourself:** integrate the cross-flow model yourself (dL/dV = -1, d(Lx)/dV = -y with the local flux "
    "ratio) with solve_ivp and reproduce `membranes.cross_flow` for alpha = 5.6, r = 0.1, theta = 0.3.")

EXTRAS["13_reverse_osmosis_and_ultrafiltration"] = _x(
    ["compute osmotic pressures and the minimum work of desalination", "solve the coupled water and salt fluxes of RO",
     "estimate concentration polarisation and the gel-polarisation limit of UF", "design a countercurrent dialyser"],
    "notebook 12", "60 min",
    [md(r"""
## Inside the algorithm: van 't Hoff
"""), code(r"""from engsep.constants import R
print("by hand:", 2 * (35 / 58.44) * 1000 * R * 298.15 / 1e5, "bar; library:", mb.osmotic_pressure(2 * 35 / 58.44, 298.15) / 1e5)""")],
    ["A brackish-water plant (2 g/L) runs at 15 bar with 75 % recovery. Compute the concentrate's osmotic pressure and check "
     "that the pressure suffices; then find the permeate concentration at the feed end and at the concentrate end of the "
     "module and their flow-weighted average.",
     "Energy: compute the pumping energy per m3 of permeate for seawater at 60 bar and 45 % recovery with and without an "
     "energy-recovery device that returns 95 % of the concentrate's pressure energy. Compare with the thermodynamic minimum."],
    "**Implement it yourself:** iterate the RO permeate concentration yourself (start with c_p = 0, compute Delta pi, J_w, "
    "J_s, update c_p) and reproduce `membranes.ro_permeate_concentration` at 60 bar.")

EXTRAS["14_crystallization"] = _x(
    ["compute cooling and evaporative yields, including hydrates", "reason about supersaturation and the metastable zone",
     "extract growth and nucleation kinetics from an MSMPR size distribution", "design an MSMPR for a target size"],
    "notebook 00", "60 min",
    [md(r"""
## Inside the algorithm: the population-density slope
"""), code(r"""slope_, intercept_ = np.polyfit(d.L_m, np.log(d.n_per_m3_m), 1)
print(f"by hand: G = {-1 / (slope_ * 2400.0):.6g} m/s, n0 = {np.exp(intercept_):.6g}; library: {fit['G']:.6g}, {fit['n0']:.6g}")""")],
    ["Sodium sulphate has a solubility maximum near 32 degC (the decahydrate-anhydrous transition). Using the table, find the "
     "temperature range over which *cooling* a saturated solution gives crystals and the range over which *heating* does. "
     "Which salt is best crystallised by evaporation at 80 degC?",
     "The MSMPR data are re-sieved with a coarser set (drop every other size). Refit G and n0 and estimate the uncertainty "
     "of the dominant size; then compute how much the residence time must change to shift the dominant size by 20 %."],
    "**Implement it yourself:** compute the third moment of the measured population density by trapezoidal integration "
    "and compare the magma density it implies (k_v = pi/6, rho_c = 2660 kg/m3) with the closed-form 6 n0 (G tau)^4.")

EXTRAS["15_drying_and_evaporation"] = _x(
    ["use psychrometric relations", "analyse drying-rate curves and compute drying times",
     "design single- and multiple-effect evaporators", "account for the boiling-point rise"],
    "notebook 00", "75 min",
    [md(r"""
## Inside the algorithm: the falling-rate time
"""), code(r"""print(f"by hand: {(X_c - X_eq) / (R_c * 1.0) * np.log((X_c - X_eq) / (X_final - X_eq)) / 60:.6g} min; library: {dt['t_falling'] / 60:.6g}")""")],
    ["The cake is dried with air at 80 degC instead of 60 (same humidity): estimate the new constant rate from the wet-bulb "
     "driving force (R_c proportional to T - T_wb) and the new drying time to X = 0.10. What about the falling-rate period - "
     "does hotter air help there?",
     "Backward feed: in a triple effect, feed the cold solution to the last (coldest) effect and pump it towards the first. "
     "Argue, with the single-effect function applied effect by effect, why backward feed gives a better economy for a cold "
     "feed and a viscous concentrate."],
    "**Implement it yourself:** write the single-effect balance yourself (vapour from the concentration ratio, duty from "
    "sensible + latent heat, steam from the latent heat at the steam temperature) and reproduce `single_effect`.")

EXTRAS["16_humidification_and_cooling_towers"] = _x(
    ["use enthalpy as the driving force for simultaneous heat and mass transfer", "compute Merkel's integral for a cooling tower",
     "relate approach, L/G and transfer units", "estimate evaporation, blowdown and make-up"],
    "notebook 15", "45 min",
    [md(r"""
## Inside the algorithm: Merkel's integrand
"""), code(r"""Tw_ = 34.0
print("by hand at 34 degC: 1/(H* - H) =", 1 / (de.enthalpy_moist_air(Tw_, de.humidity_saturation(Tw_)) / 1e3 - air_enthalpy(Tw_)), "kg/kJ")""")],
    ["Repeat the tower calculation for a humid summer day (34 degC, 70 % RH). What cold-water temperature can the same tower "
     "(same NTU) now reach, and by how much must the water rate drop (lower L/G) to hold 28 degC?",
     "A counterflow tower has a packing characteristic KaV/L = 1.8 (L/G)^-0.6. Find the L/G at which it exactly meets the "
     "demand for cooling 40 -> 28 degC at 30 degC / 40 % RH, and the corresponding air flow per kg of water."],
    "**Implement it yourself:** evaluate Merkel's integral with Simpson's rule on ten intervals and compare with quad.")

EXTRAS["17_project_column_data_to_retrofit"] = _x(
    ["clean plant data and log every repair", "verify thermodynamics against plant temperatures",
     "back out theoretical stages and tray efficiency from operating data", "evaluate retrofit options quantitatively and write a defensible recommendation"],
    "notebooks 03, 04 and 05", "90 min",
    [md(r"""
## Inside the algorithm: stages from an operating point
"""), code(r"""row_ = q.iloc[0]
print("day", int(row_.day), ": xD", row_.xD, "xB", row_.xB, "zF", row_.zF, "R", row_.R, "->", dist.mccabe_thiele(y_eq, row_.xD, row_.xB, row_.zF, row_.R)["stages"], "theoretical stages")""")],
    ["The plant also asks whether the *feed tray* is in the right place. For the current operation, find the optimum feed "
     "stage from McCabe-Thiele and compare with the actual feed tray (tray 9 from the top). How many stages are lost if "
     "the feed is two trays too high?",
     "Hydraulics: at the retrofit reflux ratio the vapour rate rises. Using the flooding estimate of notebook 05 (vapour "
     "density 2.8 kg/m3, liquid 800, surface tension 0.021 N/m) and a 1.2 m shell, check that the packing operates below "
     "80 % of flooding. If not, what reflux ratio is the hydraulic limit, and how many stages does that require?"],
    "**Implement it yourself:** write `column_stages(df)` that returns the theoretical stage count for every day and its "
    "median, and confirm it reproduces the notebook's table.")


def apply_extras(name, cells):
    ex = EXTRAS[name]
    header = md("**What you will learn**\n" + "\n".join(f"- {o}" for o in ex["objectives"])
                + f"\n\n**Before you start:** {ex['prereq']}  ·  **Time:** about {ex['time']}")
    cells = [cells[0], header] + list(cells[1:])
    idx = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and "## Exercises" in c.source]
    if idx:
        i = idx[0]
        n_items = len(re.findall(r"^\d+\. ", cells[i].source, flags=re.MULTILINE))
        cells[i] = md(cells[i].source.rstrip() + f"\n{n_items + 1}. {ex['implement']}")
        return cells[:i] + list(ex["inside"]) + cells[i:]
    items = ex["exercises"] + [ex["implement"]]
    exercises = md("## Exercises\n" + "\n".join(f"{k}. {t}" for k, t in enumerate(items, 1)))
    return cells + list(ex["inside"]) + [exercises]


def write_all(run: bool = True, only=()):
    import nbclient

    for name in sorted(NOTEBOOKS):
        if only and not name.startswith(tuple(only)):
            continue
        cells = apply_extras(name, NOTEBOOKS[name])
        nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"},
                                                 "language_info": {"name": "python"}})
        if run:
            nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                    resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    write_all(run="--no-run" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")])
