"""Build the worked solutions to every exercise of the course, as executed notebooks.

The exercise texts are taken from ../notebooks/build_notebooks.py, so questions and solutions cannot
drift apart; the build fails if a notebook's number of solutions differs from its number of exercises.

    python solutions/build_solutions.py            # build and execute all solution notebooks
    python solutions/build_solutions.py 05 12      # only those whose names start with 05 or 12
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("course", HERE.parent / "notebooks" / "build_notebooks.py")
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
md, code = course.md, course.code
SOLUTIONS: dict[str, dict] = {}


def solutions(name, setup="", imports=()):
    """Register the solutions of one course notebook: a function returning one list of cells per exercise."""

    def deco(fn):
        SOLUTIONS[name] = {"setup": setup, "imports": list(imports), "answers": fn()}
        return fn

    return deco


def exercises(name) -> list[str]:
    cells = course.apply_extras(name, list(course.NOTEBOOKS[name]))
    text = [c.source for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source][0]
    items = re.split(r"\n(?=\d+\. )", text.split("## Exercises")[1].strip())
    return [re.sub(r"^\d+\.\s*", "", it).strip() for it in items]


def notebook_cells(name) -> list:
    title = re.sub(r"^#\s*", "", course.NOTEBOOKS[name][0].source.splitlines()[0])
    entry = SOLUTIONS[name]
    ex = exercises(name)
    if len(ex) != len(entry["answers"]):
        raise ValueError(f"{name}: {len(ex)} exercises but {len(entry['answers'])} solutions")
    cells = [
        md(f"""
# Solutions · {title}

Worked solutions to the exercises of [notebook {name[:2]}](../notebooks/{name}.ipynb). Try each exercise
yourself before reading its solution - the learning happens in the attempt. Solutions are one way to
answer each question; other correct approaches exist.
"""),
        course.setup_cell(entry["imports"]),
    ] + ([code(entry["setup"])] if entry["setup"] else [])
    for i, (text, answer) in enumerate(zip(ex, entry["answers"]), 1):
        cells.append(md(f"## Exercise {i}\n\n{text}"))
        cells.extend(answer)
    return cells


def write_all(only=()):
    import nbclient

    for name in SOLUTIONS:
        if only and not name.startswith(tuple(only)):
            continue
        nb = new_notebook(
            cells=notebook_cells(name),
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        nbclient.NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
        ).execute()
        path = HERE / f"{name[:2]}_solutions.ipynb"
        nbformat.write(nb, path)
        print("wrote", path.name)




# =====================================================================================================
@solutions("00_python_for_separations", imports=["from scipy import optimize", "from engsep import vle"])
def _():
    return [
        [code(r"""bp = pd.Series({n: vle.boiling_point(n) - 273.15 for n in vle.ANTOINE}).sort_values()
print(bp.round(1).to_string())
gaps = bp.diff().dropna()
print("\nclosest adjacent boilers:", ", ".join(f"{gaps.index[i]} / {bp.index[list(bp.index).index(gaps.index[i]) - 1]} ({g:.1f} K)" for i, g in enumerate(gaps) if g < 2.5))"""),
         md(r"""
Several pairs boil within two kelvin (the xylenes, benzene and cyclohexane), which makes distillation hard
but not impossible: what matters is the relative volatility, the ratio of vapour pressures, which can differ
from unity even at equal boiling points if the vapour-pressure curves have different slopes - and non-ideality
(benzene-cyclohexane forms an azeotrope) can help or kill it. Boiling points are the first look, not the
answer.
""")],
        [code(r"""F, zF = 200.0, 0.6
for xD in (0.98, 0.999):
    xB = 0.02
    D = F * (zF - xB) / (xD - xB); B = F - D
    print(f"xD = {xD}: D = {D:.1f}, B = {B:.1f} kmol/h; benzene recovery in D {D*xD/(F*zF):.4f}; toluene recovery in B {B*(1-xB)/(F*(1-zF)):.4f}; benzene lost to bottoms {B*xB:.2f} kmol/h")"""),
         md(r"""
Making the distillate purer at a fixed bottoms composition shrinks the distillate slightly and leaves the
same benzene in the bottoms: the loss is set by $x_B$, not by $x_D$. The two purities are independent
specifications with independent costs, and a plant pays for whichever the market wants.
""")],
        [code(r"""def bubble_T(pair, x1, P=101325.0):
    c1, c2 = pair
    return optimize.brentq(lambda T: x1 * vle.antoine_pressure(c1, T) + (1 - x1) * vle.antoine_pressure(c2, T) - P, 250, 500)
for x1 in (0.2, 0.5, 0.8):
    print(f"x = {x1}: by hand {bubble_T(('benzene', 'toluene'), x1) - 273.15:.3f} degC, library {vle.bubble_T(('benzene', 'toluene'), x1, ideal=True) - 273.15:.3f}")"""),
         md(r"""
Raoult's law, two Antoine curves and a root finder: the whole bubble-point calculation in three lines,
identical to the library's.
""")],
    ]


# =====================================================================================================
@solutions("01_phase_equilibrium_and_vle", imports=["from scipy import optimize", "from engsep import datasets, vle"])
def _():
    return [
        [code(r"""d = pd.read_csv(datasets.path("vle_ethanol_water.csv"), comment="#")
pair = ("ethanol", "water")
def y_margules(A, x1):
    out = []
    for x in x1:
        def f(T):
            g1, g2 = np.exp(A * (1 - x) ** 2), np.exp(A * x ** 2)
            return x * g1 * vle.antoine_pressure(pair[0], T) + (1 - x) * g2 * vle.antoine_pressure(pair[1], T) - 101325.0
        T = optimize.brentq(f, 300, 400)
        out.append(x * np.exp(A * (1 - x) ** 2) * vle.antoine_pressure(pair[0], T) / 101325.0)
    return np.array(out)
xs, ys = d.x_ethanol.to_numpy()[1:-1], d.y_ethanol.to_numpy()[1:-1]
A_fit = optimize.minimize_scalar(lambda A: np.sum((y_margules(A, xs) - ys) ** 2), bounds=(0, 3), method="bounded").x
err_m = np.abs(y_margules(A_fit, xs) - ys)
err_w = np.abs(np.array([vle.bubble_y(pair, x)[1] for x in xs]) - ys)
az_m = optimize.brentq(lambda x: y_margules(A_fit, [x])[0] - x, 0.5, 0.99)
print(f"one-parameter Margules: A = {A_fit:.3f}; mean |dy| = {err_m.mean():.4f} (Wilson {err_w.mean():.4f}); azeotrope at x = {az_m:.3f} (Wilson {vle.azeotrope(*pair)['x']:.3f}, data 0.894)")
print("largest Margules errors at x =", np.round(xs[np.argsort(err_m)[-3:]], 3))"""),
         md(r"""
A symmetric one-parameter model cannot fit an asymmetric system: ethanol-water has $\gamma^\infty$ of about 6
for ethanol in water but under 3 for water in ethanol, and Margules with one $A$ splits the difference - its
errors are five times Wilson's, largest in the middle of the range where the real curve bends, and its
azeotrope lands at 0.82 instead of 0.89. Two parameters are the minimum for an asymmetric binary.
""")],
        [code(r"""for P_atm in (0.5, 1.0, 2.0, 5.0):
    az = vle.azeotrope("acetone", "methanol", P=P_atm * 101325.0)
    print(f"{P_atm:.1f} atm: acetone-methanol azeotrope at x = {az['x']:.3f}, {az['T']-273.15:.1f} degC" if az else f"{P_atm:.1f} atm: no azeotrope")"""),
         md(r"""
The azeotrope moves from acetone-rich at low pressure towards methanol-rich at high pressure. Two columns at
different pressures can therefore each overshoot the other's azeotrope: the low-pressure column takes the
mixture past the high-pressure azeotrope composition, the high-pressure column past the low-pressure one, and
the recycled azeotropic streams shrink. Pressure-swing distillation works when the azeotrope moves by several
percent per doubling of pressure, as here.
""")],
        [code(r"""x1, T = 0.2, 350.0
L12, L21 = vle.wilson_lambda(("ethanol", "water"), T)
x2 = 1 - x1
term = L12 / (x1 + L12 * x2) - L21 / (x2 + L21 * x1)
g1 = np.exp(-np.log(x1 + L12 * x2) + x2 * term)
g2 = np.exp(-np.log(x2 + L21 * x1) - x1 * term)
print(f"by hand: gamma = ({g1:.4f}, {g2:.4f}); library: {tuple(np.round(vle.wilson_gamma(('ethanol', 'water'), x1, T), 4))}")"""),
         md(r"""
The Wilson equations in six lines; the library evaluates the same expressions with the same Lambda
parameters.
""")],
    ]


# =====================================================================================================
@solutions("02_flash_separations", imports=["from scipy import optimize", "from engsep import flash, vle"])
def _():
    return [
        [code(r"""names, z, T = ["propane", "n-butane", "n-pentane"], [0.5, 0.3, 0.2], 298.15
P_b, P_d = flash.bubble_point_P(names, z, T), flash.dew_point_P(names, z, T)
print(f"at 25 degC the mixture is two-phase between {P_d/1e5:.2f} bar (dew) and {P_b/1e5:.2f} bar (bubble)")
fl = flash.flash_TP(names, z, T, 5e5)
print(f"flash at 5 bar: V/F = {fl['V_over_F']:.3f}, vapour {np.round(fl['y'], 3)}")
P90 = optimize.brentq(lambda P: flash.flash_TP(names, z, T, P)["V_over_F"] * flash.flash_TP(names, z, T, P)["y"][0] / 0.5 - 0.9, P_d * 1.001, P_b * 0.999)
print(f"90 % of the propane in the vapour at {P90/1e5:.2f} bar (V/F = {flash.flash_TP(names, z, T, P90)['V_over_F']:.3f})")"""),
         md(r"""
Between the dew and bubble pressures the NGL is two-phase; recovering 90 % of the propane overhead needs
most of the feed vaporised, and the vapour is then only mildly enriched - a flash recovers or purifies, not
both, which is the case for a column.
""")],
        [code(r"""names, z = ["n-butane", "n-pentane", "n-hexane"], [0.3, 0.4, 0.3]
T, P_final = 310.0, 0.85e5
one = flash.flash_TP(names, z, T, P_final)
first = flash.flash_TP(names, z, T, 1.15e5)
second = flash.flash_TP(names, first["x"], T, P_final)
V1, V2 = first["V_over_F"], (1 - first["V_over_F"]) * second["V_over_F"]
y_comb = (V1 * first["y"] + V2 * second["y"]) / (V1 + V2)
print(f"one flash to 0.85 bar: V/F = {one['V_over_F']:.3f}, butane in vapour {one['y'][0]:.3f}")
print(f"two flashes (1.15 bar then 0.85 bar): total V/F = {V1 + V2:.3f}, butane in the combined vapour {y_comb[0]:.3f}")"""),
         md(r"""
The staged flashes produce slightly less vapour but a richer combined product: the first vapour is taken at a
higher pressure, where it is richer in butane, and only the leaner liquid is flashed further. Staging is the
first step towards a column, where the stages are many and the vapours are not combined but refluxed.
""")],
        [code(r"""T_dew = optimize.brentq(lambda T: sum(zi / vle.k_value(n, T, 2e5) for n, zi in zip(names, z)) - 1, 250, 450)
print(f"by hand: dew point {T_dew - 273.15:.3f} degC; library: {flash.dew_point_T(names, z, 2e5) - 273.15:.3f}")"""),
         md(r"""
The dew-point condition $\sum y_i/K_i = 1$ and a root finder reproduce the library.
""")],
    ]


# =====================================================================================================
@solutions("03_binary_distillation_mccabe_thiele", imports=["from scipy import optimize", "from engsep import distillation as dist", "from engsep import vle", "from engsep.vle import y_from_alpha"],
           setup=r"""alpha = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
y_eq = lambda x: y_from_alpha(x, alpha)
xD, xB, zF = 0.95, 0.05, 0.45""")
def _():
    return [
        [code(r"""for q, label in ((0.6, "40 % vaporised"), (1.0, "bubble-point liquid")):
    rm = dist.minimum_reflux(y_eq, xD, zF, q)["R_min"]
    R = 1.3 * rm
    mt = dist.mccabe_thiele(y_eq, xD, xB, zF, R, q)
    print(f"{label:<20}: R_min = {rm:.3f}, at 1.3 R_min = {R:.2f}: {mt['stages']} stages, feed stage {mt['feed_stage']}; reboiler vapour (R+1) D = {(R + 1) * 44.4:.0f} kmol/h")
print("preheating to the bubble point: R_min falls by a third and the reboiler vapour by a quarter; a preheater at 20 % of a reboiler's cost that saves 25 % of the reboiler duty pays back")"""),
         md(r"""
The vapour feed forces the rectifying section to handle the extra vapour, so it needs more reflux and more
stages; preheating the feed to its bubble point (with waste heat, often the bottoms) cuts the minimum reflux
by a third. Feed preheating is one of the cheapest energy savings in distillation.
""")],
        [code(r"""y_ew = lambda x: vle.equilibrium_y(("ethanol", "water"), x)
xDs = np.linspace(0.70, 0.88, 10)
rows = []
for R in (3.0, 6.0):
    rows.append([dist.mccabe_thiele(y_ew, xd, 0.02, 0.30, R)["stages"] for xd in xDs])
print(pd.DataFrame(np.array(rows).T, index=np.round(xDs, 3), columns=["R = 3", "R = 6"]).to_string())
plt.plot(xDs, rows[0], "o-", label="R = 3"); plt.plot(xDs, rows[1], "s-", label="R = 6"); plt.axvline(vle.azeotrope("ethanol", "water")["x"], color="k", ls=":", label="azeotrope"); plt.xlabel("distillate purity"); plt.ylabel("stages"); plt.legend(); plt.show()"""),
         md(r"""
The stage count barely moves up to 80 % ethanol and then climbs steeply as the distillate approaches the
azeotrope, where the equilibrium curve meets the diagonal and the steps shrink to nothing. Doubling the
reflux helps little: the pinch is in the equilibrium curve, not the operating line. At 0.885 no reflux ratio
and no number of stages will do.
""")],
        [code(r"""def mccabe_thiele_sat_liquid(y_eq, xD, xB, zF, R):
    x = y = xD
    stages, feed = 0, None
    x_int = zF
    y_int = R / (R + 1) * zF + xD / (R + 1)
    slope_s = (y_int - xB) / (x_int - xB)
    while x > xB:
        x = optimize.brentq(lambda xx: y_eq(xx) - y, 0, 1)
        stages += 1
        if feed is None and x <= zF:
            feed = stages
        if x <= xB:
            break
        y = R / (R + 1) * x + xD / (R + 1) if x > zF else xB + slope_s * (x - xB)
    return stages, feed
print("by hand:", mccabe_thiele_sat_liquid(y_eq, xD, xB, zF, 2.0), "; library:", (dist.mccabe_thiele(y_eq, xD, xB, zF, 2.0)["stages"], dist.mccabe_thiele(y_eq, xD, xB, zF, 2.0)["feed_stage"]))"""),
         md(r"""
For a saturated-liquid feed the operating lines meet at $x = z_F$, and the stepping loop is a dozen lines
that reproduce the library's count and feed stage.
""")],
    ]


# =====================================================================================================
@solutions("04_shortcut_and_multicomponent_distillation", imports=["from scipy import optimize", "from engsep import distillation as dist", "from engsep import vle", "from engsep.vle import y_from_alpha"],
           setup=r"""names = ["propane", "isobutane", "n-butane", "n-pentane"]
z = np.array([0.30, 0.15, 0.35, 0.20])
alphas = np.array([vle.antoine_pressure(n, 320.0) for n in names]) / vle.antoine_pressure("n-butane", 320.0)""")
def _():
    return [
        [code(r"""def design(hk_in_D):
    d_frac = np.array([0.99, 0.098, hk_in_D, 0.0])
    D_i = z * d_frac; B_i = z - D_i
    xD, xB = D_i / D_i.sum(), B_i / B_i.sum()
    N_min = dist.fenske(xD[0], xB[0], alphas[0], xD_hk=xD[2], xB_hk=xB[2])
    R_min = dist.underwood(alphas, z, xD, q=1.0)["R_min"]
    return N_min, R_min, dist.gilliland(1.3 * R_min, R_min, N_min)
for hk in (0.001, 0.01, 0.05):
    N_min, R_min, N = design(hk)
    print(f"heavy key {hk:.1%} to the distillate: N_min = {N_min:.1f}, R_min = {R_min:.3f}, N = {N:.1f} at 1.3 R_min")"""),
         md(r"""
Tightening the heavy-key leak from 1 % to 0.1 % adds about three minimum stages and four actual ones while
hardly changing the minimum reflux; loosening it to 5 % saves a similar number. Stages are the price of
purity, reflux the price of recovery - and the heavy-key specification is the expensive one here because it
sets the distillate's purity.
""")],
        [code(r"""alpha_bt = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
sc = dist.fug(alpha_bt, 0.45, 0.95, 0.05, R_over_Rmin=1.3)
for mu in (0.2, 0.5, 1.0):
    E = dist.oconnell_efficiency(alpha_bt, mu)
    RR = np.linspace(1.05, 3.0, 60)
    total = []
    for r_ in RR:
        R = r_ * sc["R_min"]; N = dist.gilliland(R, sc["R_min"], sc["N_min"])
        total.append(80e3 * N / E / 10 / 1e3 + (R + 1) * 45.0 * 30e3 / 3600 * 8000 * 0.02 / 1e3)
    print(f"viscosity {mu} mPa s -> efficiency {E:.2f}: optimum at R/R_min = {RR[int(np.argmin(total))]:.2f}")"""),
         md(r"""
Less efficient trays make stages more expensive, and the optimum shifts to higher reflux: buy steam instead
of steel. The shift is modest because the cost minimum is broad - the same robustness that lets the 1.2-1.5
rule of thumb survive.
""")],
        [code(r"""alpha_bt = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
zF, xD, q = 0.45, 0.95, 1.0
theta = optimize.brentq(lambda th: alpha_bt * zF / (alpha_bt - th) + (1 - zF) / (1 - th) - (1 - q), 1.0 + 1e-9, alpha_bt - 1e-9)
R_min_uw = alpha_bt * xD / (alpha_bt - theta) + (1 - xD) / (1 - theta) - 1
print(f"by hand: theta = {theta:.4f}, R_min = {R_min_uw:.4f}; McCabe-Thiele pinch: {dist.minimum_reflux(lambda x: y_from_alpha(x, alpha_bt), xD, zF)['R_min']:.4f}")"""),
         md(r"""
Underwood's root lies between the two volatilities and reproduces the graphical minimum reflux for a binary
with constant α exactly - the two methods are the same physics.
""")],
    ]


# =====================================================================================================
@solutions("05_batch_distillation_and_column_efficiency", imports=["from scipy import optimize", "from engsep import distillation as dist", "from engsep import vle", "from engsep.vle import y_from_alpha"])
def _():
    return [
        [code(r"""y_mw = lambda x: vle.equilibrium_y(("methanol", "water"), x)
x0 = 0.40
x_end = optimize.brentq(lambda xf: dist.rayleigh(y_mw, x0, xf)["y_avg"] - 0.70, 0.02, 0.39)
r = dist.rayleigh(y_mw, x0, x_end, W0=1000.0)
print(f"distil until the still holds x = {x_end:.3f}: {r['D']:.0f} L collected at an average of 70 % methanol; {r['W']:.0f} L left in the still")"""),
         md(r"""
With the Wilson curve for methanol-water, a simple still can collect about a third of the charge at 70 %
methanol before the average falls below spec; the remainder is a weak solution for the next batch.
""")],
        [code(r"""alpha = 2.4
y_eq = lambda x: y_from_alpha(x, alpha)
cr = dist.batch_constant_reflux(y_eq, x0=0.5, x_final=0.1, R=4.0, N=4)
below = np.nonzero(cr["x_D"] < 0.9)[0]
i_stop = below[0] if below.size else len(cr["x_D"]) - 1
D_const, x_stop_const = cr["D"][i_stop], cr["x_still"][i_stop]
vapour_const = (4.0 + 1) * D_const
# constant distillate composition: at each still composition find the R giving x_D = 0.9 with 4 stages
def R_for(xW):
    def f(R):
        x = y = 0.9
        for _ in range(4):
            x = optimize.brentq(lambda xx: y_eq(xx) - y, 0, 1)
            y = R / (R + 1) * x + 0.9 / (R + 1)
        return x - xW
    return optimize.brentq(f, 0.01, 1e4)
x_limit = optimize.brentq(lambda xW: R_for(xW) - 50.0, 0.22, 0.5)         # the run ends when R reaches a practical limit of 50
xs = np.linspace(0.5, x_limit, 41)
Rs = np.array([R_for(x) for x in xs])
lnW = np.concatenate([[0.0], np.cumsum(np.diff(xs) / (0.9 - xs[1:]))])
W = np.exp(lnW)
D_var = 1 - W
trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
vapour_var = trap(Rs + 1, D_var)
print(f"constant reflux R = 4: the distillate falls below 0.9 when the still reaches x = {x_stop_const:.3f}, having collected {D_const:.3f} of the charge and boiled {vapour_const:.2f} charge-volumes of vapour")
print(f"constant composition x_D = 0.9 until R = 50 (still at x = {x_limit:.3f}): {D_var[-1]:.3f} of the charge collected, boiling {vapour_var:.2f} charge-volumes (R from {Rs[0]:.1f} to 50)")
x_tr = 0.9
for _ in range(4):
    x_tr = optimize.brentq(lambda xx: y_eq(xx) - x_tr, 0, 1)                  # total reflux: operating line y = x
print(f"four stages at total reflux cannot take the still below x = {x_tr:.3f} with a 0.9 distillate: the exercise's target of 0.1 is unreachable")"""),
         md(r"""
Holding the distillate at specification by raising the reflux collects more on-spec product from the same
charge, but the reflux ratio climbs steeply as the still is depleted and the vapour boiled (the energy and
the time) climbs with it; with only four stages the still cannot be taken below about 0.21 at any reflux,
so the exercise's 0.10 was never reachable. Real batch columns do both: constant reflux for the bulk, rising
reflux for the tail, and a cut point set by the economics.
""")],
        [code(r"""alpha_bt = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
N_theo = dist.mccabe_thiele(lambda x: y_from_alpha(x, alpha_bt), 0.95, 0.05, 0.45, 2.0)["stages"] - 1
for mu in (0.2, 0.3, 0.5):
    E = 0.492 * (alpha_bt * mu) ** (-0.245)
    print(f"mu = {mu} mPa s: E = {E:.3f} (library {dist.oconnell_efficiency(alpha_bt, mu):.3f}) -> {int(np.ceil(N_theo / E))} real trays for {N_theo} theoretical")"""),
         md(r"""
O'Connell is one line; the real tray count is the theoretical count divided by it and rounded up, and the
uncertainty of the correlation (perhaps 15 %) is a few trays either way - which is why columns get a safety
margin of trays.
""")],
    ]


# =====================================================================================================
@solutions("06_absorption_and_stripping", imports=["from engsep import absorption as ab"])
def _():
    return [
        [code(r"""V, y_in, rec = 200.0, 0.015, 0.99
y_out = y_in * (1 - rec)
for T_C, m in ((10, 25.0), (25, 40.0), (40, 70.0)):
    L_min = ab.minimum_solvent(V, y_in, y_out, m)
    L = 1.5 * L_min
    A = ab.absorption_factor(L, V, m)
    print(f"{T_C} degC (m = {m}): L_min = {L_min:.0f} kmol/h, at 1.5 L_min = {L:.0f} kmol/h: A = {A:.2f}, stages = {ab.kremser_stages(A, rec):.1f}")"""),
         md(r"""
SO$_2$ is a poorly soluble gas in water ($m$ = 40): even 99 % removal needs a solvent flow forty times the gas
flow, and warming the water to 40 degC nearly doubles it. Solubility falls with temperature, so scrubbers run
cold - or, as most SO$_2$ scrubbers do, with an alkaline liquor that reacts with the gas and slashes the
effective $m$.
""")],
        [code(r"""Ss = np.linspace(1.2, 5, 60)
fig, ax = plt.subplots()
for phi in (0.95, 0.99):
    ax.plot(Ss, [ab.kremser_stages(S, phi) for S in Ss], label=f"{phi:.0%} stripped")
ax.set(xlabel="stripping factor S = mV/L", ylabel="stages", ylim=(0, 20)); ax.legend(); plt.show()
print("stages at S = 1.5 / 2 / 3 / 5 for 99 %:", [round(ab.kremser_stages(S, 0.99), 1) for S in (1.5, 2, 3, 5)])"""),
         md(r"""
The stage count falls steeply up to a stripping factor of about 2 and flattens beyond 3: more stripping gas
then buys little and costs compression or steam. The knee of the Kremser curve, at $S$ (or $A$) between 1.5
and 2.5, is where absorbers and strippers are designed.
""")],
        [code(r"""m, A, rec = 0.8, 1.4, 0.95
L_over_V = A * m
y_in = 0.02; y_out = y_in * (1 - rec)
x, y, n = 0.0, y_out, 0
while y < y_in - 1e-12:
    x = y / m                                                          # liquid leaving stage n, in equilibrium with the gas leaving it
    y = y_out + L_over_V * x                                           # gas entering stage n from below (operating line)
    n += 1
print(f"stage by stage: {n} stages reach y_in (the last overshoots); Kremser: {ab.kremser_stages(A, rec):.2f}")"""),
         md(r"""
Two lines per stage - equilibrium, then operating line - step from the lean end until the rich gas is
reached; the count rounds up the Kremser value.
""")],
    ]


# =====================================================================================================
@solutions("07_packed_towers_and_mass_transfer", imports=["from engsep import absorption as ab", "from engsep import transport as tr"])
def _():
    return [
        [code(r"""m, y_in, rec, A = 2.0, 0.02, 0.95, 1.6
y_out = y_in * (1 - rec)
N_OG = ab.ntu_colburn(y_in, y_out, m, 0.0, A)
for G_M, K_y_a in ((0.04, 0.06), (0.02, 0.06 * 0.5 ** 0.7)):
    H_OG = ab.htu_gas(G_M, K_y_a)
    print(f"G_M = {G_M} kmol/(m2 s): H_OG = {H_OG:.2f} m, N_OG = {N_OG:.2f}, height = {ab.packed_height(H_OG, N_OG):.2f} m")"""),
         md(r"""
Halving the gas flux halves the numerator of the HTU but also reduces $K_ya$ (which grows roughly as
$G^{0.7}$), so the height falls by only about 20 %, not 50 %. A wider tower helps less than one might hope -
and its diameter follows from flooding, not from this calculation.
""")],
        [code(r"""tf = ab.overall_from_film(k_y=0.1, k_x=0.5, m=5.0)
R_total = 1 / tf["K_y"]
print(f"resistances: gas 1/k_y = {1/0.1:.0f}, liquid m/k_x = {5.0/0.5:.0f}, total {R_total:.0f} -> fraction in the liquid {1 - tf['gas_resistance_fraction']:.0%}")
for factor in (2, 5, 10):
    R_new = 1 / 0.1 + 5.0 / (0.5 * factor)
    print(f"k_x x {factor}: total resistance {R_new:.0f} ({R_new/R_total:.0%} of the original)")
print("halving the total would need the liquid resistance removed entirely: unreachable by any increase of k_x")"""),
         md(r"""
With $m$ = 5 the liquid film holds half the resistance; halving the total would need the liquid resistance
removed entirely, which no realistic increase in turbulence achieves. Two-film arithmetic tells you the
ceiling of an improvement before the money is spent.
""")],
        [code(r"""MA, MB = tr.MOLAR_MASS["NH3"], tr.MOLAR_MASS["air"]
VA, VB = tr.FULLER_V["NH3"], tr.FULLER_V["air"]
T, P = 298.15, 101325.0
D = 1.013e-2 * T ** 1.75 * np.sqrt(1 / MA + 1 / MB) / (P * (VA ** (1 / 3) + VB ** (1 / 3)) ** 2)
print(f"by hand: {D*1e4:.3f} cm2/s; library {tr.fuller('NH3', 'air', T)*1e4:.3f}; measured 0.28")"""),
         md(r"""
Fuller's correlation from its diffusion volumes gives ammonia in air to within about 10 % of the measured
value - the accuracy the method is known for.
""")],
    ]


# =====================================================================================================
@solutions("08_liquid_liquid_extraction", imports=["from engsep import datasets", "from engsep import extraction as ex"],
           setup=r"""d = pd.read_csv(datasets.path("ternary_acetic_water_ether.csv"), comment="#")
tern = ex.Ternary(np.c_[d.xA_raffinate, d.xS_raffinate], np.c_[d.yA_extract, d.yS_extract], np.c_[d.xA_raffinate, d.yA_extract])""")
def _():
    return [
        [code(r"""K, x_in, x_out = 3.0, 0.30, 0.006
S_min = ex.minimum_solvent_immiscible(K, x_in, x_out)
N = ex.countercurrent_stages_immiscible(K, 1.5 * S_min, x_in, x_out)
print(f"K = 3: S_min/F = {S_min:.3f}, at 1.5 S_min (S/F = {1.5*S_min:.2f}): {N:.1f} stages")
hn = ex.hunter_nash(tern, 100.0, (0.30, 0.0), 250.0, (0.0, 1.0), 0.02)
print(f"ether (K about 0.3-0.7, mutual solubility): S/F = 2.5 and {hn['stages']} stages for a raffinate of 0.02")"""),
         md(r"""
A solvent with a distribution coefficient of 3 does the job with a fifth of the solvent-to-feed ratio and
about the same number of stages as ether, whose coefficient is below 1 over most of the range - and far
less solvent to recover afterwards. Solvent selection - distribution
coefficient, selectivity, immiscibility, and ease of recovery - decides an extraction before the stages are
counted.
""")],
        [code(r"""F, xF = 100.0, np.array([0.30, 0.0])
R, xR = F, xF.copy()
total_S = 0.0
for stage in range(1, 4):
    M, xM = ex.mixing_point(R, xR, 100.0, (0.0, 1.0))
    # split M along the tie line through it: find the raffinate a such that M lies on the tie line raffinate(a)-extract_in_equilibrium(a)
    from scipy import optimize
    def off_line(a):
        r, e = tern.raffinate(a), tern.extract_in_equilibrium(a)
        return (e[0] - r[0]) * (xM[1] - r[1]) - (e[1] - r[1]) * (xM[0] - r[0])
    a = optimize.brentq(off_line, 1e-4, xR[0])
    r, e = tern.raffinate(a), tern.extract_in_equilibrium(a)
    frac_E = ex.lever_split(xM, e, r)
    E, R = M * frac_E, M * (1 - frac_E)
    xR = r
    total_S += 100.0
    print(f"stage {stage}: raffinate {R:.1f} kg/h at {xR[0]:.3f} acid; extract {E:.1f} kg/h at {e[0]:.3f}")
cc = ex.hunter_nash(tern, 100.0, (0.30, 0.0), total_S, (0.0, 1.0), 0.02)
print(f"\ncountercurrent with the same total solvent ({total_S:.0f} kg/h): {cc['stages']} stages reach 0.02; cross-current in 3 stages reached {xR[0]:.3f}")"""),
         md(r"""
Three cross-current stages with 100 kg/h of fresh ether each take the raffinate to about 12 % acid; the same
300 kg/h used countercurrently reaches 2 % in a few stages. The lever rule on the tie line through each
mixing point is the whole cross-current calculation; the countercurrent one needs the operating point.
""")],
        [code(r"""E, N, K, x_in = 1.5, 4, 2.0, 0.10
# stage balances: R x_{n-1} + S y_{n+1} = R x_n + S y_n with y_n = K x_n; iterate the profile
x = np.full(N, x_in / 2)
for _ in range(500):
    x_new = x.copy()
    for n in range(N):
        x_prev = x_in if n == 0 else x[n - 1]
        y_next = 0.0 if n == N - 1 else K * x[n + 1]
        x_new[n] = (x_prev + E * y_next / K) / (1 + E)
    if np.max(np.abs(x_new - x)) < 1e-14:
        break
    x = x_new
print(f"stage-by-stage raffinate: x_N = {x[-1]:.6f}; Kremser closed form: {ex.countercurrent_immiscible(K, E / K, x_in, N)['x_out']:.6f}")"""),
         md(r"""
Iterating the stage balances to convergence reproduces the Kremser closed form to six digits: the closed
form is the solution of that linear system, and the iteration is what one would code for a nonlinear
distribution coefficient.
""")],
    ]


# =====================================================================================================
@solutions("09_leaching_and_washing", imports=["from engsep import extraction as ex"])
def _():
    return [
        [code(r"""sugar, retained = 0.17, 1.2                                          # kg sugar per kg cossettes (wet), kg solution per kg dry pulp
pulp = 1 - sugar                                                     # dry pulp per kg cossettes, treating the rest as insoluble... (approximation)
for ratio in (1.5, 2.0, 3.0):
    W = ratio / retained
    N = ex.leaching_stages(0.99, W)
    juice = ratio + sugar * 0.99                                       # water plus recovered sugar per kg cossettes
    conc = sugar * 0.99 / juice
    print(f"water/pulp = {ratio}: W = {W:.2f}, {N:.1f} stages for 99 % recovery; juice {conc:.1%} sugar -> {(1/conc - 1/0.65):.2f} kg water to evaporate per kg sugar for a 65 % syrup")"""),
         md(r"""
More water means fewer stages but a thinner juice, and every kilogram of extra water must be boiled off in the
evaporators - typically the dominant energy cost of a sugar factory. The lowest water ratio that keeps the
stage count practical (here 1.5-2) minimises evaporation: leaching and evaporation are designed together.
""")],
        [code(r"""for pressed in (1.2, 0.6):
    base = ex.leaching_countercurrent(8, 2.0, 1.2, 100.0)
    loss = base["solute_lost"] * pressed / 1.2                          # the underflow solution carries the loss; pressing removes part of it back into the juice
    rec = 1 - loss / 100.0
    print(f"underflow moisture {pressed} kg/kg after the last stage: recovery {rec:.4f}")
N_extra = ex.leaching_stages(1 - ex.leaching_countercurrent(8, 2.0, 1.2, 100.0)["solute_lost"] * 0.6 / 1.2 / 100.0, 2.0 / 1.2) - 8
print(f"pressing the pulp from 1.2 to 0.6 kg/kg halves the loss, worth about {abs(N_extra):.1f} stages")"""),
         md(r"""
The solute lost is what leaves in the last underflow's solution, so pressing the pulp to half its moisture
halves the loss - equivalent to about one extra stage. Presses are cheaper than stages, which is why every
diffuser has one.
""")],
        [code(r"""N, W = 4, 2.0
c = np.full(N, 0.1)
for _ in range(1000):
    c_new = c.copy()
    for n in range(N):
        c_in_under = 1.0 if n == 0 else c[n - 1]                         # solute in the underflow from the previous stage (feed carries 1 unit in 1 unit of solution)
        c_in_over = 0.0 if n == N - 1 else c[n + 1]
        c_new[n] = (c_in_under + W * c_in_over) / (1 + W)                # ideal stage: both streams leave at c_n
    if np.max(np.abs(c_new - c)) < 1e-14:
        break
    c = c_new
print(f"iterated: recovery = {1 - c[-1]:.6f}; closed form: {ex.leaching_countercurrent(N, W, 1.0, 1.0)['recovery']:.6f}")"""),
         md(r"""
With constant underflow every stage is a linear mixing rule, and iterating the four of them reproduces the
Kremser closed form.
""")],
    ]


# =====================================================================================================
@solutions("10_adsorption_and_fixed_beds", imports=["from engsep import adsorption as ad", "from engsep import datasets"],
           setup=r"""d = pd.read_csv(datasets.path("isotherm_phenol_carbon.csv"), comment="#")
lf = ad.fit_langmuir(d.c_mol_m3, d.q_mol_kg)
rho_b, eps, L, u, k = 500.0, 0.4, 1.0, 0.01, 6e-4""")
def _():
    return [
        [code(r"""t = np.linspace(0, 40000, 800)
for c0 in (10.0, 5.0):
    q0 = float(ad.langmuir(c0, lf["q_max"], lf["K"]))
    t_star = ad.stoichiometric_time(q0, c0, rho_b, eps, L, u)
    curve = ad.bohart_adams(t, c0, q0, rho_b, L, u, k)
    an = ad.breakthrough_analysis(t, curve)
    print(f"c0 = {c0}: q0 = {q0:.2f} mol/kg, ideal breakthrough {t_star/3600:.2f} h; 5 % breakthrough {an['t_break']/3600:.2f} h, exhaustion {an['t_exhaust']/3600:.2f} h, LUB {an['LUB_fraction']*L:.2f} m")"""),
         md(r"""
Halving the feed concentration more than halves the loading rate (the isotherm is favourable, so the
loading falls less than the concentration), which lengthens the ideal breakthrough time; but the
mass-transfer zone spreads in proportion to $1/(kc_0)$, so the curve is broader and the unused bed longer.
The LUB depends on the feed as well as the packing - a caution when scaling from laboratory to plant at a
different concentration.
""")],
        [code(r"""q0 = float(ad.langmuir(10.0, lf["q_max"], lf["K"]))
t = np.linspace(0, 40000, 800)
an = ad.breakthrough_analysis(t, ad.bohart_adams(t, 10.0, q0, rho_b, L, u, k))
LUB = an["LUB_fraction"] * L
from scipy import optimize
L_needed = optimize.brentq(lambda L_new: ad.lub_scale_up(LUB, L, an["t_break"], L_new)["t_break_new"] - 4 * 3600, L, 10.0)
print(f"LUB = {LUB:.3f} m; a breakthrough time of 4 h needs a bed of {L_needed:.2f} m ({ad.lub_scale_up(LUB, L, an['t_break'], L_needed)['used_fraction_new']:.0%} used at breakthrough)")"""),
         md(r"""
The LUB method scales the laboratory column to the plant bed with one number; the 4 h bed is a little over
the proportional length because the unused part is fixed. Two such beds, one adsorbing while the other is
regenerated for 4 h, make a continuous unit.
""")],
        [code(r"""b = pd.read_csv(datasets.path("breakthrough_curve.csv"), comment="#")
mask = np.ones(len(b), bool); mask[12] = False
t_, c_ = b.t_s[mask].to_numpy(), np.clip(b.c_over_c0[mask].to_numpy(), 0, 1)
area = np.trapezoid(1 - c_, t_) if hasattr(np, "trapezoid") else np.trapz(1 - c_, t_)
print(f"by hand: t* = {area/3600:.3f} h; library: {ad.breakthrough_analysis(t_, c_)['t_stoich']/3600:.3f} h")"""),
         md(r"""
The stoichiometric time is the area above the breakthrough curve - the bed's total capacity divided by the
feed rate - and the trapezoidal rule on the cleaned data reproduces the library's value.
""")],
    ]


# =====================================================================================================
@solutions("11_ion_exchange_and_chromatography", imports=["from engsep import adsorption as ad"])
def _():
    return [
        [code(r"""flow, hours, util = 5.0, 24.0, 0.75
for name, cap, load, regen, regen_eq_L in (("cation", 1.8, 3e-3, "HCl (2 M)", 2.0), ("anion", 1.2, 3e-3, "NaOH (2 M)", 2.0)):
    eq_per_cycle = load * 1e3 * flow * hours                            # equivalents fed per cycle
    V = eq_per_cycle / (cap * 1e3 * util)                               # m3 of resin
    regen_vol = 2.0 * V * cap * 1e3 / (regen_eq_L * 1e3)               # m3 of regenerant at 200 % of stoichiometry
    print(f"{name} bed: {eq_per_cycle:.0f} eq per cycle -> {V:.2f} m3 of resin; regeneration with {regen_vol:.2f} m3 of {regen}")"""),
         md(r"""
A deioniser is two softeners in series with acid and base as regenerants; the beds are sized on the
equivalents fed per cycle over the usable capacity, and the regenerant demand is twice the stoichiometric
amount. Mixed-bed polishers and reverse osmosis pretreatment exist to shrink these numbers.
""")],
        [code(r"""t0, H, L = 1.0, 20e-6, 0.20
for kA, kB in ((1.5, 1.8),):
    N = L / H
    tA, tB = t0 * (1 + kA), t0 * (1 + kB)
    Rs = (tB - tA) / (2 * (tA + tB) / np.sqrt(N))
    print(f"{L*100:.0f} cm column, N = {N:.0f}: R_s = {Rs:.2f}")
    L_needed = L * (1.5 / Rs) ** 2
    print(f"R_s = 1.5 needs only {L_needed*100:.0f} cm (resolution scales with sqrt(L)): the 20 cm column over-resolves, and a run {L_needed/L:.1f} x as long would do")"""),
         md(r"""
Resolution grows only as the square root of column length while the run time grows linearly: the 20 cm
column resolves these proteins nearly twice as well as needed, and a column a third as long would do the
job in a third of the time. Better selectivity (a larger difference in retention factors) is worth far more
than a longer column; when the selectivity is there, shorten the column.
""")],
        [code(r"""def peak(t, t_R, N):
    sigma = t_R / np.sqrt(N)
    return np.exp(-0.5 * ((t - t_R) / sigma) ** 2)
t = np.linspace(100, 300, 20001)
N, t0 = 500, 60.0
tA, tB = t0 * 3.0, t0 * 3.4
def width_half(t_R):
    p = peak(t, t_R, N)
    above = t[p >= 0.5]
    return above[-1] - above[0]
wA, wB = width_half(tA), width_half(tB)
Rs_w = 1.18 * (tB - tA) / (wA + wB)
print(f"half-height widths {wA:.2f} and {wB:.2f} s -> R_s = {Rs_w:.3f}; formula: {(tB - tA) / (2 * (tA + tB) / np.sqrt(N)):.3f}")"""),
         md(r"""
Measuring the widths at half height and using $R_s = 1.18\,\Delta t/(w_A + w_B)$ gives the same resolution as
the plate-number formula - which is how resolution is read from a real chromatogram.
""")],
    ]


# =====================================================================================================
@solutions("12_membrane_gas_separation", imports=["from scipy import integrate, optimize", "from engsep import datasets", "from engsep import membranes as mb"])
def _():
    return [
        [code(r"""d = pd.read_csv(datasets.path("gas_permeabilities.csv"), comment="#")
ps = d[d.polymer == "polysulfone"].iloc[0]
alpha = ps.H2 / ps.CH4
thetas = np.linspace(0.05, 0.7, 30)
res = [mb.cross_flow(alpha, 0.60, 0.1, t) for t in thetas]
purity = np.array([r["y_permeate"] for r in res]); recovery = thetas * purity / 0.60
plt.plot(thetas, purity, label="H2 purity of the permeate"); plt.plot(thetas, recovery, label="H2 recovery"); plt.axhline(0.95, color="k", ls=":"); plt.xlabel("stage cut"); plt.ylabel("fraction"); plt.legend(); plt.show()
i = np.nonzero(purity >= 0.95)[0]
print(f"H2/CH4 selectivity {alpha:.0f}: 95 % purity holds up to a stage cut of {thetas[i[-1]]:.2f}, recovering {recovery[i[-1]]:.0%} of the hydrogen")"""),
         md(r"""
With a selectivity of 22 the permeate stays above 95 % hydrogen only up to a stage cut of about a third,
recovering little more than half the hydrogen; pushing the cut further dilutes the permeate with methane. Purity and
recovery trade against each other along the curve, and the plant picks the point the hydrogen is worth.
""")],
        [code(r"""alpha = float(d[d.polymer.str.startswith("polyimide")].iloc[0]["CO2"] / d[d.polymer.str.startswith("polyimide")].iloc[0]["CH4"])
x_f, r = 0.40, 0.1
one = mb.cross_flow(alpha, x_f, r, 0.45)
loss_one = 0.45 * (1 - one["y_permeate"]) / (1 - x_f)
# second stage on the permeate: its retentate (methane-rich) is recycled; approximate the steady state by iterating the mixed feed
feed_x, feed_F = x_f, 1.0
for _ in range(50):
    s1 = mb.cross_flow(alpha, feed_x, r, 0.45)
    perm1 = 0.45 * feed_F
    s2 = mb.cross_flow(alpha, s1["y_permeate"], r, 0.6)
    ret2 = perm1 * (1 - 0.6)                                            # recycled to the feed
    feed_F_new = 1.0 + ret2
    feed_x_new = (1.0 * x_f + ret2 * s2["x_retentate"]) / feed_F_new
    if abs(feed_x_new - feed_x) < 1e-10:
        break
    feed_x, feed_F = feed_x_new, feed_F_new
loss_two = perm1 * 0.6 * (1 - s2["y_permeate"]) / (1 - x_f)
print(f"single stage: methane lost {loss_one:.1%}; two stages with recycle: {loss_two:.1%} (second-stage permeate {s2['y_permeate']:.3f} CO2)")"""),
         md(r"""
A second stage on the first permeate recovers most of the methane that slipped through and returns it to
the feed: the loss falls from about a tenth to well under one percent, at the cost of recompressing the
first permeate. Two-stage membrane cascades are how biogas upgrading meets pipeline methane recoveries.
""")],
        [code(r"""alpha, r, theta, x_f = 5.6, 0.1, 0.3, 0.21
def local_y(x):
    return optimize.brentq(lambda y: y / (1 - y) - alpha * (x - r * y) / ((1 - x) - r * (1 - y)), x + 1e-12, 1 - 1e-12)
sol = integrate.solve_ivp(lambda v, s: [-1.0, -(local_y(s[1]) - s[1]) / s[0]], (0, theta), [1.0, x_f], rtol=1e-9, atol=1e-12, max_step=theta / 200)
L_out, x_out = sol.y[0][-1], sol.y[1][-1]
print(f"by hand: y = {(x_f - L_out * x_out) / theta:.5f}, x = {x_out:.5f}; library: {mb.cross_flow(alpha, x_f, r, theta)['y_permeate']:.5f}, {mb.cross_flow(alpha, x_f, r, theta)['x_retentate']:.5f}")"""),
         md(r"""
The cross-flow model is two ODEs along the membrane with the local flux ratio solved at each step; the hand
integration reproduces the library to five decimals.
""")],
    ]


# =====================================================================================================
@solutions("13_reverse_osmosis_and_ultrafiltration", imports=["from engsep import membranes as mb"])
def _():
    return [
        [code(r"""T, c_feed, recovery, dP = 298.15, 2 / 58.44, 0.75, 15e5
c_conc = c_feed / (1 - recovery)
pi_conc = mb.osmotic_pressure(2 * c_conc, T)
print(f"concentrate {c_conc*58.44:.1f} g/L, osmotic pressure {pi_conc/1e5:.2f} bar against 15 bar applied: {'sufficient' if pi_conc < dP else 'insufficient'}")
A_w, B_s = 6e-12, 5e-8
ro_in = mb.ro_permeate_concentration(A_w, B_s, dP, c_feed, T)
ro_out = mb.ro_permeate_concentration(A_w, B_s, dP, c_conc, T)
w_in, w_out = ro_in["J_w"], ro_out["J_w"]
c_avg = (w_in * ro_in["c_permeate"] + w_out * ro_out["c_permeate"]) / (w_in + w_out)
print(f"permeate at the feed end {ro_in['c_permeate']*58.44*1000:.0f} mg/L, at the concentrate end {ro_out['c_permeate']*58.44*1000:.0f} mg/L; flux-weighted average about {c_avg*58.44*1000:.0f} mg/L")"""),
         md(r"""
Brackish water is easy: the concentrate at 75 % recovery has an osmotic pressure of a few bar, so 15 bar
leaves ample driving force; the permeate gets saltier along the module as the feed concentrates and the
water flux falls, and the product is the flux-weighted blend - four times the salt of the feed-end permeate.
""")],
        [code(r"""dP, recovery, rho, eta_pump = 60e5, 0.45, 1025.0, 0.85
feed_per_perm = 1 / recovery
E_pump = dP * feed_per_perm / eta_pump / 3.6e6                       # kWh per m3 of permeate, pressurising the whole feed
conc_per_perm = feed_per_perm - 1
E_recovered = 0.95 * dP * conc_per_perm / 3.6e6
pi_avg = mb.osmotic_pressure(2 * 35 / 58.44 / (1 - recovery / 2), 298.15)
E_min = pi_avg / 3.6e6
print(f"without energy recovery: {E_pump:.2f} kWh/m3; with 95 % recovery from the concentrate: {E_pump - E_recovered:.2f} kWh/m3; thermodynamic minimum (osmotic pressure of the average concentrate): {E_min:.2f} kWh/m3")"""),
         md(r"""
Pressurising 2.2 m$^3$ of seawater to make 1 m$^3$ of permeate costs about 4 kWh without energy recovery;
recovering the pressure energy of the concentrate halves it, within a factor of two of the thermodynamic
minimum. Energy-recovery devices are what made seawater desalination affordable.
""")],
        [code(r"""A_w, B_s, dP, c_feed, T = 3e-12, 2e-8, 60e5, 35 / 58.44, 298.15
c_p = 0.0
for it in range(50):
    d_pi = 2 * (c_feed - c_p) * 1000 * 8.314462618 * T
    J_w = A_w * (dP - d_pi); J_s = B_s * (c_feed - c_p)
    c_new = J_s / J_w
    if abs(c_new - c_p) < 1e-12:
        break
    c_p = c_new
print(f"by hand: c_p = {c_p*58.44*1000:.2f} mg/L after {it} iterations; library: {mb.ro_permeate_concentration(A_w, B_s, dP, c_feed, T)['c_permeate']*58.44*1000:.2f}")"""),
         md(r"""
The permeate concentration is a fixed point of the two flux equations; a few iterations converge to the
library's value.
""")],
    ]


# =====================================================================================================
@solutions("14_crystallization", imports=["from engsep import crystallization as cr", "from engsep import datasets"])
def _():
    return [
        [code(r"""T = np.linspace(0, 100, 201)
s = cr.solubility("Na2SO4", T)
i_max = int(np.argmax(s))
print(f"Na2SO4 solubility peaks at about {T[i_max]:.0f} degC in this coarse table ({s[i_max]:.1f} g/100 g; the true maximum is at 32.4 degC): below the peak, cooling crystallises; above it, heating does (retrograde solubility)")
for salt in cr.SOLUBILITY:
    print(f"{salt:<8}: solubility at 80 degC {cr.solubility(salt, 80.0):6.1f} g/100 g water -> crystals per kg water evaporated at 80 degC: {cr.solubility(salt, 80.0)/100*1000:.0f} g")"""),
         md(r"""
Sodium sulphate's solubility rises to the decahydrate-anhydrous transition (32.4 degC; the 20-degree table
puts the peak near 40) and then falls slightly: above the transition, heating crystallises it - the retrograde solubility used in some salt-recovery
processes. For evaporative crystallisation the most soluble salt at the evaporation temperature gives the
most crystals per kilogram of water removed: sucrose and potassium nitrate far ahead of sodium chloride.
""")],
        [code(r"""d = pd.read_csv(datasets.path("csd_msmpr.csv"), comment="#")
full = cr.msmpr_from_csd(d.L_m, d.n_per_m3_m, 2400.0)
coarse = cr.msmpr_from_csd(d.L_m[::2], d.n_per_m3_m[::2], 2400.0)
print(f"all sizes: G = {full['G']*1e9:.2f} nm/s, L_D = {cr.msmpr_dominant_size(full['G'], 2400.0)*1e6:.0f} um; every other size: G = {coarse['G']*1e9:.2f} nm/s, L_D = {cr.msmpr_dominant_size(coarse['G'], 2400.0)*1e6:.0f} um")
print(f"a 20 % larger dominant size needs tau = {1.2 * 2400 / 60:.0f} min instead of 40 (at the same growth rate)")"""),
         md(r"""
Dropping half the sieve fractions changes the growth rate by a few percent - the slope of a straight line
on a semi-log plot is robust - and the dominant size with it. Residence time scales the dominant size
linearly at constant growth rate, so 20 % more size means 20 % more time in the crystalliser.
""")],
        [code(r"""L, n = d.L_m.to_numpy(), d.n_per_m3_m.to_numpy()
trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
m3 = trap(n * L**3, L)
k_v, rho_c = np.pi / 6, 2660.0
fit = cr.msmpr_from_csd(L, n, 2400.0)
m3_closed = 6 * fit["n0"] * (fit["G"] * 2400.0) ** 4
print(f"third moment from the data: {m3:.3e} m3/m3 (magma density {k_v*rho_c*m3:.0f} kg/m3); closed form 6 n0 (G tau)^4: {m3_closed:.3e} - the data stop at 800 um and miss the tail")"""),
         md(r"""
The measured moments fall short of the closed form because the sieves stop at 800 micrometres and the mass
distribution has a long tail; integrating data means integrating only what was measured. The closed form
extrapolates the exponential - one more reason to fit the line rather than sum the fractions.
""")],
    ]


# =====================================================================================================
@solutions("15_drying_and_evaporation", imports=["from engsep import datasets", "from engsep import drying_evaporation as de"],
           setup=r"""d = pd.read_csv(datasets.path("drying_curve.csv"), comment="#")
an = de.drying_curve_analysis(d.t_s, d.X_dry_basis)
X_c, X_eq, R_c = an["X_critical"], 0.06, an["R_c"]""")
def _():
    return [
        [code(r"""H = 0.01
T_wb60, T_wb80 = de.wet_bulb_temperature(60.0, H), de.wet_bulb_temperature(80.0, H)
R_c80 = R_c * (80.0 - T_wb80) / (60.0 - T_wb60)
t60 = de.drying_time(0.9, 0.10, X_c, X_eq, R_c, 1.0)["t_total"]
t80 = de.drying_time(0.9, 0.10, X_c, X_eq, R_c80, 1.0)["t_total"]
print(f"wet-bulb {T_wb60:.1f} degC at 60 degC, {T_wb80:.1f} at 80: constant rate x{R_c80/R_c:.2f}; drying time {t60/60:.0f} -> {t80/60:.0f} min")
print("the falling-rate period is shortened in the same proportion here because the model scales it with R_c; in reality internal diffusion depends on the solid's temperature and structure, and hotter air helps less than proportionally")"""),
         md(r"""
Hotter air raises the wet-bulb driving force by about 40 % and shortens the constant-rate period
accordingly; the simple model shortens the falling-rate period too, but real falling-rate drying is
limited by moisture movement inside the solid, which air temperature affects only through the solid's
temperature. Hot air is the lever for the surface, not the interior.
""")],
        [code(r"""# backward feed by hand: the cold feed enters the coldest effect (3), is heated there by the vapour of effect 2, and so on
F, x_f, x_p = 2.0, 0.10, 0.40
V_total = F * (1 - x_f / x_p)
T = [107.0, 90.0, 60.0]                                                  # boiling temperatures of effects 1, 2, 3 (as in the forward-feed table)
# forward feed: cold feed to effect 1 at 107 degC
Q_fwd_1 = F * 4000 * (T[0] - 25.0) + V_total / 3 * de.latent_heat(T[0])
# backward feed: cold feed to effect 3 at 60 degC, then pumped hotter: each effect heats its liquor by the step to its own temperature
Q_bwd_3 = F * 4000 * (T[2] - 25.0) + V_total / 3 * de.latent_heat(T[2])
Q_bwd_2 = (F - V_total / 3) * 4000 * (T[1] - T[2]) + V_total / 3 * de.latent_heat(T[1])
Q_bwd_1 = (F - 2 * V_total / 3) * 4000 * (T[0] - T[1]) + V_total / 3 * de.latent_heat(T[0])
print(f"forward feed: the first effect must supply {Q_fwd_1/1e6:.2f} MW (of which {F*4000*(T[0]-25)/1e6:.2f} MW heats the cold feed) with live steam")
print(f"backward feed: live steam supplies only effect 1's {Q_bwd_1/1e6:.2f} MW; the feed's sensible heating ({F*4000*(T[2]-25)/1e6:.2f} MW) is done in effect 3 by cheap low-pressure vapour")"""),
         md(r"""
With forward feed the cold feed is heated in the first effect by live steam, the most expensive heat in the
train; with backward feed it enters the last effect and is heated by the last effect's vapour, which would
otherwise go to the condenser. Backward feed also puts the viscous concentrate in the hottest effect, where
it is least viscous. The price is pumps between effects and the concentrate leaving hot.
""")],
        [code(r"""F, x_f, x_p, T_f, T_b, T_s, U, bpr = 2.0, 0.10, 0.40, 25.0, 100.0, 120.0, 2000.0, 3.0
V = F * (1 - x_f / x_p)
T_v = T_b + bpr
Q = F * 4000.0 * (T_v - T_f) + V * de.latent_heat(T_v)
steam = Q / de.latent_heat(T_s)
lib = de.single_effect(F, x_f, x_p, T_f, T_b, T_s, U, bpr=bpr)
print(f"by hand: V = {V:.3f}, Q = {Q/1e6:.3f} MW, steam = {steam:.3f} kg/s, area = {Q/(U*(T_s - T_v)):.1f} m2; library: {lib['vapour_kg_s']:.3f}, {lib['Q_W']/1e6:.3f}, {lib['steam_kg_s']:.3f}, {lib['area_m2']:.1f}")"""),
         md(r"""
The single-effect balance is four lines of arithmetic, reproduced exactly.
""")],
    ]


# =====================================================================================================
@solutions("16_humidification_and_cooling_towers", imports=["from scipy import integrate, optimize", "from engsep import drying_evaporation as de"],
           setup=r"""def ntu(T_hot, T_cold, T_air, RH, L_over_G):
    H_in = de.enthalpy_moist_air(T_air, RH * de.humidity_saturation(T_air)) / 1e3
    driving = lambda Tw: de.enthalpy_moist_air(Tw, de.humidity_saturation(Tw)) / 1e3 - (H_in + L_over_G * 4.186 * (Tw - T_cold))
    if min(driving(Tw) for Tw in np.linspace(T_cold, T_hot, 60)) <= 0:
        return np.inf                                                   # the operating line touches the saturation curve: a pinch
    return integrate.quad(lambda Tw: 4.186 / driving(Tw), T_cold, T_hot)[0]""")
def _():
    return [
        [code(r"""NTU_design = ntu(40.0, 28.0, 30.0, 0.4, 1.2)
T_wb = de.wet_bulb_temperature(34.0, 0.7 * de.humidity_saturation(34.0))
T_cold_new = optimize.brentq(lambda Tc: ntu(40.0, Tc, 34.0, 0.7, 1.2) - NTU_design, T_wb + 0.3, 39.0)
print(f"design NTU {NTU_design:.2f}; on a 34 degC / 70 % day (wet-bulb {T_wb:.1f} degC) the same tower delivers water at {T_cold_new:.1f} degC instead of 28")
LG_new = optimize.brentq(lambda lg: ntu(40.0, 28.0, 34.0, 0.7, lg) - NTU_design, 0.2, 1.2) if ntu(40.0, 28.0, 34.0, 0.7, 0.2) < NTU_design else None
print(f"to hold 28 degC the water rate must drop to L/G = {LG_new:.2f}" if LG_new else "28 degC cannot be held on that day at any water rate: the approach to wet-bulb would be too small")"""),
         md(r"""
On a humid summer day the wet-bulb temperature rises by six kelvin and the tower's cold water rises with it;
holding the design temperature would need the water rate cut - or, as the calculation may find, is not
possible at all because the approach would shrink below what any tower achieves. The design wet-bulb is
chosen from the hottest weeks for exactly this reason.
""")],
        [code(r"""char = lambda lg: 1.8 * lg ** (-0.6)
LG = optimize.brentq(lambda lg: char(lg) - ntu(40.0, 28.0, 30.0, 0.4, lg), 0.3, 3.0)
print(f"the packing meets the demand at L/G = {LG:.2f}: {1/LG:.2f} kg of air per kg of water, KaV/L = {char(LG):.2f}")"""),
         md(r"""
The demand curve (Merkel's integral) rises with L/G and the packing's characteristic falls; their crossing
is the operating point, and the air rate follows. Tower selection is this one intersection.
""")],
        [code(r"""H_in = de.enthalpy_moist_air(30.0, 0.4 * de.humidity_saturation(30.0)) / 1e3
f = lambda Tw: 4.186 / (de.enthalpy_moist_air(Tw, de.humidity_saturation(Tw)) / 1e3 - (H_in + 1.2 * 4.186 * (Tw - 28.0)))
Tw = np.linspace(28.0, 40.0, 11)
h = Tw[1] - Tw[0]
simpson = h / 3 * (f(Tw[0]) + 4 * sum(f(t) for t in Tw[1:-1:2]) + 2 * sum(f(t) for t in Tw[2:-1:2]) + f(Tw[-1]))
print(f"Simpson (10 intervals): {simpson:.4f}; quad: {ntu(40.0, 28.0, 30.0, 0.4, 1.2):.4f}")"""),
         md(r"""
Simpson's rule on ten intervals matches adaptive quadrature to four decimals: the integrand is smooth, and
this is the method the classic cooling-tower charts were computed with.
""")],
    ]


# =====================================================================================================
@solutions("17_project_column_data_to_retrofit", imports=["from engsep import datasets, vle", "from engsep import distillation as dist", "from engsep.vle import y_from_alpha"],
           setup=r"""p = pd.read_csv(datasets.path("plant_column_benzene_toluene.csv"), comment="#")
q = p[p.note != "target for the retrofit"].copy()
q.loc[q.xD > 1, "xD"] /= 100; q.loc[q.T_top_C > 200, "T_top_C"] -= 273.15; q = q[q.zF > 0]
alpha = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
y_eq = lambda x: y_from_alpha(x, alpha)""")
def _():
    return [
        [code(r"""row = q.iloc[0]
mt = dist.mccabe_thiele(y_eq, row.xD, row.xB, row.zF, row.R)
E = 10 / 18
opt_tray = round(mt["feed_stage"] / E)
print(f"day {int(row.day)}: optimum feed on theoretical stage {mt['feed_stage']} of {mt['stages']} -> real tray {opt_tray} from the top (actual feed tray 9)")
# stages lost with the feed two trays too high: step with the feed forced onto stage feed-2 (rectifying line continued past the intersection)
def stages_with_feed_at(k_feed):
    ol = mt["lines"]
    x = y = row.xD; n = 0
    while x > row.xB and n < 200:
        x = float(__import__("scipy").optimize.brentq(lambda xx: y_eq(xx) - y, 0, 1)); n += 1
        if x <= row.xB:
            break
        y = ol["rect"](x) if n < k_feed else ol["strip"](x)
        if y <= y_eq(x) and False:
            pass
    return n
print(f"stages needed with the feed at stage {mt['feed_stage'] - 2} instead of {mt['feed_stage']}: {stages_with_feed_at(mt['feed_stage'] - 2)} (optimal: {mt['stages']})")"""),
         md(r"""
The optimum feed stage sits where the staircase crosses the operating lines' intersection - theoretical
stage 6, real tray 11 of 18 - two trays below the actual feed tray 9: the feed is entering slightly too high.
Forcing the feed two theoretical stages early makes the column step along the stripping line where the
rectifying line would have been better and costs two stages; moving a feed nozzle is a cheap fix for a
common inefficiency.
""")],
        [code(r"""rho_L, rho_V, sigma, D_shell = 800.0, 2.8, 0.021, 1.2
u_f = 0.08 * (sigma / 0.02) ** 0.2 * np.sqrt((rho_L - rho_V) / rho_V)
area = np.pi * D_shell ** 2 / 4
D_flow = 45.0 / 3600 * 29.0                                             # distillate molar flow x molar volume, m3/s per unit (R+1)
for R in (1.88, 2.5, 3.0):
    V = (R + 1) * D_flow
    u = V / area
    print(f"R = {R}: vapour velocity {u:.2f} m/s = {u/u_f:.0%} of flooding ({u_f:.2f} m/s)")
R_limit = 0.8 * u_f * area / D_flow - 1
print(f"80 % of flooding is reached at R = {R_limit:.2f}")"""),
         md(r"""
The 1.2 m shell carries the retrofit reflux ratio of 1.9 at about two thirds of flooding, and the 80 %
limit is reached at R = 2.4: enough margin for normal operation, not for a large reflux increase. Structured
packing's higher capacity than trays at the same diameter (the Souders-Brown factor above is a tray value)
is one more argument for it, and the vendor's capacity curve should confirm the margin.
""")],
        [code(r"""def column_stages(df):
    n = df.apply(lambda r: dist.mccabe_thiele(y_eq, r.xD, r.xB, r.zF, r.R)["stages"], axis=1)
    return n, int(round(n.median()))
n, med = column_stages(q)
print("stage counts by day:", n.tolist()); print("median:", med)"""),
         md(r"""
One line per day and a median: the reproducible core of the project's performance assessment.
""")],
    ]


if __name__ == "__main__":
    write_all(only=[a for a in sys.argv[1:] if not a.startswith("--")])
