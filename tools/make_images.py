"""Generate the README images (docs/images/*.png and *.gif) from the library. Run: python tools/make_images.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engsep import absorption as ab  # noqa: E402
from engsep import adsorption as ad  # noqa: E402
from engsep import crystallization as cr  # noqa: E402
from engsep import datasets, flash, vle  # noqa: E402
from engsep import distillation as dist  # noqa: E402
from engsep import extraction as ex  # noqa: E402
from engsep.vle import y_from_alpha  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 1. T-x-y with an azeotrope
d = vle.Txy(("ethanol", "water"), n=61)
data = pd.read_csv(datasets.path("vle_ethanol_water.csv"), comment="#")
az = vle.azeotrope("ethanol", "water")
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.plot(d["x"], d["T"] - 273.15, label="bubble (Wilson)")
ax.plot(d["y"], d["T"] - 273.15, label="dew (Wilson)")
ax.plot(data.x_ethanol, data.T_C, "ko", ms=3, label="data (Perry)")
ax.plot(az["x"], az["T"] - 273.15, "r*", ms=12, label=f"azeotrope {az['x']:.2f}")
ax.set(xlabel="x, y ethanol", ylabel="T (degC)", title="Ethanol-water at 1 atm: the azeotrope")
ax.legend(frameon=False, fontsize=8)
save(fig, "txy_ethanol_water.png")

# ------------------------------------------------------------------ 2. Rachford-Rice
z, K = np.array([0.3, 0.4, 0.3]), np.array([2.9, 0.97, 0.34])
psi = np.linspace(0, 1, 300)
f = [np.sum(z * (K - 1) / (1 + p * (K - 1))) for p in psi]
rr = flash.rachford_rice(z, K)
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.plot(psi, f)
ax.axhline(0, color="k", ls=":")
ax.plot(rr["V_over_F"], 0, "ro", ms=8, label=f"V/F = {rr['V_over_F']:.3f}")
ax.set(xlabel="vapour fraction V/F", ylabel="Rachford-Rice function", title="Flash: one monotone equation, one root")
ax.legend(frameon=False)
save(fig, "rachford_rice.png")

# ------------------------------------------------------------------ 3. Kremser
fig, ax = plt.subplots(figsize=(5.6, 4))
Ns = np.arange(1, 16)
for A in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
    ax.plot(Ns, [ab.kremser_fraction_absorbed(A, N) for N in Ns], "o-", ms=3, label=f"A = {A}")
ax.set(
    xlabel="equilibrium stages",
    ylabel="fraction absorbed",
    ylim=(0, 1.02),
    title="Kremser: the absorption factor decides",
)
ax.legend(frameon=False, fontsize=8, ncol=2)
save(fig, "kremser.png")

# ------------------------------------------------------------------ 4. Hunter-Nash
t = pd.read_csv(datasets.path("ternary_acetic_water_ether.csv"), comment="#")
tern = ex.Ternary(
    np.c_[t.xA_raffinate, t.xS_raffinate], np.c_[t.yA_extract, t.yS_extract], np.c_[t.xA_raffinate, t.yA_extract]
)
hn = ex.hunter_nash(tern, 100.0, (0.30, 0.0), 250.0, (0.0, 1.0), xR_final=0.02)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.plot(t.xA_raffinate, t.xS_raffinate, "C0-", label="raffinate branch")
ax.plot(t.yA_extract, t.yS_extract, "C1-", label="extract branch")
for r, e in zip(hn["raffinates"], hn["extracts"]):
    ax.plot([r[0], e[0]], [r[1], e[1]], "k-", lw=0.8)
ax.plot(*hn["x_M"], "ks", label="M")
ax.plot(0.30, 0.0, "C2o", label="feed")
ax.plot(0.0, 1.0, "C2^", label="solvent")
ax.set(
    xlabel="mass fraction acetic acid",
    ylabel="mass fraction isopropyl ether",
    title=f"Hunter-Nash: {hn['stages']} stages (tie lines drawn are stages)",
)
ax.legend(frameon=False, fontsize=8)
save(fig, "hunter_nash.png")

# ------------------------------------------------------------------ 5. breakthrough curve
b = pd.read_csv(datasets.path("breakthrough_curve.csv"), comment="#")
tt = np.linspace(0, 20000, 400)
t_star = ad.stoichiometric_time(float(ad.langmuir(10.0, 2.5, 0.3)), 10.0, 500.0, 0.4, 1.0, 0.01)
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.plot(b.t_s / 3600, b.c_over_c0, "o", ms=4, label="laboratory column")
ax.plot(tt / 3600, ad.bohart_adams(tt, 10.0, 2.0, 500.0, 1.0, 0.01, 6e-4), label="Bohart-Adams")
ax.axvline(2.0 * 500.0 / (0.01 * 10.0) / 3600, color="k", ls=":", label="ideal shock")
ax.set(xlabel="t (h)", ylabel="c / c0", title="Fixed-bed breakthrough")
ax.legend(frameon=False)
save(fig, "breakthrough.png")

# ------------------------------------------------------------------ 6. MSMPR
c = pd.read_csv(datasets.path("csd_msmpr.csv"), comment="#")
fit = cr.msmpr_from_csd(c.L_m, c.n_per_m3_m, 2400.0)
L = np.linspace(0, 1e-3, 300)
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.semilogy(c.L_m * 1e6, c.n_per_m3_m, "o", label="sieve analysis")
ax.semilogy(
    L * 1e6,
    cr.msmpr_population(L, fit["n0"], fit["G"], 2400.0),
    label=f"n0 exp(-L/G tau): G = {fit['G'] * 1e9:.1f} nm/s",
)
ax2 = ax.twinx()
ax2.plot(L * 1e6, cr.msmpr_mass_distribution(L, fit["G"], 2400.0) * 1e-6, "C3", alpha=0.7)
ax2.set_ylabel("mass density", color="C3")
ax.set(
    xlabel="crystal size (um)",
    ylabel="population density (1/(m3 m))",
    ylim=(1e10, 1e18),
    title="MSMPR crystal size distribution",
)
ax.legend(frameon=False, fontsize=8, loc="upper right")
save(fig, "msmpr.png")

# ------------------------------------------------------------------ hero: McCabe-Thiele staircase as the reflux ratio sweeps
alpha = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
y_eq = lambda x: y_from_alpha(x, alpha)  # noqa: E731
xD, xB, zF = 0.95, 0.05, 0.45
R_min = dist.minimum_reflux(y_eq, xD, zF)["R_min"]
ratios = np.concatenate([np.linspace(1.05, 4.0, 36), np.linspace(4.0, 1.05, 36)])
fig, ax = plt.subplots(figsize=(5.4, 5.4))
xs = np.linspace(0, 1, 200)
ax.plot(xs, y_eq(xs), "k", lw=1.5)
ax.plot([0, 1], [0, 1], "k:", lw=0.8)
(rect,) = ax.plot([], [], "C0", lw=1.2)
(strip,) = ax.plot([], [], "C1", lw=1.2)
(steps,) = ax.plot([], [], "C3", lw=1.8)
ax.plot([zF, zF], [zF, 1], "C2", lw=0.8)
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x benzene", ylabel="y benzene")
title = ax.set_title("")
fig.tight_layout()


def update(i):
    R = ratios[i] * R_min
    mt = dist.mccabe_thiele(y_eq, xD, xB, zF, R)
    ol = mt["lines"]
    rect.set_data([ol["x_int"], xD], [ol["y_int"], xD])
    strip.set_data([xB, ol["x_int"]], [xB, ol["y_int"]])
    steps.set_data(mt["x"], mt["y"])
    title.set_text(f"McCabe-Thiele: R = {ratios[i]:.2f} R_min -> {mt['stages']} stages")
    return rect, strip, steps, title


anim = FuncAnimation(fig, update, frames=len(ratios), blit=False)
anim.save(OUT / "mccabe_thiele.gif", writer=PillowWriter(fps=6), dpi=90)
plt.close(fig)
print("wrote mccabe_thiele.gif")
