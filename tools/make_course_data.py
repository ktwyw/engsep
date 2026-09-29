"""Write the course data files into src/engsep/data. Literature tables are transcribed with their sources;
pseudo-experimental files are generated from the library's own models with the parameters, noise level and
deliberate faults stated in each header (seed 2026). Run: python tools/make_course_data.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engsep import adsorption as ad  # noqa: E402
from engsep import crystallization as cr  # noqa: E402
from engsep import distillation as dist  # noqa: E402
from engsep import vle  # noqa: E402
from engsep.vle import y_from_alpha  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "engsep" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)


def write(name, header_lines, rows, columns):
    def cell(v):
        v = str(v)
        return f'"{v}"' if "," in v else v

    lines = [f"# {line}" for line in header_lines] + [",".join(columns)] + [",".join(cell(v) for v in r) for r in rows]
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


# ---- literature: ethanol-water VLE at 1 atm
x = [
    0.0,
    0.019,
    0.0721,
    0.0966,
    0.1238,
    0.1661,
    0.2337,
    0.2608,
    0.3273,
    0.3965,
    0.5079,
    0.5198,
    0.5732,
    0.6763,
    0.7472,
    0.8943,
    1.0,
]
y = [
    0.0,
    0.17,
    0.3891,
    0.4375,
    0.4704,
    0.5089,
    0.5445,
    0.558,
    0.5826,
    0.6122,
    0.6564,
    0.6599,
    0.6841,
    0.7385,
    0.7815,
    0.8943,
    1.0,
]
T = [100.0, 95.5, 89.0, 86.7, 85.3, 84.1, 82.7, 82.3, 81.5, 80.7, 79.8, 79.7, 79.3, 78.74, 78.41, 78.15, 78.3]
write(
    "vle_ethanol_water.csv",
    [
        "Vapour-liquid equilibrium of ethanol (1) - water (2) at 1 atm: mole fractions and temperature.",
        "Data of Carey & Lewis (1932) as tabulated in Perry's Chemical Engineers' Handbook, Table 13-1.",
    ],
    list(zip(x, y, T)),
    ["x_ethanol", "y_ethanol", "T_C"],
)

# ---- literature: gas permeabilities (Barrer) of common membrane polymers
write(
    "gas_permeabilities.csv",
    [
        "Gas permeabilities in Barrer (1 Barrer = 3.35e-16 mol m/(m2 s Pa)) of common membrane polymers at 25-35 degC.",
        "Indicative values from Baker, Membrane Technology and Applications (3rd ed.), Table 8.1, and Robeson (2008).",
    ],
    [
        ("polysulfone", 5.6, 1.4, 0.25, 14.0, 0.25),
        ("cellulose acetate", 2.6, 0.8, 0.2, 6.0, 0.2),
        ("polyimide (Matrimid)", 28.0, 2.1, 0.32, 10.0, 0.28),
        ("silicone rubber (PDMS)", 650.0, 600.0, 280.0, 3200.0, 950.0),
        ("polycarbonate", 12.0, 1.4, 0.3, 6.5, 0.36),
        ("polyethersulfone", 7.8, 0.6, 0.11, 3.4, 0.1),
    ],
    ["polymer", "H2", "O2", "N2", "CO2", "CH4"],
)

# ---- literature: the solubility table (exported from the library)
write(
    "solubility.csv",
    [
        "Solubility of salts in water, g anhydrous solute per 100 g water, at 0-100 degC. CRC Handbook of Chemistry and Physics."
    ],
    [(salt, *vals) for salt, vals in cr.SOLUBILITY.items()],
    ["salt", "T0", "T20", "T40", "T60", "T80", "T100"],
)

# ---- a ternary system for extraction: acetic acid (A) - water (C) - isopropyl ether (S) at 20 degC
raff = [
    (0.0069, 0.012),
    (0.0141, 0.015),
    (0.0289, 0.016),
    (0.0642, 0.019),
    (0.133, 0.023),
    (0.255, 0.034),
    (0.367, 0.044),
    (0.443, 0.106),
    (0.464, 0.165),
]
extr = [
    (0.0018, 0.993),
    (0.0037, 0.989),
    (0.0079, 0.984),
    (0.0193, 0.971),
    (0.0482, 0.933),
    (0.114, 0.847),
    (0.216, 0.715),
    (0.311, 0.580),
    (0.362, 0.487),
]
write(
    "ternary_acetic_water_ether.csv",
    [
        "Liquid-liquid equilibrium of acetic acid (A) - water (C) - isopropyl ether (S) at 20 degC, mass fractions;",
        "raffinate (water-rich) and extract (ether-rich) ends of each tie line. Treybal, Mass-Transfer Operations (3rd ed.), Table 10.1-type data;",
        "values transcribed from Seader, Henley & Roper, Example 8.1. Columns: x_A and x_S of the raffinate, y_A and y_S of the extract.",
    ],
    [(r[0], r[1], e[0], e[1]) for r, e in zip(raff, extr)],
    ["xA_raffinate", "xS_raffinate", "yA_extract", "yS_extract"],
)

# ---- adsorption isotherm measurements (batch, phenol on activated carbon-like)
c = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
q = ad.langmuir(c, 2.5, 0.3) * (1 + rng.normal(0, 0.04, c.size))
write(
    "isotherm_phenol_carbon.csv",
    [
        "Equilibrium loading of phenol on activated carbon at 25 degC from batch experiments. Generated from a Langmuir",
        "isotherm with q_max = 2.5 mol/kg and K = 0.3 m3/mol with 4 % noise. Columns: c (mol/m3), q (mol/kg).",
    ],
    list(zip(c, np.round(q, 4))),
    ["c_mol_m3", "q_mol_kg"],
)

# ---- fixed-bed breakthrough curve
t = np.arange(0, 20001, 500.0)
curve = ad.bohart_adams(t, 10.0, 2.0, 500.0, 1.0, 0.01, k=6e-4) * (1 + rng.normal(0, 0.02, t.size))
curve = np.clip(curve, 0, None)
curve[12] = curve[12] * 3 + 0.05  # a contaminated sample
write(
    "breakthrough_curve.csv",
    [
        "Outlet concentration ratio c/c0 of a 1 m laboratory adsorption column (u = 1 cm/s, c0 = 10 mol/m3). Generated from",
        "the Bohart-Adams model (q0 = 2 mol/kg, rho_b = 500 kg/m3, k = 6e-4 m3/(mol s)) with 2 % noise; sample 13 (t = 6000 s)",
        "was contaminated. Columns: time (s), c/c0.",
    ],
    list(zip(t, np.round(curve, 4))),
    ["t_s", "c_over_c0"],
)

# ---- drying curve of a filter cake
t = np.arange(0, 7201, 300.0)
X = np.where(t < 3000, 0.9 - 1.2e-4 * t, 0.06 + 0.48 * np.exp(-(t - 3000) * 0.6e-4 / 0.48)) * (
    1 + rng.normal(0, 0.015, t.size)
)
write(
    "drying_curve.csv",
    [
        "Moisture content (dry basis) of a filter cake dried in a tray dryer at 60 degC, 2 m/s air. Generated with a constant rate",
        "of 1.2e-4 kg/(kg s) to a critical moisture of 0.54 and a linear falling-rate period to X* = 0.06, with 1.5 % noise.",
        "Columns: time (s), X (kg water / kg dry solid).",
    ],
    list(zip(t, np.round(X, 4))),
    ["t_s", "X_dry_basis"],
)

# ---- crystal size distribution from an MSMPR crystalliser
L = np.array([25, 50, 100, 150, 200, 300, 400, 500, 600, 800.0]) * 1e-6
n = cr.msmpr_population(L, 2e17, 8e-9, 2400.0) * (1 + rng.normal(0, 0.05, L.size))
write(
    "csd_msmpr.csv",
    [
        "Population density of potassium sulphate crystals from a continuous MSMPR crystalliser (tau = 40 min), by sieve",
        "analysis. Generated from n = n0 exp(-L/(G tau)) with n0 = 2e17 1/(m3 m) and G = 8e-9 m/s with 5 % noise.",
        "Columns: size (m), population density (1/(m3 m)).",
    ],
    list(zip(L, [f"{v:.4e}" for v in n])),
    ["L_m", "n_per_m3_m"],
)

# ---- project: plant data of a benzene-toluene column over 30 days
alpha = vle.geometric_mean_alpha(("benzene", "toluene"), ideal=True)
rows = []
for day in range(1, 31):
    zF = 0.45 + rng.normal(0, 0.02)
    R = 2.2 + 0.15 * np.sin(day / 5) + rng.normal(0, 0.05)
    F = 100.0 + rng.normal(0, 3)
    q = 1.0
    # the column has 18 real trays with an overall efficiency of 0.6 -> ~11 theoretical stages incl. reboiler: find xD, xB it delivers
    N_theo = 11
    xD = 0.95
    # solve for xB given N_theo stages, R, zF: scan xB
    best = None
    for xB in np.linspace(0.01, 0.2, 400):
        try:
            n = dist.mccabe_thiele(lambda x: y_from_alpha(x, alpha), xD, xB, zF, R, q)["stages"]
        except ValueError:
            continue
        if n <= N_theo:
            best = xB
            break
    xB = best if best is not None else 0.2
    xD_m, xB_m = xD * (1 + rng.normal(0, 0.004)), xB * (1 + rng.normal(0, 0.03))
    T_top = vle.dew_T(("benzene", "toluene"), xD_m, ideal=True) - 273.15 + rng.normal(0, 0.3)
    T_bot = vle.bubble_T(("benzene", "toluene"), xB_m, ideal=True) - 273.15 + rng.normal(0, 0.4)
    rows.append(
        (
            day,
            round(F, 1),
            round(zF, 3),
            round(R, 2),
            round(xD_m, 4),
            round(xB_m, 4),
            round(T_top, 1),
            round(T_bot, 1),
            "ok",
        )
    )
rows[9] = (*rows[9][:4], rows[9][4] * 100, *rows[9][5:8], "ok")  # distillate purity logged in percent
rows[16] = (*rows[16][:2], 0.0, *rows[16][3:8], "feed analyser offline")  # missing feed composition
rows[22] = (*rows[22][:6], rows[22][6] + 273.15, *rows[22][7:8], "ok")  # top temperature in K
rows.append((31, 100.0, 0.45, 2.2, 0.985, 0.015, 0.0, 0.0, "target for the retrofit"))
write(
    "plant_column_benzene_toluene.csv",
    [
        "Daily averages from a benzene-toluene distillation column (18 sieve trays, total condenser, kettle reboiler, 1 atm)",
        "over 30 days: feed (kmol/h), feed benzene mole fraction, reflux ratio, distillate and bottoms benzene mole fractions,",
        "top and bottom temperatures (degC). Generated from constant relative volatility (geometric mean of the Antoine values)",
        "with 11 theoretical stages (overall efficiency 0.6), temperatures from the product dew and bubble points, and measurement noise; faults: one distillate purity in percent, one",
        "missing feed composition (logged as 0), one top temperature in K. Row 31 is the retrofit target, not a measurement.",
        "Columns: day, F, zF, R, xD, xB, T_top, T_bottom, note.",
    ],
    rows,
    ["day", "F_kmol_h", "zF", "R", "xD", "xB", "T_top_C", "T_bottom_C", "note"],
)
