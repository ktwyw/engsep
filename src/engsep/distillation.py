"""Binary and multicomponent distillation: the McCabe-Thiele construction (operating lines, the q-line,
stage-by-stage stepping with feed-stage location and Murphree efficiency), minimum reflux and minimum stages
for a binary from any equilibrium curve, the Fenske-Underwood-Gilliland shortcut with the Kirkbride
feed-stage estimate, batch (Rayleigh) distillation, packed columns by HETP, and column material balances.

>>> from engsep import distillation as dist
>>> from engsep.vle import y_from_alpha
>>> mt = dist.mccabe_thiele(lambda x: y_from_alpha(x, 2.5), xD=0.95, xB=0.05, zF=0.5, R=2.0, q=1.0)
>>> mt["stages"], mt["feed_stage"]
(11, 5)
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize


# ------------------------------------------------------------------ material balances
def column_balance(F: float, zF: float, xD: float, xB: float) -> dict:
    """Distillate and bottoms flows from the overall and component balances."""
    D = F * (zF - xB) / (xD - xB)
    return {"D": D, "B": F - D, "recovery_light": D * xD / (F * zF)}


def operating_lines(xD: float, xB: float, zF: float, R: float, q: float) -> dict:
    """Rectifying line y = R/(R+1) x + xD/(R+1); q-line y = q/(q-1) x - zF/(q-1); stripping line through (xB, xB)
    and the intersection of the other two."""
    if abs(q - 1) < 1e-12:
        x_int = zF
    else:
        x_int = (zF / (q - 1) + xD / (R + 1)) / (q / (q - 1) - R / (R + 1))
    y_int = R / (R + 1) * x_int + xD / (R + 1)
    slope_s = (y_int - xB) / (x_int - xB)
    return {
        "rect": lambda x: R / (R + 1) * x + xD / (R + 1),
        "strip": lambda x: xB + slope_s * (x - xB),
        "q_line": (lambda x: q / (q - 1) * x - zF / (q - 1)) if abs(q - 1) > 1e-12 else None,
        "x_int": x_int,
        "y_int": y_int,
        "slope_strip": slope_s,
    }


# ------------------------------------------------------------------ McCabe-Thiele
def _x_on_curve(y_eq, y: float) -> float:
    """x in equilibrium with y (inverse of the equilibrium curve)."""
    return float(optimize.brentq(lambda xx: float(y_eq(xx)) - y, 0.0, 1.0))


def _operating(ol: dict, x: float) -> float:
    return ol["rect"](x) if x > ol["x_int"] else ol["strip"](x)


def _x_murphree(y_eq, ol: dict, y: float, E: float) -> float:
    """With a Murphree vapour efficiency E the vapour leaving a stage reaches only part of the way from the
    operating line to equilibrium: y = y_op(x) + E (y_eq(x) - y_op(x)); solve for the liquid x leaving."""
    return float(
        optimize.brentq(lambda xx: _operating(ol, xx) + E * (float(y_eq(xx)) - _operating(ol, xx)) - y, 0.0, 1.0)
    )


def mccabe_thiele(
    y_eq, xD: float, xB: float, zF: float, R: float, q: float = 1.0, murphree: float = 1.0, max_stages: int = 500
) -> dict:
    """Step off stages from the top (total condenser) with the equilibrium curve y_eq(x). Returns the number of
    stages including the reboiler (ideal, or with a Murphree vapour efficiency), the feed stage counted from the
    top, and the stair-step coordinates for plotting. Raises if the reflux ratio is below the minimum."""
    ol = operating_lines(xD, xB, zF, R, q)
    if ol["x_int"] < xB or ol["y_int"] > float(y_eq(ol["x_int"])):
        raise ValueError(
            "The operating lines intersect above the equilibrium curve: the reflux ratio is below the minimum."
        )
    x = y = xD
    xs, ys = [x], [y]
    stages, feed_stage = 0, None
    while x > xB and stages < max_stages:
        x = _x_on_curve(y_eq, y) if murphree >= 1.0 else _x_murphree(y_eq, ol, y, murphree)
        stages += 1
        xs.append(x)
        ys.append(y)
        if feed_stage is None and x <= ol["x_int"]:
            feed_stage = stages
        if x <= xB:
            break
        y = _operating(ol, x)
        xs.append(x)
        ys.append(y)
    return {"stages": stages, "feed_stage": feed_stage, "x": np.array(xs), "y": np.array(ys), "lines": ol}


def minimum_reflux(y_eq, xD: float, zF: float, q: float = 1.0) -> dict:
    """R_min from the pinch where the q-line meets the equilibrium curve, checked against a tangent pinch (a curve
    with an inflection can touch the rectifying line before the q-line point; the larger R wins)."""
    if abs(q - 1) < 1e-12:
        x_p = zF
    else:
        x_p = float(optimize.brentq(lambda x: float(y_eq(x)) - (q / (q - 1) * x - zF / (q - 1)), 1e-9, 1 - 1e-9))
    y_p = float(y_eq(x_p))
    R_pinch = (xD - y_p) / (y_p - x_p)

    def R_of(x):
        yv = float(y_eq(x))
        return (xD - yv) / (yv - x) if yv > x else np.inf

    R_tangent = max(R_of(x) for x in np.linspace(x_p, xD, 400)[1:-1])
    return {
        "R_min": float(max(R_pinch, R_tangent)),
        "x_pinch": x_p,
        "y_pinch": y_p,
        "tangent_pinch": bool(R_tangent > R_pinch + 1e-9),
    }


def minimum_stages(y_eq, xD: float, xB: float, max_stages: int = 500) -> int:
    """Stages at total reflux (operating line y = x), stepped off from the top."""
    x, n = xD, 0
    while x > xB and n < max_stages:
        x = _x_on_curve(y_eq, x)
        n += 1
    return n


# ------------------------------------------------------------------ shortcut methods
def fenske(xD_lk: float, xB_lk: float, alpha: float, xD_hk: float | None = None, xB_hk: float | None = None) -> float:
    """Minimum stages N_min = ln[(xD_lk/xD_hk)(xB_hk/xB_lk)] / ln alpha (binary: xD_hk = 1 - xD_lk)."""
    xD_hk = 1 - xD_lk if xD_hk is None else xD_hk
    xB_hk = 1 - xB_lk if xB_hk is None else xB_hk
    return float(np.log((xD_lk / xD_hk) * (xB_hk / xB_lk)) / np.log(alpha))


def underwood(alphas, zF, xD, q: float = 1.0) -> dict:
    """Underwood's method for the minimum reflux of a multicomponent (or binary) mixture: find the roots theta of
    sum alpha_i z_i/(alpha_i - theta) = 1 - q between adjacent volatilities, then
    R_min = sum alpha_i xD_i/(alpha_i - theta) - 1 with the root between the keys (the largest R_min)."""
    alphas, zF, xD = np.asarray(alphas, float), np.asarray(zF, float), np.asarray(xD, float)
    f = lambda th: np.sum(alphas * zF / (alphas - th)) - (1 - q)  # noqa: E731
    order = np.argsort(alphas)
    roots = []
    for a, b in zip(alphas[order][:-1], alphas[order][1:]):
        try:
            roots.append(float(optimize.brentq(f, a + 1e-9, b - 1e-9)))
        except ValueError:
            pass
    R_mins = [float(np.sum(alphas * xD / (alphas - th)) - 1) for th in roots]
    R_min = max(R_mins) if R_mins else np.nan
    return {"theta": roots, "R_min": R_min}


def gilliland(R: float, R_min: float, N_min: float) -> float:
    """Number of theoretical stages from the Gilliland correlation in Molokanov's form."""
    X = (R - R_min) / (R + 1)
    if X <= 0:
        return float("inf")
    Y = 1 - np.exp((1 + 54.4 * X) / (11 + 117.2 * X) * (X - 1) / np.sqrt(X))
    return float((N_min + Y) / (1 - Y)) if Y < 1 else float("inf")


def kirkbride(N: float, B: float, D: float, zF_hk: float, zF_lk: float, xB_lk: float, xD_hk: float) -> dict:
    """Feed-stage location: N_R/N_S = [(B/D)(zF_hk/zF_lk)(xB_lk/xD_hk)^2]^0.206."""
    ratio = ((B / D) * (zF_hk / zF_lk) * (xB_lk / xD_hk) ** 2) ** 0.206
    N_S = N / (1 + ratio)
    return {"N_rectifying": N - N_S, "N_stripping": N_S, "feed_stage_from_top": N - N_S}


def fug(alpha: float, zF: float, xD: float, xB: float, R_over_Rmin: float = 1.3, q: float = 1.0) -> dict:
    """Fenske-Underwood-Gilliland shortcut for a binary with constant relative volatility."""
    N_min = fenske(xD, xB, alpha)
    R_min = underwood([alpha, 1.0], [zF, 1 - zF], [xD, 1 - xD], q)["R_min"]
    R = R_over_Rmin * R_min
    N = gilliland(R, R_min, N_min)
    return {"N_min": N_min, "R_min": R_min, "R": R, "N": N}


# ------------------------------------------------------------------ batch distillation
def rayleigh(y_eq, x0: float, x_final: float, W0: float = 1.0) -> dict:
    """Simple batch (Rayleigh) distillation: ln(W/W0) = int_{x0}^{x} dx/(y - x); returns the remaining charge and
    the average distillate composition."""
    val, _ = integrate.quad(lambda x: 1 / (float(y_eq(x)) - x), x_final, x0)
    W = W0 * np.exp(-val)
    D = W0 - W
    y_avg = (W0 * x0 - W * x_final) / D if D > 0 else np.nan
    return {"W": float(W), "D": float(D), "y_avg": float(y_avg)}


def batch_constant_reflux(y_eq, x0: float, x_final: float, R: float, W0: float = 1.0, N: int = 3) -> dict:
    """Batch rectification at constant reflux ratio R with N ideal stages above the still: for each still
    composition the distillate composition is the x_D from which N stages down the rectifying line reach the
    still; the charge follows the Rayleigh equation with y = x_D. Returns the still and distillate composition
    histories, the charge remaining and the accumulated distillate composition."""

    def xD_of_still(xW):
        # find xD such that N stages from xD along the rectifying line reach xW
        def stages_reach(xd):
            x = y = xd
            for _ in range(N):
                x = _x_on_curve(y_eq, y)
                y = R / (R + 1) * x + xd / (R + 1)
            return x - xW

        return float(optimize.brentq(stages_reach, xW + 1e-9, 1 - 1e-9))

    xs = np.linspace(x0, x_final, 60)
    xDs = np.array([xD_of_still(x) for x in xs])
    integrand = 1 / (xDs - xs)
    lnW = -np.concatenate([[0.0], integrate.cumulative_trapezoid(integrand, xs)])  # xs decreasing
    W = W0 * np.exp(-np.abs(lnW))
    return {"x_still": xs, "x_D": xDs, "W": W, "D": W0 - W, "y_avg": (W0 * x0 - W * xs) / np.maximum(W0 - W, 1e-12)}


# ------------------------------------------------------------------ packed columns and efficiencies
def packed_height(N_theoretical: float, HETP: float) -> float:
    return N_theoretical * HETP


def overall_efficiency(N_theoretical: float, N_actual: float) -> float:
    return N_theoretical / N_actual


def oconnell_efficiency(alpha: float, mu_mPa_s: float) -> float:
    """O'Connell's correlation for the overall plate efficiency of distillation columns: E = 0.492 (alpha mu)^-0.245."""
    return float(min(0.492 * (alpha * mu_mPa_s) ** (-0.245), 1.0))
