"""Liquid-liquid extraction and leaching: immiscible-solvent extraction with a distribution coefficient (single,
cross-current and countercurrent stages, Kremser form), partially miscible ternary systems on triangular
coordinates (tie-line interpolation, the lever rule, the Hunter-Nash countercurrent construction), and
countercurrent leaching/washing with constant underflow.

>>> from engsep import extraction as ex
>>> round(ex.countercurrent_stages_immiscible(K=2.0, S_over_F=1.0, x_in=0.10, x_out=0.005), 2)  # 95 % recovery
3.39
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


# ------------------------------------------------------------------ immiscible solvent, linear distribution y = K x
def extraction_factor(K: float, S_over_F: float) -> float:
    """E = K S/F: solvent-to-feed ratio times the distribution coefficient."""
    return K * S_over_F


def single_stage_immiscible(K: float, S_over_F: float, x_in: float) -> dict:
    """One equilibrium stage: fraction remaining in the raffinate 1/(1 + E)."""
    E = extraction_factor(K, S_over_F)
    return {"x_out": x_in / (1 + E), "fraction_extracted": E / (1 + E), "y_out": K * x_in / (1 + E)}


def crosscurrent_immiscible(K: float, S_over_F_each: float, x_in: float, N: int) -> dict:
    """N cross-current stages, fresh solvent to each: x_N/x_in = 1/(1 + E)^N."""
    E = extraction_factor(K, S_over_F_each)
    return {"x_out": x_in / (1 + E) ** N, "fraction_extracted": 1 - (1 + E) ** (-N)}


def countercurrent_immiscible(K: float, S_over_F: float, x_in: float, N: int, y_in: float = 0.0) -> dict:
    """Kremser: (x_in - x_out)/(x_in - y_in/K) = (E^(N+1) - E)/(E^(N+1) - 1)."""
    E = extraction_factor(K, S_over_F)
    phi = N / (N + 1) if abs(E - 1) < 1e-12 else (E ** (N + 1) - E) / (E ** (N + 1) - 1)
    x_out = x_in - phi * (x_in - y_in / K)
    return {"x_out": x_out, "fraction_extracted": (x_in - x_out) / x_in, "E": E}


def countercurrent_stages_immiscible(K: float, S_over_F: float, x_in: float, x_out: float, y_in: float = 0.0) -> float:
    E = extraction_factor(K, S_over_F)
    phi = (x_in - x_out) / (x_in - y_in / K)
    if abs(E - 1) < 1e-12:
        return phi / (1 - phi)
    return float(np.log((E - phi) / (1 - phi)) / np.log(E) - 1)


def minimum_solvent_immiscible(K: float, x_in: float, x_out: float, y_in: float = 0.0) -> float:
    """S/F at which infinite stages are needed: the extract leaving in equilibrium with the feed, y_out = K x_in."""
    return (x_in - x_out) / (K * x_in - y_in)


# ------------------------------------------------------------------ ternary systems on triangular coordinates
class Ternary:
    """A type-I ternary (solute A, carrier C, solvent S), compositions as (x_A, x_S) mass fractions with
    x_C = 1 - x_A - x_S. The two branches of the binodal are given as points (x_A, x_S) - the raffinate branch
    (carrier-rich) and the extract branch (solvent-rich) - and the tie lines as pairs (x_A raffinate, x_A extract).
    Everything else is interpolation along the branches."""

    def __init__(self, raffinate_branch, extract_branch, tie_lines):
        r, e = np.asarray(raffinate_branch, float), np.asarray(extract_branch, float)
        self.rA, self.rS = r[np.argsort(r[:, 0]), 0], r[np.argsort(r[:, 0]), 1]
        self.eA, self.eS = e[np.argsort(e[:, 0]), 0], e[np.argsort(e[:, 0]), 1]
        t = np.asarray(tie_lines, float)
        if t[:, 0].min() > 0:  # the dilute limit: the tie line degenerates to the carrier-solvent base (0, 0)
            t = np.vstack([[0.0, 0.0], t])
        self.tie_r, self.tie_e = t[np.argsort(t[:, 0]), 0], t[np.argsort(t[:, 0]), 1]

    def raffinate(self, a):
        """Point on the raffinate branch with solute fraction a."""
        return np.array([a, np.interp(a, self.rA, self.rS)])

    def extract(self, a):
        return np.array([a, np.interp(a, self.eA, self.eS)])

    def extract_in_equilibrium(self, a_raffinate):
        """Extract on the tie line through the raffinate of solute fraction a_raffinate."""
        return self.extract(float(np.interp(a_raffinate, self.tie_r, self.tie_e)))

    def raffinate_in_equilibrium(self, a_extract):
        return self.raffinate(float(np.interp(a_extract, self.tie_e, self.tie_r)))

    def distribution_coefficient(self, a_raffinate):
        return float(np.interp(a_raffinate, self.tie_r, self.tie_e)) / a_raffinate


def mixing_point(F, xF, S, xS) -> tuple:
    """Lever rule: the mixture of F at xF and S at xS (compositions as (x_A, x_S))."""
    M = F + S
    return M, (F * np.asarray(xF, float) + S * np.asarray(xS, float)) / M


def lever_split(xM, x1, x2) -> float:
    """Fraction of the mixture M that is phase 1, from the lever rule (distance ratio along the tie line)."""
    xM, x1, x2 = map(np.asarray, (xM, x1, x2))
    return float(np.linalg.norm(x2 - xM) / np.linalg.norm(x2 - x1))


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _extract_on_line(tern: Ternary, p, q, a_hi: float | None = None):
    """Solute fraction of the extract-branch point collinear with p and q (searching below a_hi)."""
    a_hi = tern.eA.max() - 1e-9 if a_hi is None else a_hi
    f = lambda a: _cross(tern.extract(a) - np.asarray(p), np.asarray(q) - np.asarray(p))  # noqa: E731
    grid = np.linspace(1e-6, a_hi, 400)
    vals = [f(a) for a in grid]
    for a1, a2, f1, f2 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if f1 * f2 < 0:
            return float(optimize.brentq(f, a1, a2))
    return None


def hunter_nash(tern: Ternary, F: float, xF, S: float, xS, xR_final: float, max_stages: int = 50) -> dict:
    """Countercurrent extraction with partially miscible phases (Hunter-Nash construction):
    M = F + S; the final raffinate R_N lies on the raffinate branch at solute fraction xR_final; E_1 is the
    extract-branch point collinear with R_N and M; the operating point P = F - E_1 = R_N - S sits on the
    extension of both lines; stages alternate between tie lines (equilibrium) and lines through P (balances)."""
    M, xM = mixing_point(F, xF, S, xS)
    xRN = tern.raffinate(xR_final)
    aE1 = _extract_on_line(tern, xRN, xM)
    if aE1 is None:
        raise ValueError("No extract on the binodal is collinear with the mixing point and the final raffinate.")
    xE1 = tern.extract(aE1)
    E1 = M * lever_split(xM, xE1, xRN)
    RN = M - E1
    xP = (F * np.asarray(xF, float) - E1 * xE1) / (F - E1)
    extracts, raffinates = [xE1], []
    xE = xE1
    for _ in range(max_stages):
        xR = tern.raffinate_in_equilibrium(xE[0])
        if raffinates and abs(xR[0] - raffinates[-1][0]) < 1e-12:
            raise ValueError("The construction stalled: the tie-line data do not cover the required dilute region.")
        raffinates.append(xR)
        if xR[0] <= xR_final + 1e-9:
            break
        aE = _extract_on_line(tern, xR, xP, a_hi=xE[0] - 1e-9)
        if aE is None:  # the operating line no longer reaches the extract branch below the last extract: a pinch
            raise ValueError(
                "The construction pinches: the solvent rate is at or below the minimum for this separation."
            )
        xE = tern.extract(aE)
        extracts.append(xE)
    return {
        "stages": len(raffinates),
        "E1": E1,
        "RN": RN,
        "x_E1": tuple(xE1),
        "x_RN": tuple(xRN),
        "x_M": tuple(xM),
        "x_P": tuple(xP),
        "raffinates": raffinates,
        "extracts": extracts,
    }


def minimum_solvent_ternary(tern: Ternary, F: float, xF, xS, xR_final: float, S_range=None) -> float:
    """Solvent rate at which the construction needs infinite stages: the operating line through P coincides with a
    tie line (found by bisection on the stage count)."""

    def stages(S):
        try:
            return hunter_nash(tern, F, xF, S, xS, xR_final, max_stages=200)["stages"]
        except ValueError:
            return 200

    lo, hi = (0.01 * F, 50.0 * F) if S_range is None else S_range
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if stages(mid) >= 200:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------ leaching / washing with constant underflow
def leaching_countercurrent(
    N: int, solvent_per_solid: float, underflow_per_solid: float, solute_in_feed: float, solvent_in: float = 0.0
) -> dict:
    """Ideal countercurrent leaching with constant underflow (solution retained per unit inert solid): the solute
    balance is a Kremser-type problem with washing factor W = V/L (overflow solution per underflow solution).
    Returns the fraction of solute recovered in the overflow."""
    W = solvent_per_solid / underflow_per_solid
    if abs(W - 1) < 1e-12:
        recovery = N / (N + 1)
    else:
        recovery = (W ** (N + 1) - W) / (W ** (N + 1) - 1)
    return {"washing_factor": W, "recovery": recovery, "solute_lost": solute_in_feed * (1 - recovery)}


def leaching_stages(recovery: float, W: float) -> float:
    if abs(W - 1) < 1e-12:
        return recovery / (1 - recovery)
    return float(np.log((W - recovery) / (1 - recovery)) / np.log(W) - 1)
