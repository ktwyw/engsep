import numpy as np
import pytest

from engsep import distillation as dist
from engsep import flash, vle
from engsep.vle import y_from_alpha


def test_vle():
    assert vle.boiling_point("benzene") == pytest.approx(353.2, abs=0.5)
    assert vle.has_wilson(("water", "ethanol")) and not vle.has_wilson(("benzene", "n-hexane"))
    g1, g2 = vle.gammas(("benzene", "n-hexane"), np.array([0.3, 0.7]), 340.0)
    assert np.all(g1 == 1) and np.all(g2 == 1)
    L12, L21 = vle.wilson_lambda(("water", "ethanol"), 350.0)  # reversed pair order handled
    L12b, L21b = vle.wilson_lambda(("ethanol", "water"), 350.0)
    assert L12 == pytest.approx(L21b) and L21 == pytest.approx(L12b)
    assert vle.azeotrope("benzene", "toluene") is None
    assert vle.k_value("water", 373.15, 101325.0) == pytest.approx(1.0, abs=0.01)
    d = vle.xy(("benzene", "toluene"), n=5, ideal=True)
    assert d["y"][0] == 0 and d["y"][-1] == pytest.approx(1.0)
    assert vle.relative_volatility(("benzene", "toluene"), 360.0, ideal=True) > 2


def test_flash():
    rr = flash.rachford_rice([0.2, 0.3, 0.5], [3.0, 1.2, 0.2])
    assert rr["state"] == "two-phase" and np.sum(rr["y"]) == pytest.approx(1.0)
    assert flash.bubble_point_T(["water"], [1.0], 101325.0) == pytest.approx(
        373.15, abs=0.1
    )  # pure component: its boiling point
    assert (
        flash.adiabatic_flash(["n-pentane", "n-hexane"], [0.5, 0.5], 300.0, 101325.0, [167.0, 195.0], [25.8e3, 28.9e3])[
            "V_over_F"
        ]
        == 0.0
    )


def test_distillation():
    y_eq = lambda x: y_from_alpha(x, 3.0)  # noqa: E731
    mt = dist.mccabe_thiele(y_eq, 0.9, 0.1, 0.5, R=1.5, q=0.5)
    assert mt["stages"] >= 3 and mt["feed_stage"] <= mt["stages"] and mt["lines"]["q_line"] is not None
    assert dist.operating_lines(0.9, 0.1, 0.5, 1.5, 1.0)["q_line"] is None
    uw = dist.underwood([4.0, 2.0, 1.0], [0.3, 0.4, 0.3], [0.7, 0.29, 0.01], q=1.0)
    assert len(uw["theta"]) == 2 and uw["R_min"] > 0
    assert dist.packed_height(10, 0.5) == 5.0 and dist.overall_efficiency(10, 20) == 0.5
    assert dist.oconnell_efficiency(1.0, 0.001) == 1.0
    assert dist.fenske(0.99, 0.01, 2.0, xD_hk=0.01, xB_hk=0.99) == pytest.approx(2 * np.log(99) / np.log(2))


def test_package_exposes_every_module():
    import pkgutil

    import engsep

    modules = sorted(
        m.name for m in pkgutil.iter_modules(engsep.__path__) if not m.name.startswith("_") and m.name != "data"
    )
    assert all(hasattr(engsep, m) for m in modules), [m for m in modules if not hasattr(engsep, m)]
    assert set(modules) <= set(engsep.__all__)
