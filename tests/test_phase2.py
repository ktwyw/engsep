import numpy as np
import pytest

from engsep import absorption as ab
from engsep import adsorption as ad
from engsep import crystallization as cr
from engsep import drying_evaporation as de
from engsep import extraction as ex
from engsep import membranes as mb
from engsep import transport as tr


def test_absorption():
    assert ab.absorption_factor(2.0, 1.0, 0.5) == 4.0 and ab.stripping_factor(2.0, 1.0, 0.5) == 0.25
    assert ab.kremser_stages(1.0, 0.8) == pytest.approx(4.0)
    g = ab.kremser_general(1.5, 4, 0.02, 0.001, 0.8)
    assert 0 < g["y_out"] < 0.02
    assert ab.htu_gas(10.0, 50.0) == 0.2 and ab.packed_height(0.5, 6.0) == 3.0 and ab.henry_slope(1e5, 2e5) == 0.5
    assert ab.stripping_tower_ntu(0.02, 0.002, 1.25, 0.0, 1.0) == pytest.approx(9.0)


def test_extraction():
    assert ex.extraction_factor(2.0, 0.5) == 1.0
    assert ex.countercurrent_immiscible(2.0, 0.5, 0.1, 3)["fraction_extracted"] == pytest.approx(0.75)  # E = 1: N/(N+1)
    assert ex.countercurrent_stages_immiscible(2.0, 0.5, 0.1, 0.025) == pytest.approx(3.0)
    assert ex.leaching_countercurrent(2, 1.0, 1.0, 10.0)["recovery"] == pytest.approx(2 / 3)
    assert ex.leaching_stages(2 / 3, 1.0) == pytest.approx(2.0)
    M, xM = ex.mixing_point(1.0, (0.4, 0.0), 1.0, (0.0, 1.0))
    assert M == 2.0 and tuple(xM) == (0.2, 0.5)
    assert ex.lever_split((0.2, 0.5), (0.4, 0.0), (0.0, 1.0)) == pytest.approx(0.5)


def test_adsorption_and_membranes():
    assert ad.langmuir(1e9, 4.0, 0.8) == pytest.approx(4.0, rel=1e-6)
    assert ad.psa_cycle(2.0, 0.5, 500.0, 1.0, 0.2, 10.0)["working_capacity_mol"] == 750.0
    assert ad.psa_cycle(2.0, 0.5, 500.0, 1.0, 0.2, 0.0)["feed_volume_per_cycle"] == np.inf
    assert mb.permeance(1e-12, 1e-6) == 1e-6 and mb.selectivity(4.0, 1.0) == 4.0
    assert mb.gas_flux(1e-6, 10e5, 0.5, 1e5, 0.8) == pytest.approx(1e-6 * (5e5 - 0.8e5))
    assert mb.membrane_area(1.0, 0.2, 0.6, 1e-6, 10e5, 0.4, 1e5) > 0
    assert mb.ro_recovery(0.6, 10e5, 298.15) == 0.0  # pressure below osmotic: no recovery
    assert mb.dialysis_flux(1e-5, 2.0, 1.0) == 1e-5
    assert mb.dialysis_countercurrent(1e-5, 1.0, 1e-5, 1e-5, 1.0)["effectiveness"] == pytest.approx(0.5)


def test_crystallization_drying_transport():
    assert cr.solubility("NaCl", 50.0) == pytest.approx(0.5 * (36.6 + 37.3))
    assert cr.cooling_yield("NaCl", 100.0, 60.0, 60.0)["crystals_kg"] == 0.0
    assert cr.supersaturation(1.2, 1.0)["relative"] == pytest.approx(0.2)
    assert cr.growth_rate(1e-6, 0.5, 1.0) == 5e-7 and cr.nucleation_rate(1e3, 0.5, 2.0) == 250.0
    assert de.drying_time(0.8, 0.5, 0.3, 0.05, 1e-3, 0.5)["t_falling"] == 0.0
    assert de.constant_rate(50.0, 60.0, 30.0) == pytest.approx(50.0 * 30.0 / de.latent_heat(30.0))
    assert de.enthalpy_moist_air(0.0, 0.0) == 0.0 and de.humid_heat(0.0) == de.CP_AIR
    assert tr.film_coefficient(1e-9, 1e-4) == 1e-5 and tr.sherwood_flat_plate_laminar(1e4, 1.0) == pytest.approx(66.4)
    assert tr.sherwood_pipe_turbulent(1e4, 1.0) == pytest.approx(0.023 * 1e4**0.83)
