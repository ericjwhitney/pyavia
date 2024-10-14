from functools import partial

import numpy as np
import pytest

from pyavia.dynamics.landing import landing_energy

# ======================================================================

_tyre_ft_lbf = np.array([
    # Tyre:      Goodyear 5.00-5, Type III, 4 Ply Nylon
    # Loading:   Dynamic
    # Inflation: 30 psig
    [0.000, 0],  # [in] (initially), [lbf]
    [0.535, 270],
    [0.910, 530],
    [1.180, 740],
    [1.300, 840],
    [1.460, 980],
    [1.650, 1160],
    [1.840, 1345],
    [2.020, 1525],
    [2.210, 1725],
    [2.455, 2000],
    [2.670, 2250],
    [2.825, 2500],
    [2.895, 2700]
])
_tyre_ft_lbf[:, 0] /= 12  # in -> [ft]
_tyre_m_slug = 5.0 * 0.031081  # lbm -> [slug]

_leaf_std_kwargs = {
    'strut': 2158.3,  # Strut spring K [lbf/ft]
    'tyre': _tyre_ft_lbf,  # Tabulated (ft, lbf)
    'g': 32.174,  # [ft/s²]
    'K': 2 / 3,  # L/W for light aircraft
    'm_ac': 1200 * 0.031081 / 2,  # lbm -> [slug]
    'V_sink': 7.0  # [fps]
}

# Test with no sprung mass and also with 5 lbm tyre.
# Tests are labelled by tyre mass.
_leaf_ref_vals = {
    0.0: {  # Sprung mass [slug].
        'δ_s': 0.695,  # Strut stroke [ft].
        'E_s': 521.6,  # Strut energy [ft-lbf].
        'F_s': 1500.5,  # Strut force [lbf].
    },
    _tyre_m_slug: {  # Sprung mass [slug].
        'δ_s': 0.693,  # Strut stroke [ft].
        'E_s': 517.7,  # Strut energy [ft-lbf].
        'F_s': 1494.9,  # Strut force [lbf].
    }
}


# ----------------------------------------------------------------------

# Test of a basic linear spring and tyre system.  This is done with no
# sprung mass and also with 5 lbm tyre -> [slug].  This test covers all
# the general aspects and basic properties of the function.
@pytest.mark.parametrize('m_t', list(_leaf_ref_vals.keys()))
def test_leaf_strut_tyre(m_t: float):
    results = landing_energy(**_leaf_std_kwargs, m_t=m_t)

    # Test tolerances.
    almost_equal = partial(pytest.approx, abs=0.001)
    approx = partial(pytest.approx, abs=0.001, rel=0.001)

    # -- Check Constants -----------------------------------------------

    assert results.W_ac == almost_equal(600)
    assert results.L == almost_equal(400)
    assert results.V_sink == almost_equal(7)
    assert results.g == almost_equal(32.174)
    assert results.K == almost_equal(2 / 3)

    # -- Tyre ----------------------------------------------------------

    # Common results.
    assert results.δ_t == approx(0.166)
    assert results.E_t == approx(107.5)
    assert results.F_t == approx(1499.9)
    assert results.η_t == approx(0.43107)

    # -- Strut ---------------------------------------------------------

    # Common results.
    assert results.η_s == almost_equal(0.5)  # Pure spring.

    # Per-test results.
    assert results.δ_s == approx(_leaf_ref_vals[m_t]['δ_s'])
    assert results.E_s == approx(_leaf_ref_vals[m_t]['E_s'])
    assert results.F_s == approx(_leaf_ref_vals[m_t]['F_s'])


# ----------------------------------------------------------------------

# Test an example oleo-pneumatic strut to check basic trends for oleos
# are correct:
#  - The static deflection curve is from Currey, "Aircraft Landing Gear
#    Design - Principles and Practices", Fig 5.31.  The
#    narrow preload region of 0.040" is added to be representative and
#    checks for hiccups due to rapid slope changes.
#  - The oil damping effect is added as a constant throughout the stroke
#    based on the sink speed (this is not very accurate but only for
#    testing the function).
#  - The tyre is assumed to be much stiffer (i.e. infinitely stiff).
#    The aircraft mass is 23,844 lbm.

_oleo_static_ft_lbf = np.array([
    [0.000, 0],  # [in (initially), lbf]
    [0.020, 6541],  # <- Preload.
    [0.889, 6778],
    [1.965, 7131],
    [2.951, 7827],
    [3.937, 8639],
    [4.924, 9565],
    [5.933, 10606],
    [7.012, 11877],
    [7.954, 13147],
    [8.988, 14762],
    [9.978, 16951],
    [10.992, 19484],
    [11.966, 23852],
    [13.083, 30402],
    [14.048, 39705],
    [14.530, 47972],
    [14.895, 55319],
    [15.093, 60484],
    [15.223, 65764],
    [15.379, 72077],
    [15.521, 81717],
])
_oleo_static_ft_lbf[:, 0] /= 12  # in -> [ft]

_oleo_damping = 1500  # lbf.s/ft

_oleo_std_kwargs = {
    'tyre': 1e10,  # Tyre stiffness (≈infinite) [lbf/ft]
    'g': 32.174,  # [ft/s²]
    'K': 1,  # L/W (large aircraft).
    'm_ac': 23844 * 0.031081 / 2,  # Per strut lbm -> [slug]
    'm_t': 0.0,  # Massless.
    # 'V_sink': 10.0  # [fps]
}


# ----------------------------------------------------------------------

# Test a range of different sink speeds up to 10 fps.
@pytest.mark.parametrize('V_sink', [1, 2, 5, 10])
def test_example_oleo(V_sink: float):
    # Produce a 'corrected' static curve including damping (approx).
    oleo_ft_lbf = _oleo_static_ft_lbf + np.array([
        0, V_sink * _oleo_damping])

    results = landing_energy(strut=oleo_ft_lbf, V_sink=V_sink,
                             **_oleo_std_kwargs)

    # Test tolerances.
    almost_equal = partial(pytest.approx, abs=0.005)

    # -- Check Constants -----------------------------------------------

    assert results.W_ac == almost_equal(23844 / 2)
    assert results.L == almost_equal(23844 / 2)

    # -- Strut ---------------------------------------------------------

    # Check dynamic load factor within expected range (no spikes).
    assert 0.1 < results.N_s < 10.0

    # Efficiency should at least be over 60% (typically).
    assert 0.6 < results.η_s < 1.0

    # -- Tyre ----------------------------------------------------------

    # Infinitely stiff tyre should be η=50% with negligible deflection.
    assert results.η_t == almost_equal(0.5)
    assert results.δ_t / results.δ_s < 1e-3
