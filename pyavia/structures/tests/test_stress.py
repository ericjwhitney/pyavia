import numpy as np
from pytest import approx

from pyavia.structures import mohr2d

d2r = np.deg2rad


# ======================================================================

def _check_stress_state(result_σ, reqd_σ, σ_tol=0.1, θ_tol=0.01):
    assert result_σ[0] == approx(reqd_σ[0], abs=σ_tol)  # σ_11
    assert result_σ[0] == approx(reqd_σ[0], abs=σ_tol)  # σ_11
    assert result_σ[1] == approx(reqd_σ[1], abs=σ_tol)  # σ_22
    assert result_σ[2] == approx(reqd_σ[2], abs=σ_tol)  # τ_12
    assert result_σ[3] == approx(reqd_σ[3], abs=θ_tol)  # θ


# ----------------------------------------------------------------------

def test_mohr2d():
    # -- Check Principal Stress / Angles Combinations ------------------

    σ_pr = mohr2d(σ_xx=+80, σ_yy=+40, τ_xy=+30)  # +X, +y, +xy.
    _check_stress_state(σ_pr, [96.05, 23.95, 0.0, d2r(28.15)])

    σ_pr = mohr2d(15000, 5000, 4000)  # +X, +y, +xy.
    _check_stress_state(σ_pr, [16403.1, 3596.9, 0.0, d2r(+19.33)])

    σ_pr = mohr2d(15000, 25000, 4000)  # +x, +Y, +xy.
    _check_stress_state(σ_pr, [26403.1, 13596.9, 0.0, d2r(+70.67)])

    σ_pr = mohr2d(0, 2000, -5000)  # +x, +Y, -xy.
    _check_stress_state(σ_pr, [6099, -4099, 0.0, d2r(-50.66)])

    σ_pr = mohr2d(σ_xx=-80, σ_yy=+50, τ_xy=-25)  # -X, +y, -xy.
    _check_stress_state(σ_pr, [54.6, -84.6, 0.0, d2r(-79.5)])

    # -- Check Prescribed Angles ---------------------------------------

    σ_pr = mohr2d(1000, 2000, 3000, d2r(+60))
    _check_stress_state(σ_pr, [4348.1, -1348.1, -1067.0, d2r(+60)])

    σ_pr = mohr2d(1000, 2000, 3000, d2r(-90))
    _check_stress_state(σ_pr, [2000, 1000, -3000, -np.pi / 2])
