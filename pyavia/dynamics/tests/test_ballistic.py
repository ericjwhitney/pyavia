from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pyavia.dynamics import ballistic


# Written by Eric J. Whitney, October 2023.


# ======================================================================


def _make_no_drag_tests() -> list[tuple[float, ...]]:
    """"
    Test conditions for checking against exact solutions (no drag).
    """
    test_conds = []

    for V0 in [0.001, 1000.0]:
        for θ0 in np.deg2rad([0.0, 0.1, 30.0, 45.0, 60.0, 89.9, 90.0]):
            for m in [0.001, 1000.0]:
                for g in [9.80665, 32.17405]:
                    test_conds.append((V0, θ0, m, g))

    return test_conds


_no_drag_test_conds = _make_no_drag_tests()


# ----------------------------------------------------------------------


@pytest.mark.parametrize("V0, θ0, m, g", _no_drag_test_conds)
def test_ballistic_no_drag(V0: float, θ0: float, m: float, g: float):
    """
    Test basic dragless ballistic trajectory against classical
    solutions across a number of velocities and launch angles.  Check
    both data points and fitted solutions where available.

    .. note::As function `ballistic` calls `ballistic_variable`, this
       test automatically tests the latter by definition.
    """
    traj = ballistic(V0, θ0, m, g)

    # -- Check Initial Position ----------------------------------------

    approx = partial(pytest.approx, abs=1e-10)  # For init. values.
    u0, w0 = V0 * np.cos(θ0), V0 * np.sin(θ0)

    assert traj.V[0] == approx(V0)
    assert traj.V_t(0) == approx(V0)
    assert traj.θ[0] == approx(θ0)
    assert traj.θ_t(0) == approx(θ0)
    assert traj.u[0] == approx(u0)
    assert traj.u_t(0) == approx(u0)
    assert traj.w[0] == approx(w0)
    assert traj.w_t(0) == approx(w0)

    # -- Check Apogee --------------------------------------------------

    abs_prec = 1e-5  # Precision for function results.
    rel_prec = 1e-5
    approx = partial(pytest.approx, abs=abs_prec, rel=rel_prec)

    t_apogee = w0 / g
    u_apogee, w_apogee = u0, 0.0
    z_apogee = w0 ** 2 / (2 * g)

    assert traj.t_apogee == approx(t_apogee)
    assert traj.u_t(traj.t_apogee) == approx(u_apogee)
    assert traj.w_t(traj.t_apogee) == approx(w_apogee)
    assert traj.z_t(traj.t_apogee) == approx(z_apogee)

    # -- Check Impact --------------------------------------------------

    t_impact = 2 * t_apogee
    u_impact, w_impact = u0, -w0
    z_impact = 0.0

    assert traj.t_impact == approx(t_impact)
    assert traj.u_t(traj.t_impact) == approx(u_impact)
    assert traj.w_t(traj.t_impact) == approx(w_impact)
    assert 0.0 == approx(z_impact)

    # -- Check Z vs X --------------------------------------------------

    # Generate theoretical x, z points.
    t_pts = np.linspace(0.0, t_impact, num=20)
    x_pts = u0 * t_pts
    z_pts = w0 * t_pts - 0.5 * g * t_pts ** 2

    assert_allclose(x_pts, traj.x_t(t_pts), atol=abs_prec)
    assert_allclose(z_pts, traj.z_t(t_pts), atol=abs_prec)


# ----------------------------------------------------------------------


def _make_fixed_drag_tests() -> list[tuple[float, ...]]:
    """
    Test conditions for checking against approximate solutions (drag).
    Example from Chudinov (2014), baseball trajectory.
    """
    g = 9.81  # [m/s^2]
    m = 0.1455  # [kg] (assumed).
    ρ = 1.225  # [kg/m^3] (assumed).

    # Rearrange Chudinov drag term (pp 99): k = ρ * f / (2 * m * g)
    f = 0.0014565  # [m^2] or effective diameter 43 mm (1.70 in).

    test_conds = []
    for V0 in [5.0, 40.0]:  # [m/s]
        for θ0 in np.deg2rad([30.0, 45.0, 60.0]):  # ° -> [rad]
            test_conds.append((V0, θ0, m, g, ρ, f))

    return test_conds


_fixed_drag_test_conds = _make_fixed_drag_tests()


# ----------------------------------------------------------------------


@pytest.mark.parametrize("V0, θ0, m, g, ρ, f", _fixed_drag_test_conds)
def test_ballistic_fixed_drag(V0: float, θ0: float, m: float, g: float,
                              ρ: float, f: float):
    """
    As an approximate closed-form solution is used for comparison to the
    calculated trajectory.  Because of this tolerances are relatively
    loose.  Checks are designed to ensure result matches within the
    precision of the method and has the correct behaviour, sign, etc.

    .. note::Also tests function 'ballistic_variable' for the same
       reasons as given in 'test_ballistic_no_drag' above.
    """
    calc = ballistic(V0, θ0, m, g, ρ, f)
    chud = _traj_chudinov(V0, θ0, m, g, ρ, f)

    # -- Check Apogee --------------------------------------------------

    # Allow tolerances of :
    #   - ±0.`05` s and ±0.5% on times.
    #   - ±0.05 m and ±5% on distance and velocity.
    approx_t = partial(pytest.approx, abs=0.1, rel=0.005)
    approx_ux = partial(pytest.approx, abs=0.05, rel=0.05)

    assert calc.t_apogee == approx_t(chud['t_apogee'])
    assert calc.V_t(calc.t_apogee) == approx_ux(chud['V_apogee'])
    assert calc.z_t(calc.t_apogee) == approx_ux(chud['z_apogee'])

    # -- Check Impact --------------------------------------------------

    assert calc.t_impact == approx_t(chud['t_impact'])
    assert calc.V_t(calc.t_impact) == approx_ux(chud['V_impact'])
    assert calc.x_t(calc.t_impact) == approx_ux(chud['x_impact'])

    # -- Check Trajectory Errors ---------------------------------------

    # Trajectories are considered to be parameteric curves generated by
    # 't'.  We compare the distance between points on the calculated
    # trajectory to the corresponding point on the reference trajectory.
    # This is OK because the overall apogee time, impact time and impact
    # distance have already been checked.  Because of the limitations of
    # the approximate method we allow ±0.075*V0 m tolerance on the
    # height.

    # Generate trajectory x, z points.
    t_calc = np.linspace(0.0, calc.t_impact, num=15)
    x_calc = calc.x_t(t_calc)
    z_calc = calc.z_t(t_calc)
    z_chud = chud['z_x_fn'](x_calc)
    assert_allclose(z_calc, z_chud, atol=V0 * 0.075, rtol=0.0)


# ----------------------------------------------------------------------

def test_ballistic_variable_drag():
    # Future Work: Add any special test cases as required if bespoke /
    # variable drag functions using 'θ' and 'z' variables need checking.
    # Basic functionality of 'V' parameter is already checked in
    # 'test_ballistic_fixed_drag'.
    pass


# ======================================================================

def _traj_chudinov(V0: float, θ0: float, m: float, g: float, ρ: float,
                   f: float) -> {str: float}:
    # For checking fixed drag solution, this function gives approximate
    # analytic equations for fixed drag motion from  Chudinov, P.,
    # 'Approximate Analytical Description of the Projectile Motion with
    # a Quadratic Drag Force', Athens Journal of Sciences - Volume 1,
    # Issue 2 – Pages 97-106, 2014.

    k = ρ * f / (2 * m * g)  # Chudinov drag term, pp 99.
    p = k * V0 ** 2  # Dimensionless check term.

    if p < 0 or p > 4:
        raise RuntimeError("Dimensionless parameter 'p' out of range for "
                           "approx. method.")
    if V0 < 0 or V0 > 80:
        raise RuntimeError("'V0' out of range for approx. method.")

    if θ0 < 0 or θ0 > np.deg2rad(90):
        raise RuntimeError("'θ0' out of range for approx. method.")

    # Chudinov Table 1.
    H = ((V0 ** 2 * np.sin(θ0) ** 2) /  # Apogee.
         (g * (2 + k * V0 ** 2 * np.sin(θ0))))
    T = 2 * np.sqrt(2 * H / g)  # Flight time.

    # Chudinov f(θ) equation.
    def fθ(θ: float) -> float:
        return np.sin(θ) / np.cos(θ) ** 2 + np.log(np.tan(0.5 * θ +
                                                          0.25 * np.pi))

    # Chudinov V(θ) equation as closure.
    def V(θ):
        return (V0 * np.cos(θ0) /
                (np.cos(θ) * np.sqrt(1 + k * V0 ** 2 * np.cos(θ0) ** 2 *
                                     (fθ(θ0) - fθ(θ)))))

    Va = V(0.0)  # Apogee velocity.
    L = Va * T  # Impact distance.
    t_a = (T - k * H * Va) / 2  # Apogee time.
    x_a = np.sqrt(L * H / np.tan(θ0))  # Apogee downrange distance.
    θ1 = -np.arctan(L * H / (L - x_a) ** 2)  # Impact angle.
    V1 = V(θ1)  # Impact velocity.

    # Chudinov y(x) equation - as closure.
    def y(x):
        return H * x * (L - x) / (x_a ** 2 + (L - 2 * x_a) * x)

    return {'t_apogee': t_a, 'V_apogee': Va,
            'x_apogee': x_a, 'z_apogee': H,
            't_impact': T, 'V_impact': V1, 'x_impact': L,
            'z_x_fn': y}

# ----------------------------------------------------------------------
