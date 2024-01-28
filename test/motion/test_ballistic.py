from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_less


# Written by Eric J. Whitney, October 2023.


# ============================================================================

class TestBallistic(TestCase):
    def test_ballistic_no_drag(self):
        # Note: As function `ballistic` calls `ballistic_variable`, this test
        # automatically tests the latter by definition.

        from pyavia.motion.ballistic import ballistic

        # Test basic dragless ballistic trajectory against classical
        # solutions across a number of velocities and launch angles.  Check
        # both data points and fitted solutions where available.
        for V0 in [0.001, 1000.0]:
            for θ0 in np.deg2rad([0.0, 0.1, 30.0, 45.0, 60.0, 89.9, 90.0]):
                for m in [0.001, 1000.0]:
                    for g in [9.80665, 32.17405]:
                        traj = ballistic(V0, θ0, m, g)

                        # -- Check Initial Position --------------------------

                        u0, w0 = V0 * np.cos(θ0), V0 * np.sin(θ0)

                        prec = 11  # Precision for initial conditions.

                        # TODO: Inexplicable type warning on line below
                        #  should disappear with later versions of NumPy.
                        self.assertAlmostEqual(traj.V[0], V0, places=prec)

                        self.assertAlmostEqual(traj.V_t(0), V0, places=prec)
                        self.assertAlmostEqual(traj.θ[0], θ0, places=prec)
                        self.assertAlmostEqual(traj.θ_t(0), θ0, places=prec)
                        self.assertAlmostEqual(traj.u[0], u0, places=prec)
                        self.assertAlmostEqual(traj.u_t(0), u0, places=prec)
                        self.assertAlmostEqual(traj.w[0], w0, places=prec)
                        self.assertAlmostEqual(traj.w_t(0), w0, places=prec)

                        # -- Check Apogee ------------------------------------

                        t_apogee = w0 / g
                        u_apogee, w_apogee = u0, 0.0
                        z_apogee = w0 ** 2 / (2 * g)

                        prec = 5  # Precision for function results.

                        self.assertAlmostEqual(traj.t_apogee, t_apogee,
                                               places=prec)
                        self.assertAlmostEqual(traj.u_t(traj.t_apogee),
                                               u_apogee, places=prec)
                        self.assertAlmostEqual(traj.w_t(traj.t_apogee),
                                               w_apogee, places=prec)
                        self.assertAlmostEqual(traj.z_t(traj.t_apogee),
                                               z_apogee, places=prec)

                        # -- Check Impact ------------------------------------

                        t_impact = 2 * t_apogee
                        u_impact, w_impact = u0, -w0
                        z_impact = 0.0

                        self.assertAlmostEqual(traj.t_impact, t_impact,
                                               places=prec)
                        self.assertAlmostEqual(traj.u_t(traj.t_impact),
                                               u_impact, places=prec)
                        self.assertAlmostEqual(traj.w_t(traj.t_impact),
                                               w_impact, places=prec)
                        self.assertAlmostEqual(0.0, z_impact, places=prec)

                        # -- Check Z vs X ------------------------------------

                        # Generate theoretical x, z points.
                        t_pts = np.linspace(0.0, t_impact, num=20)
                        x_pts = u0 * t_pts
                        z_pts = w0 * t_pts - 0.5 * g * t_pts ** 2

                        assert_almost_equal(x_pts, traj.x_t(t_pts),
                                            decimal=prec)
                        assert_almost_equal(z_pts, traj.z_t(t_pts),
                                            decimal=prec)

    def test_ballistic_fixed_drag(self):
        # Note: Also tests function 'ballistic_variable' for the same
        # reasons as given in 'test_ballistic_no_drag' above.

        from pyavia.motion.ballistic import ballistic

        # Example from Chudinov (2014), baseball trajectory.
        g = 9.81  # [m/s^2]
        m = 0.1455  # [kg] (assumed).
        ρ = 1.225  # [kg/m^3] (assumed).

        # Rearrange Chudinov drag term (pp 99): k = ρ * f / (2 * m * g)
        f = 0.0014565  # [m^2] or effective diameter 43 mm (1.70 in).

        for V0 in [5.0, 40.0]:  # [m/s]
            for θ0 in np.deg2rad([30.0, 45.0, 60.0]):  # ° -> [rad]

                traj = ballistic(V0, θ0, m, g, ρ, f)
                approx = _traj_chudinov(V0, θ0, m, g, ρ, f)

                # -- Check Apogee --------------------------------------------

                tol = 0.03  # 3% tolerance allowed on overall values.

                self.assertAlmostEqual(traj.t_apogee, approx['t_apogee'],
                                       delta=tol*approx['t_apogee'])
                self.assertAlmostEqual(traj.V_t(traj.t_apogee),
                                       approx['V_apogee'],
                                       delta=tol*approx['V_apogee'])
                self.assertAlmostEqual(traj.z_t(traj.t_apogee),
                                       approx['z_apogee'],
                                       delta=tol*approx['z_apogee'])

                # -- Check Impact --------------------------------------------

                self.assertAlmostEqual(traj.t_impact, approx['t_impact'],
                                       delta=tol*approx['t_impact'])
                self.assertAlmostEqual(traj.V_t(traj.t_impact),
                                       approx['V_impact'],
                                       delta=tol*approx['V_impact'])
                self.assertAlmostEqual(traj.x_t(traj.t_impact),
                                       approx['x_impact'],
                                       delta=tol * approx['x_impact'])

                # -- Check Z vs X --------------------------------------------

                # Generate trajectory x, z points.
                t_traj = np.linspace(0.0, traj.t_impact, num=20)
                x_traj = traj.x_t(t_traj)
                z_traj = traj.z_t(t_traj)

                # Generate approximate z points and compare relative
                # difference in terms of apogee.
                z_approx = approx['z_x_fn'](x_traj)
                z_err = np.abs((z_traj - z_approx) / traj.z_t(traj.t_apogee))

                tol = 0.075  # 7.5% tolerance allowed on heights.
                assert_array_less(z_err, tol)

    def test_ballistic_variable_drag(self):
        # Future Work: Add any special test cases as required if bespoke /
        # variable drag functions using 'θ' and 'z' variables need checking.
        # Basic functionality of 'V' parameter is already checked in
        # 'test_ballistic_fixed_drag'.
        pass


# ----------------------------------------------------------------------------


def _traj_chudinov(V0: float, θ0: float, m: float, g: float, ρ: float,
                   f: float) -> {str: float}:
    # For checking fixed drag solution, this function gives approximate
    # analytic equations for fixed drag motion from  Chudinov, P.,
    # 'Approximate Analytical Description of the Projectile Motion with a
    # Quadratic Drag Force', Athens Journal of Sciences - Volume 1,
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

    return {'t_apogee': t_a, 'V_apogee': Va, 'x_apogee': x_a, 'z_apogee': H,
            't_impact': T, 'V_impact': V1, 'x_impact': L,
            'z_x_fn': y}
