from unittest import TestCase


class TestStress(TestCase):
    def test_mohr2d(self):
        import numpy as np
        from pyavia.structure import mohr2d
        d2r = np.deg2rad

        def check_stress_state(result_σ, reqd_σ):
            self.assertAlmostEqual(result_σ[0], reqd_σ[0], places=1)  # σ_11
            self.assertAlmostEqual(result_σ[1], reqd_σ[1], places=1)  # σ_22
            self.assertAlmostEqual(result_σ[2], reqd_σ[2], places=1)  # τ_12
            self.assertAlmostEqual(result_σ[3], reqd_σ[3], places=2)  # θ

        # -- Check Principal Stress / Angles Combinations --------------------

        σ_pr = mohr2d(σ_xx=+80, σ_yy=+40, τ_xy=+30)  # +X, +y, +xy.
        check_stress_state(σ_pr, [96.05, 23.95, 0.0, d2r(28.15)])

        σ_pr = mohr2d(15000, 5000, 4000)  # +X, +y, +xy.
        check_stress_state(σ_pr, [16403.1, 3596.9, 0.0, d2r(+19.33)])

        σ_pr = mohr2d(15000, 25000, 4000)  # +x, +Y, +xy.
        check_stress_state(σ_pr, [26403.1, 13596.9, 0.0, d2r(+70.67)])

        σ_pr = mohr2d(0, 2000, -5000)  # +x, +Y, -xy.
        check_stress_state(σ_pr, [6099, -4099, 0.0, d2r(-50.66)])

        σ_pr = mohr2d(σ_xx=-80, σ_yy=+50, τ_xy=-25)  # -X, +y, -xy.
        check_stress_state(σ_pr, [54.6, -84.6, 0.0, d2r(-79.5)])

        # -- Check Prescribed Angles -----------------------------------------

        σ_pr = mohr2d(1000, 2000, 3000, d2r(+60))
        check_stress_state(σ_pr, [4348.1, -1348.1, -1067.0, d2r(+60)])

        σ_pr = mohr2d(1000, 2000, 3000, d2r(-90))
        check_stress_state(σ_pr, [2000, 1000, -3000, -np.pi / 2])
