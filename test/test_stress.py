from unittest import TestCase


class TestStress(TestCase):
    def test_mohr2d(self):
        from math import pi
        from pyavia.struct import mohr2d
        from pyavia import Dim

        def check_stress_state(result_ss, reqd_ss):
            self.assertAlmostEqual(result_ss[0], reqd_ss[0], places=1)  # s_11
            self.assertAlmostEqual(result_ss[1], reqd_ss[1], places=1)  # s_22
            self.assertAlmostEqual(result_ss[2], reqd_ss[2], places=1)  # t_12
            self.assertAlmostEqual(result_ss[3], reqd_ss[3], places=2)  # Ang

        # Check for correct principal angles at a variety of angles and Dim()
        # inputs.
        prin_str = mohr2d(15000, 5000, 4000)
        check_stress_state(prin_str, [16403.1, 3596.9, 0.0, +19.33 * pi / 180])
        prin_str = mohr2d(15000, 25000, 4000)
        check_stress_state(prin_str, [26403.1, 13596.9, 0.0, +70.67 * pi / 180])
        prin_str = list(mohr2d(Dim(0, 'MPa'), Dim(2000, 'psi'), Dim(-5, 'ksi')))
        prin_str[0:3] = [x.convert('psi').value for x in prin_str[0:3]]
        check_stress_state(prin_str, [6099, -4099, 0.0, -50.66 * pi / 180])

        # Test prescribed angles.
        prin_str = mohr2d(1000, 2000, 3000, Dim(60, '°'))
        check_stress_state(prin_str, [4348.1, -1348.1, -1067.0, +60 * pi /
                                      180])
        prin_str = mohr2d(1000, 2000, 3000, Dim(-90, 'deg'))
        check_stress_state(prin_str, [2000, 1000, -3000, -pi / 2])
