from unittest import TestCase


# noinspection PyUnusedLocal
class TestDim(TestCase):
    def test___init__(self):
        from pyavia.core.units import dim

        # Test no arguments:  Value = 1, no units.
        x = dim()
        self.assertEqual(x.value, 1)
        self.assertIsInstance(x.value, int)

        # Test one positional argument: Dimensionless.
        x = dim(3)
        self.assertEqual(x.value, 3)
        self.assertIsInstance(x.value, int)
        self.assertEqual(x.units, '')

        # Test one positional argument: Units only.
        x = dim('ft')
        self.assertEqual(x.value, 1)
        self.assertIsInstance(x.value, int)
        self.assertEqual(x.units, 'ft')

        # Test two positional arguments: Value and units given.
        x, y = dim(3, 'ft'), dim(5 + 6j, 's')
        self.assertEqual(x.value, 3)
        self.assertIsInstance(x.value, int)
        self.assertEqual(x.units, 'ft')
        self.assertEqual(y.value, 5 + 6j)
        self.assertIsInstance(y.value, complex)

        # Test cannot use derived units with offset temperature.
        with self.assertRaises(ValueError):
            x = dim(1, '°C/day')

    def test___add__(self):
        # Also tests __radd__
        from pyavia.core.units import dim

        # Test addition of like units, preserving character.
        x, y = dim('ft'), dim(12, 'in')
        x_p_y = x + y
        self.assertEqual(x_p_y.value, 2)
        self.assertIsInstance(x_p_y.value, int)
        self.assertEqual(x_p_y.units, 'ft')

        x = 5 + dim(1, '')  # __radd__ and unit cancellation check.
        self.assertIsInstance(x, int)
        self.assertEqual(x, 6)

        # Test addition of imcompatible units disallowed.
        with self.assertRaises(ValueError):
            x = dim(10, 'kg') + dim(5, 'm')

        with self.assertRaises(ValueError):
            x = 2 + dim(2, 'kg')  # __radd__ check.

        # Test total temperature addition (special case).
        x = dim(25, '°C') + dim(5, 'Δ°C')
        self.assertEqual(x.units, '°C')
        self.assertEqual(x.value, 30)
        with self.assertRaises(ValueError):
            x = dim(32, '°C') + dim(32, '°F')  # Not allowed.

    def test___sub__(self):
        # Also tests __rsub__
        from pyavia.core.units import dim

        # Test subtraction of imcompatible units disallowed.
        with self.assertRaises(ValueError):
            x = (1 * dim('mol')) - (2 * dim('kg'))

        with self.assertRaises(ValueError):
            x = 3 - dim(2, 'mmol')  # __rsub__ check

        # Test offset temperature subtration.
        x = dim(25, '°C') - dim(5, 'Δ°C')
        self.assertEqual(x.units, '°C')
        self.assertEqual(x.value, 20)

        x = dim(32, '°F') - dim(40, 'K')
        self.assertAlmostEqual(x.value, -40)
        self.assertEqual(x.units, '°F')

        x = dim(32, '°F') - dim(0, '°C')
        self.assertAlmostEqual(x.value, 0)
        self.assertEqual(x.units, 'Δ°F')

    def test___mul__(self):
        from pyavia.core.units import dim

        # Check multiply by plain units gives Dim result.
        x = 7.5 * dim('km')
        self.assertEqual(x.value, 7.5)
        self.assertEqual(x.units, 'km')

        # Check value type follows multiply rules.
        x = 4 * 3 * dim('N')
        self.assertEqual(x.value, 12)
        self.assertIsInstance(x.value, int)

        x = dim(4, 'm') * dim(2.0, 'm')
        self.assertEqual(x.value, 8.0)
        self.assertIsInstance(x.value, float)

        # Check multiply gives correct final units from LHS.
        x = dim(1, 'kg') * dim('G')
        self.assertAlmostEqual(x.value, 9.80665, places=5)
        self.assertEqual(x.units, 'N')

        x = dim(10, 'kg') / dim(5, 'mol')
        self.assertAlmostEqual(x.value, 2)
        self.assertEqual(x.units, 'kg.mol⁻¹')

        # Check degrees are retained.
        x = dim(5, 'kg') * dim('°^-1')
        self.assertEqual(x.units, 'kg.°⁻¹')

        x = dim(10, 'm') * dim('deg')
        self.assertEqual(x.units, 'm.deg')

        # Check radians / steradians disappear automatically when
        # multiplied by other units.
        r, omega = dim(10, 'ft'), dim(30, 'rad/s')
        v_t = r * omega
        self.assertEqual(v_t.value, 300)
        self.assertIsInstance(v_t.value, int)
        self.assertEqual(v_t.units, 'fps')

        # RHS multiplication will cancel 'dimensionless' results.
        x = 1 * dim('sr')
        self.assertEqual(x, 1)
        self.assertIsInstance(x, int)

        # Check LHS multiplication by a scalar skips simiplification and
        # retains 'dimensionless' results.  This allows constants and
        # similar objects to retain preferred units.
        x = dim('rad') * 2
        self.assertEqual(x.value, 2)
        self.assertIsInstance(x.value, int)
        self.assertEqual(x.units, 'rad')

        # Check multiply giving no units results in plain value.
        x = 5.0 * dim()
        self.assertIsInstance(x, float)
        self.assertEqual(x, 5)

        x = dim(6, 'km') * dim(2, 'km^-1')
        self.assertIsInstance(x, int)
        self.assertEqual(x, 12)

    def test___truediv__(self):
        from pyavia.core.units import dim

        # Check value type follows division rules.
        x = dim(4, 'm') / 2
        self.assertEqual(x.value, 2.0)
        self.assertIsInstance(x.value, float)
        self.assertEqual(x.units, 'm')

        # Check division gives correct final units from LHS.
        x = dim(9.80665, 'N') / dim(1, 'G').convert('ft/s^2')
        self.assertAlmostEqual(x.value, 1, places=5)
        self.assertEqual(x.units, 'kg')

    def test___matmul__(self):
        from pyavia.core.units import dim
        import numpy as np

        # Test multiplication of Dim objects holding arrays.
        a = dim(np.array([[1, -1, 2], [0, -3, 1]]), 'J/K')
        # b = dim(np.array(
        b = dim(np.array([-271.15, -272.15, -273.15]), '°C')  # 2 K, 1 K, 0 K.
        x = a @ b
        self.assertTrue((x.value == np.array([1, -3])).all())
        self.assertEqual(x.units, 'J')

    def test_change_basis(self):
        from pyavia.core.units import dim, STD_UNIT_SYSTEM

        STD_UNIT_SYSTEM = 'kg.m.s'  # Abbreviated, allows check on defaults.
        x = dim(1000, 'psi')
        y = x.to_real_sys()  # Result should be in Pa.
        self.assertAlmostEqual(y, 6894757.2932, places=3)

    def test_convert_operations(self):
        from pyavia.core.units import dim

        # Test conversion to lhs does nothing.
        x = dim(3, 'kg').convert('kg')
        self.assertEqual(x.value, 3)
        self.assertEqual(x.units, 'kg')

        # Test integer conserved during whole-number conversion.
        x = dim(144, 'in^2').convert('ft^2')
        self.assertEqual(x.value, 1)
        self.assertIsInstance(x.value, int)

        # Test float conserved when integer is possible.
        x = dim(1.0, 'mol').convert('mmol')
        self.assertIsInstance(x.value, float)

        # Test type promotion for non-whole-number conversion of int.
        x = dim(1, 'in').convert('mm')
        self.assertEqual(x.value, 25.4)
        self.assertIsInstance(x.value, float)

        # Test multi-step conversion with large jump avoids warning message
        # (requires shortcut definitions to be in units.py).
        x = dim(1, 'nmol').convert('Gmol')  # Should give no warning.

        # Test unit multipliers are carried correctly.
        x = dim(1, 'US_gal').convert('in^3')
        self.assertEqual(x.value, 231)

    def test_convert_fundamental(self):
        from pyavia.core.units import dim

        # Test current.
        x = dim(1, 'A').convert('mA')
        self.assertEqual(x.value, 1000)
        self.assertEqual(x.units, 'mA')

        # Test angle units.
        x = (5 * dim('m/°')).convert('m/rad')
        self.assertAlmostEqual(x.value, 286.47889, places=4)

    def test_convert_derived(self):
        from pyavia.core.units import dim

        # Test acceleration.
        x = (9.80665 * dim('m.s⁻²')).convert('ft/s/s')
        self.assertAlmostEqual(x.value, 32.17404856, places=8)

        # Test pressures (a common derived unit with many forms).
        x = dim(1, 'atm')
        mm_hg = x.convert('mmHg').value
        kpa = x.convert('kPa').value
        bar = x.convert('bar').value
        in_hg = x.convert('inHg').value
        psi = x.convert('psi').value
        psf = x.convert('psf').value
        self.assertAlmostEqual(mm_hg, 760, places=3)
        self.assertAlmostEqual(kpa, 101.325)
        self.assertAlmostEqual(bar, 1.01325)
        self.assertAlmostEqual(in_hg, 29.921, places=3)
        self.assertAlmostEqual(psi, 14.696, places=3)
        self.assertAlmostEqual(psf, 2116.22, places=2)

        # Test speed.
        rpm = dim(2500, 'RPM')  # Rotational speed.
        omega = rpm.convert('rad/s')
        self.assertAlmostEqual(omega.value, 261.799388, places=5)
        self.assertEqual(omega.units, 'rad/s')  # Radians should be here.

        v_t = dim(3, 'ft') * omega  # Radians should fall off here.
        self.assertAlmostEqual(v_t.value, 785.398163, places=5)
        self.assertEqual(v_t.units, 'fps')

        # Test power.
        x = dim(100, 'hp')
        x = x.convert('kW')
        self.assertAlmostEqual(x.value, 74.56999, places=3)

        # Test some unusual units.
        k_ic_metric = dim(51.3, 'MPa.m⁰ᐧ⁵')  # Fract. toughness 7039-T6351.
        k_ic_imp = k_ic_metric.convert('ksi.in^0.5')
        self.assertAlmostEqual(k_ic_imp.value, 46.7, places=1)
        cal_energy = dim(1, 'cal')
        self.assertAlmostEqual(cal_energy.convert('J').value, 4.184, places=3)
        self.assertAlmostEqual(cal_energy.convert('Btu').value, 0.003965667,
                               places=6)

    def test_convert_temperature(self):
        from pyavia.core.units import dim

        # Temperatures °C, °F, require thorough checks.
        x = dim(32, '°F').convert('°C')
        self.assertEqual(x.value, 0)

        x = dim(-40, '°C').convert('°F')
        self.assertAlmostEqual(x.value, -40)

        x = dim(373.15, 'K').convert('°F')
        self.assertAlmostEqual(x.value, 212)

        x = dim(0, '°R').convert('°C')
        self.assertAlmostEqual(x.value, -273.15)

        x = dim(15, '°C').convert('K')
        self.assertAlmostEqual(x.value, 288.15)

        # Only °C^1 allowed.
        with self.assertRaises(ValueError):
            x = dim('°C^2')

        # Offset not allowed in derived units.
        with self.assertRaises(ValueError):
            x = dim('km.°F')

        # Conversion from offset total to difference not allowed.
        with self.assertRaises(ValueError):
            x = dim(65, '°F').convert('Δ°F')

        # Test temperature differences.
        x, y = dim(1, 'Δ°C'), dim(1, 'K')
        self.assertEqual(x.value, y.value)
        x, y = dim(1, 'Δ°F'), dim(1, '°F')
        self.assertEqual(x.value, y.value)

        # Test temperature components convert, including denominator.
        air_const_metric = dim(287.05287, 'J/kg/K')
        air_const_imp_slug = air_const_metric.convert('ft.lbf/slug/°R')
        self.assertAlmostEqual(air_const_imp_slug.value, 1716.56188,
                               places=5)

        air_const_imp_lbm = air_const_metric.convert('ft.lbf/lbm/°R')
        self.assertAlmostEqual(air_const_imp_lbm.value, 53.35, places=2)

        check_delta = air_const_metric.convert('J/kg/Δ°C')
        self.assertAlmostEqual(air_const_metric.value, check_delta.value,
                               places=5)

        # Test absolute temperatures cancel (K / K).
        a_ssl_k = (1.4 * air_const_metric * dim(288.15, 'K')) ** 0.5
        self.assertEqual(a_ssl_k.units, 'm.s⁻¹')
        self.assertAlmostEqual(a_ssl_k.to_real('m/s'), 340.29399, places=3)

        # Test total offset temperatures cancel (°C / K).
        a_ssl_c = (1.4 * air_const_metric * dim(15.0, '°C')) ** 0.5
        self.assertEqual(a_ssl_c.units, 'm.s⁻¹')
        self.assertAlmostEqual(a_ssl_c.to_real('m/s'), 340.29399, places=3)
