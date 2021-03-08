from unittest import TestCase


class TestDim(TestCase):
    def test___init__(self):
        from pyavia import Dim, Units, UnitsError

        # Test no arguments:  Value = 1, no units.
        x = Dim()
        self.assertEqual(x.value, 1)
        self.assertIsInstance(x.value, int)

        # Test one positional argument: Dimensionless.
        x = Dim(3)
        self.assertEqual(x.value, 3)
        self.assertIsInstance(x.value, int)
        self.assertEqual(x.units, Units())

        # Test two positional arguments: Value, units given.
        x, y = Dim(3, 'ft'), Dim(5 + 6j, 's')
        self.assertEqual(x.value, 3)
        self.assertIsInstance(x.value, int)
        self.assertEqual(x.units, Units(L=('ft', 1)))
        self.assertEqual(x.label, 'ft')
        self.assertEqual(y.value, 5 + 6j)
        self.assertIsInstance(y.value, complex)

        # Test incorrect positional arguments.
        with self.assertRaises(TypeError):
            x = Dim(2, 3)  # Second arg should be string.
        with self.assertRaises(TypeError):
            x = Dim(1, 'km', 4)  # 0-2 args only.

        # Test cannot create derived units with offset temperature.
        with self.assertRaises(UnitsError):
            x = Dim(1, '°C/day')

    def test___add__(self):
        # Also tests __radd__
        from pyavia import Dim, Units, UnitsError

        # Test addition of like units, preserving character.
        x, y = Dim(1, 'ft'), Dim(12, 'in')
        x_p_y = x + y
        self.assertEqual(x_p_y.value, 2)
        self.assertIsInstance(x_p_y.value, int)
        self.assertEqual(x_p_y.units, Units('ft'))
        x = 5 + Dim(1, Units())  # __radd__ and unit cancellation check.
        self.assertIsInstance(x, int)
        self.assertEqual(x, 6)

        # Test addition of imcompatible units disallowed.
        with self.assertRaises(UnitsError):
            x = Dim(10, 'kg') + Dim(5, 'm')
            x = 2 + Dim(2, 'kg')  # __radd__ check.

        # Test total temperature addition (special case).
        x = Dim(25, '°C') + Dim(5, 'Δ°C')
        self.assertEqual(x.units.θ[0], '°C')
        self.assertEqual(x.value, 30)
        with self.assertRaises(UnitsError):
            x = Dim(32, '°C') + Dim(32, '°F')  # Not allowed.

    def test__sub__(self):
        # Also tests __rsub__
        from pyavia import Dim, Units, UnitsError

        # Test subtraction of imcompatible units disallowed.
        with self.assertRaises(UnitsError):
            x = (1 * Units('mol')) - (2 * Units('kg'))
            x = 3 - Dim(2, 'mmol')  # __rsub__ check

        # Test offset temperature subtration.
        x = Dim(25, '°C') - Dim(5, 'Δ°C')
        self.assertEqual(x.units.θ[0], '°C')
        self.assertEqual(x.value, 20)
        x = Dim(32, '°F') - Dim(40, 'K')
        self.assertAlmostEqual(x.value, -40)
        self.assertEqual(x.units, Units('°F'))
        x = Dim(32, '°F') - Dim(0, '°C')
        self.assertAlmostEqual(x.value, 0)
        self.assertEqual(x.units, Units('Δ°F'))

    def test___mul__(self):
        from pyavia import Dim, Units

        # Check multiply by plain units gives Dim result.
        x = 7.5 * Units('km')
        self.assertEqual(x.value, 7.5)
        self.assertEqual(x.units, Units('km'))

        # Check value type follows multiply rules.
        x = 4 * 3 * Units('N')
        self.assertEqual(x.value, 12)
        self.assertIsInstance(x.value, int)
        x = Dim(4, 'm') * Dim(2.0, 'm')
        self.assertEqual(x.value, 8.0)
        self.assertIsInstance(x.value, float)

        # Check multiply gives correct final units from LHS.
        x = Dim(1, 'kg') * Units('G')
        self.assertAlmostEqual(x.value, 9.80665, places=5)
        self.assertEqual(x.units, Units('N'))
        x = Dim(10, 'kg') / Dim(5, 'mol')
        self.assertAlmostEqual(x.value, 2)
        self.assertEqual(x.units, Units('kg/mol'))

        # Check degrees are retained.
        x = Dim(5, 'kg') * Units('°^-1')
        self.assertEqual(x.units, Units('kg/°'))
        x = Dim(10, 'm') * Units('deg')
        self.assertEqual(x.units, Units('m.deg'))

        # Check radians / steradians disappear automatically.
        r, omega = Dim(10, 'ft'), Dim(30, 'rad/s')
        v_t = r * omega
        self.assertEqual(v_t.value, 300)
        self.assertIsInstance(v_t.value, int)
        self.assertEqual(v_t.units, Units('ft/s'))
        # x = 1 * Units('sr')  # XXX TODO CHECK MATHS ON THIS
        # lhs.assertEqual(x, 1)
        # lhs.assertIsInstance(x, int)

        # Check inverse radians / steradians are retained.
        x = 50 / Units('rad')
        self.assertEqual(x.units, Units('rad^-1'))
        x = 1 / Units('sr')
        self.assertEqual(x.units, Units('sr^-1'))

        # Check multiply giving no units should result in plain value.
        x = 5.0 * Dim()
        self.assertIsInstance(x, float)
        self.assertEqual(x, 5)
        x = Dim(6, 'km') * Dim(2, 'km^-1')
        self.assertIsInstance(x, int)
        self.assertEqual(x, 12)

    def test___truediv__(self):
        from pyavia import Dim, Units

        # Check value type follows division rules.
        x = Dim(4, 'm') / 2
        self.assertEqual(x.value, 2.0)
        self.assertIsInstance(x.value, float)
        self.assertEqual(x.units, Units(L=('m', 1)))

        # Check division gives correct final units from LHS.
        x = Dim(9.80665, 'N') / Dim(1, 'G').convert('ft/s^2')
        self.assertAlmostEqual(x.value, 1, places=5)
        self.assertEqual(x.units, Units('kg'))

    def test_convert_operations(self):
        from pyavia import Dim, Units, UnitsError

        # Test conversion to lhs does nothing.
        x = Dim(3, 'kg').convert('kg')
        self.assertEqual(x.value, 3)
        self.assertEqual(x.units, Units('kg'))

        # Test integer conserved during whole-number conversion.
        x = Dim(144, 'in^2').convert('ft^2')
        self.assertEqual(x.value, 1)
        self.assertIsInstance(x.value, int)

        # Test float conserved when integer is possible.
        x = Dim(1.0, 'mol').convert('mmol')
        self.assertIsInstance(x.value, float)

        # Test type promotion for non-whole-number conversion of int.
        x = Dim(1, 'in').convert('mm')
        self.assertEqual(x.value, 25.4)
        self.assertIsInstance(x.value, float)

        # Test k values causes miscompare (constructed g vs kg).
        with self.assertRaises(UnitsError):
            x = Dim(5, 'kg').convert(Units(k=1000, M=('g', 1)))

        # Test multi-step conversion with large jump avoids warning message
        # (requires shortcut definitions to be in units.py).
        x = Dim(1, 'nmol').convert('Gmol')  # Should give no warning.

        # Test unit multipliers are carried correctly.
        x = Dim(1, 'US_gal').convert('in^3')
        self.assertEqual(x.value, 231)

        # Test passing Dim to itself does conversion.
        x = Dim(Dim(25.4, 'mm'), 'in')
        self.assertAlmostEqual(x.value, 1.0)
        self.assertEqual(x.units, Units('in'))

        with self.assertRaises(UnitsError):
            x = Dim(Dim(100, 'm'), 'ha')

    def test_convert_fundamental(self):
        from pyavia import Dim, Units, UnitsError

        # Test total temperatures °C, °F, etc (special case units).
        x = Dim(32, '°F').convert('°C')
        self.assertEqual(x.value, 0)
        x = Dim(-40, '°C').convert('°F')
        self.assertAlmostEqual(x.value, -40)
        x = Dim(373.15, 'K').convert('°F')
        self.assertAlmostEqual(x.value, 212)
        x = Dim(0, '°R').convert('°C')
        self.assertAlmostEqual(x.value, -273.15)

        with self.assertRaises(UnitsError):
            # Only °C^1 allowed.
            x = Units(θ=('°C', 2))
            # Offset notallowed in derived.
            x = Units(L=('km', 1), θ=('°F', 1))
            # Conversion from offset total to difference not allowed.
            x = Dim(65, '°F').convert('Δ°F')

        # Test temperature differences.
        x, y = Dim(1, 'Δ°C'), Dim(1, 'K')
        self.assertEqual(x.value, y.value)
        x, y = Dim(1, 'Δ°F'), Dim(1, '°F')
        self.assertEqual(x.value, y.value)

        # Test temperature components including denominator.
        air_const_metric = Dim(287.05287, 'J/kg/K')
        air_const_imp_slug = air_const_metric.convert('ft.lbf/slug/°R')
        self.assertAlmostEqual(air_const_imp_slug.value, 1716.56188,
                               places=5)
        air_const_imp_lbm = air_const_metric.convert('ft.lbf/lbm/°R')
        self.assertAlmostEqual(air_const_imp_lbm.value, 53.35, places=2)
        check_delta = air_const_metric.convert('J/kg/Δ°C')
        self.assertAlmostEqual(air_const_metric.value, check_delta.value,
                               places=5)

        # Test current.
        x = Dim(1, 'A').convert('mA')
        self.assertEqual(x.value, 1000)
        self.assertEqual(x.units, Units('mA'))

        # Test angle units.
        x = (5 * Units('m/°')).convert('m/rad')
        self.assertAlmostEqual(x.value, 286.47889, places=4)

    def test_convert_derived(self):
        from pyavia import Dim, Units, UnitsError

        # Test acceleration.
        x = (9.80665 * Units('m.s⁻²')).convert('ft/s/s')
        self.assertAlmostEqual(x.value, 32.17404856, places=8)

        # Test pressures (a common derived unit with many forms).
        x = Dim(1, 'atm')
        mm_hg = x.convert('mmHg').value
        kpa = x.convert('kPa').value
        bar = x.convert('bar').value
        in_hg = x.convert('inHg').value
        psi = x.convert('psi').value
        psf = x.convert('psf').value
        self.assertAlmostEqual(mm_hg, 760, places=3)  # Approx check value.
        self.assertAlmostEqual(kpa, 101.325)
        self.assertAlmostEqual(bar, 1.01325)
        self.assertAlmostEqual(in_hg, 29.921, places=3)  # Approx check value.
        self.assertAlmostEqual(psi, 14.696, places=3)  # Approx check value.
        self.assertAlmostEqual(psf, 2116.22, places=2)  # Approx check value.

        # Test speed.
        rpm = Dim(2500, 'RPM')  # Rotational speed.
        omega = rpm.convert('rad/s')
        self.assertAlmostEqual(omega.value, 261.799388, places=5)
        v_t = Dim(3, 'ft') * omega  # Radians should fall off at this point.
        self.assertAlmostEqual(v_t.value, 785.398163, places=5)
        self.assertEqual(v_t.units, Units('fps'))

        # Test power.
        x = Dim(100, 'hp')
        x = x.convert('kW')
        self.assertAlmostEqual(x.value, 74.56999, places=3)

        # Test some unusual units.
        k_ic_metric = Dim(51.3, 'MPa.m⁰ᐧ⁵')  # Fracture toughness 7039-T6351.
        k_ic_imp = k_ic_metric.convert('ksi.in^0.5')
        self.assertAlmostEqual(k_ic_imp.value, 46.7, places=1)
        cal_energy = Dim(1, 'cal')
        self.assertAlmostEqual(cal_energy.convert('J').value, 4.184, places=3)
        self.assertAlmostEqual(cal_energy.convert('Btu').value, 0.003965667,
                               places=6)
