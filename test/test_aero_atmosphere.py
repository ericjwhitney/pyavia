from unittest import TestCase


class TestAtmosphere(TestCase):
    # noinspection PyUnresolvedReferences
    def test__init__(self):
        from pyavia.aero import Atmosphere
        from pyavia.units import Dim

        # Check invalid constructions.
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker,PyArgumentList
            Atmosphere(3)  # Invalid postional arg.
            Atmosphere(h=Dim(1000, 'ft'))  # Invalid combination.

        with self.assertRaises(ValueError):
            Atmosphere(H=0)  # Arg needs dimensions.

        # Test for correct results from some standard altitude values.
        Atmosphere.set_default_style('SI')  # K, kPa, m, etc.

        atm = Atmosphere(H='SSL')  # Sea level - thorough test.
        self.assertAlmostEqual(atm.T.value, 288.15)
        self.assertAlmostEqual(atm.P.value, 101.325)
        self.assertAlmostEqual(atm.ρ.value, 1.225)
        self.assertAlmostEqual(atm.a.value, 340.294, places=3)
        self.assertAlmostEqual(atm.μ.value, 1.7894e-5, places=8)
        self.assertAlmostEqual(atm.ν.value, 1.4607e-5, places=8)
        # self.assertAlmostEqual(atm.c_p.value, 1005.7, places=1)
        # self.assertAlmostEqual(atm.c_v.value, 719, places=1)
        # self.assertAlmostEqual(atm.gamma, 1.4, places=1)

        atm = Atmosphere(H=Dim(11, 'km'))  # Known geompotential level.
        self.assertAlmostEqual(atm.T.value, 216.65, places=4)
        self.assertAlmostEqual(atm.P.value, 22.632, places=4)

        atm = Atmosphere(H=Dim(1000, 'm'))  # Inter-level, low.
        self.assertAlmostEqual(atm.T.value, 281.65, places=3)
        self.assertAlmostEqual(atm.P.value, 89.875, places=3)

        atm = Atmosphere(H=Dim(47.5, 'km'))  # Inter-level, high, no lapse.
        self.assertAlmostEqual(atm.T.value, 270.65, places=3)
        self.assertAlmostEqual(atm.P.value, 0.104123, places=5)
        self.assertAlmostEqual(atm.μ.value, 1.7037e-5, places=9)

        # Test arbitrary non-standard atmospheres.
        atm = Atmosphere(P=Dim(90, 'kPa'), T=Dim(-15, '°C'))
        self.assertAlmostEqual(atm.ρ.value, 1.21453067, places=8)

        # Test pressure altitude construction.
        atm = Atmosphere(H_press=Dim(48000, 'ft'), T=Dim(-52, '°C'))
        self.assertAlmostEqual(atm.pressure.convert('psi').value, 1.85176,
                               places=4)
        self.assertAlmostEqual(atm.density.convert('slug/ft^3').value,
                               0.000390283, places=7)
        self.assertAlmostEqual(atm.density_altitude.convert('ft').value,
                               48427.68, places=2)

        # Test pressure altitude construction with offset.
        atm = Atmosphere(H_press=Dim(34000, 'ft'), T_offset=Dim(+15, 'Δ°C'))
        self.assertAlmostEqual(atm.P.value, 25.00, places=2)
        self.assertAlmostEqual(atm.T.convert('°C').value, -37.36, places=2)
        self.assertAlmostEqual(atm.density.value, 0.369349, places=6)
        self.assertAlmostEqual(atm.density_altitude.convert('ft').value,
                               35707.89, places=1)

        atm = Atmosphere(H_press=Dim(7000, 'ft'), T_offset=Dim(5, 'Δ°C'))
        self.assertAlmostEqual(atm.density.convert('slug/ft^3').value,
                               0.0018923202, places=7)
        self.assertAlmostEqual(atm.density_altitude.convert('ft').value,
                               7586.4166, places=2)

        # Test geometric altitude.
        atm = Atmosphere(h_geometric=Dim(50120.16, 'ft'))
        self.assertAlmostEqual(atm.pressure_altitude.convert('ft').value,
                               50000, places=2)

        # Test hot gases.
        # atm = Atmosphere(P=(10, 'kPa'), T=(1500, 'K'))
        # self.assertAlmostEqual(atm.c_p.value, 1216, places=-1)
        # self.assertAlmostEqual(atm.c_v.value, 929, places=-1)
        # self.assertAlmostEqual(atm.gamma, 1.309, places=2)
        # atm = Atmosphere(P=(101325, 'Pa'), T=(4000, '°R'))
        # self.assertAlmostEqual(atm.gamma, 1.298, places=2)

    # noinspection PyTypeChecker
    def test_methods(self):
        from pyavia.aero import Atmosphere
        from pyavia.units import Dim

        Atmosphere.set_default_style('SI')  # K, kPa, m, etc.

        # Test pressure altitude result (arbitrary temp).
        atm = Atmosphere(P=Dim(300, 'hPa'), T=Dim(5, 'K'))
        self.assertAlmostEqual(atm.pressure_altitude.value, 9164, places=1)

        # Test density altitude result.
        atm = Atmosphere(P=Dim(308.0113, 'psf'), T=Dim(-69.7, '°F'))
        h_d = atm.density_altitude.convert('ft')
        self.assertAlmostEqual(h_d.value, 45000, places=0)

        # Test ratios.
        atm = Atmosphere(H_press=Dim(40000, 'ft'), T_offset=Dim(-10, 'Δ°C'))
        self.assertAlmostEqual(atm.δ, 0.185087, places=5)
        self.assertAlmostEqual(atm.θ, 0.717161, places=5)
        self.assertAlmostEqual(atm.σ, 0.258083, places=5)
