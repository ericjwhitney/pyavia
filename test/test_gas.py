from unittest import TestCase


# ======================================================================

class TestImperfectGas(TestCase):
    def test_all(self):
        from pyavia.aerodynamics import ImperfectGas
        from pyavia.units import dim

        def stp_checks(test_gas):
            self.assertAlmostEqual(test_gas.R.to_value('J/kg/K'),
                                   287.05287, places=5)
            self.assertAlmostEqual(test_gas.c_p.to_value('J/kg/K'),
                                   1003.33, places=2)
            self.assertAlmostEqual(test_gas.c_v.to_value('J/kg/K'),
                                   716.28, places=2)
            self.assertAlmostEqual(test_gas.gamma, 1.40, places=2)
            self.assertAlmostEqual(test_gas.a.to_value('m/s'), 340.4,
                                   places=1)
            # Check US units.
            self.assertAlmostEqual(test_gas.c_p.to_value('Btu/lbm/°R'),
                                   0.240, places=3)

        # Check standard temp. properties (Ps = 1 bar i.e. not relevant
        # to this).  Default gas is air.
        gas = ImperfectGas(P=dim(1, 'bar'), T=dim(15, '°C'), M=0.0)
        stp_checks(gas)

        # Check stagnation.
        gas = ImperfectGas(P0=dim(1, 'bar'), T0=dim(15, '°C'), M=0.0)
        stp_checks(gas)

        # Check air at higher temp (poly range). First setup room
        # temperature.
        gas = ImperfectGas(h=dim(1.937741828959338, 'MJ/kg'),
                           P=dim(1, 'bar'), M=0.0, gas='air')
        # Assigning h = 1.937741828959338 MJ/kg should set T = 1400 K
        # and leave everything else alone. Gas constant 'R' should be
        # unchanged.
        self.assertAlmostEqual(gas.T.value, 1400.0, places=1)
        self.assertAlmostEqual(gas.R.value, 287.05287, places=5)
        self.assertAlmostEqual(gas.c_p.value, 1200, places=0)
        self.assertAlmostEqual(gas.gamma, 1.314, places=2)

        # Check h, s terms across a compressor (W&F Ex. C3.2(iii).
        inlet_flow = ImperfectGas(P=dim(1, 'atm'), T=dim(288.15, 'K'),
                                  M=0.0, gas='air')
        h_in = inlet_flow.h.convert('kJ/kg').value
        s_in = inlet_flow.s.convert('kJ/kg/K').value

        isen_exit = ImperfectGas(P=inlet_flow.P * 20,
                                 T=dim(667.36, 'K'), M=0.0)
        h_out_i = isen_exit.h.convert('kJ/kg').value
        s_out_i = isen_exit.s.convert('kJ/kg/K').value
        self.assertAlmostEqual(h_out_i - h_in, 389.997, places=2)
        self.assertAlmostEqual(s_out_i - s_in, 0.0, places=3)

        act_exit = ImperfectGas(P=inlet_flow.P * 20,
                                T=dim(731.42, 'K'), M=0.0)
        h_out_a = act_exit.h.convert('kJ/kg').value
        self.assertAlmostEqual(h_out_a - h_in, 458.82, places=2)

        # Check high mach atmospheric flow in US units. For a standard
        # day at altitude = 40,000' and 400 KTAS -> P0 = 544.0402 psf,
        # T0 = 427.9023 °R, M = 0.69738 gamma = 1.4015
        cruise_flow = ImperfectGas(T0=dim(428.0438, '°R'),
                                   P0=dim(542.2000, 'psf'),
                                   M=0.69738)

        # Check ambient P, T.
        self.assertAlmostEqual(cruise_flow.P.convert('psf').value,
                               391.686, places=1)
        self.assertAlmostEqual(cruise_flow.T.convert('°F').value,
                               -69.70, places=1)

        # Check back computed airspeed.
        v_ktas = (cruise_flow.a * cruise_flow.M).convert('kt').value
        self.assertAlmostEqual(v_ktas, 400, places=0)

        # Check conversion from h, s both ways preserves properties.
        gas_in = ImperfectGas(P=dim(1, 'bar'), T=dim(15, '°C'), M=0.8)
        gas_out = ImperfectGas(h=gas_in.h, s=gas_in.s, M=0.8)
        self.assertAlmostEqual(gas_in.P.convert('kPa').value,
                               gas_out.P.convert('kPa').value,
                               places=5)
        self.assertAlmostEqual(gas_in.T.convert('K').value,
                               gas_out.T.convert('K').value,
                               places=5)

        # Check that we can construct obscure but valid combinations.
        #
        # Take standard day alt = 50,000', M = 2.0.  Ambient P =
        # 242.215 psf and T = 389.97°R.  Calculating stagnation
        # properties using γ = 1.4 gives:
        #   T0/T = 1+(γ-1)/2*M^2 = 1.8
        #   P0/P = (1+(γ-1)/2*M^2)^(γ/(γ-1)) = (1.8)^(1.4/0.4) = 7.82445
        #
        # Multiplying:
        #   T0 = 1.8 * 389.970 = 701.95°R
        #   P0 = 7.82445 * 242.215 = 1895.2 psf
        #
        # We check that P, T, M are correctly converged from stagnation T,
        # P and expanded pressure.
        from_stag = ImperfectGas(P0=dim(1895.2, 'psf'),
                                 P=dim(242.215, 'psf'),
                                 T0=dim(701.95, '°R'))
        self.assertAlmostEqual(from_stag.T.convert('°R').value,
                               389.97, places=1)  # Real γ effect noted.
        self.assertAlmostEqual(from_stag.P.convert('psf').value,
                               242.215, places=3)
        self.assertAlmostEqual(from_stag.M, 2.00, places=2)


# ======================================================================


# noinspection PyPep8Naming
class TestPerfectGas(TestCase):
    def test_all(self):
        from pyavia.aerodynamics import PerfectGas
        from pyavia.units import dim

        def stp_checks(test_gas):
            self.assertAlmostEqual(test_gas.gamma, 1.400, places=3)
            self.assertAlmostEqual(test_gas.R.to_value('J/kg/K'),
                                   287.05287, places=5)
            self.assertAlmostEqual(test_gas.c_p.to_value('J/kg/K'),
                                   1004.685, places=2)
            self.assertAlmostEqual(test_gas.c_v.to_value('J/kg/K'),
                                   717.632, places=2)
            self.assertAlmostEqual(test_gas.gamma, 1.40, places=2)
            self.assertAlmostEqual(test_gas.a.value, 340.3, places=1)

            # Check US units.
            self.assertAlmostEqual(test_gas.c_p.to_value('Btu/lbm/°R'),
                                   0.240, places=3)

        # Check standard temp. properties (Ps = 1 bar) i.e. not relevant
        # to this).  Default gas is air.
        gas = PerfectGas(P=dim(1, 'bar'), T=dim(15, '°C'), M=0.0)
        stp_checks(gas)

        # Test entropy in = entropy out.
        gas = PerfectGas(P=dim(20, 'bar'), T=dim(800, 'K'), M=0.0)
        P, s = gas.P, gas.s
        newgas = PerfectGas(P=P, s=s, M=0.0)

        self.assertAlmostEqual(gas.P.to_value('kPa'),
                               newgas.P.to_value('kPa'), places=3)
        self.assertAlmostEqual(gas.T.to_value('K'),
                               newgas.T.to_value('K'), places=3)

# ======================================================================
