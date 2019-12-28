from unittest import TestCase


class TestGasFlowWF(TestCase):
    def test_all(self):
        from gasflow import GasFlowWF
        from units import Dim, Units

        def stp_checks(test_gas):
            self.assertEqual(test_gas.gas, 'air')
            self.assertEqual(test_gas.R.units, Units('J/kg/K'))
            self.assertAlmostEqual(test_gas.R.value, 287.05287, places=5)
            self.assertEqual(test_gas.cp.units, Units('J/kg/K'))
            self.assertAlmostEqual(test_gas.cp.value, 1003.33, places=2)
            self.assertEqual(test_gas.cv.units, Units('J/kg/K'))
            self.assertAlmostEqual(test_gas.cv.value, 716.28, places=2)
            self.assertAlmostEqual(test_gas.gamma, 1.40, places=2)
            self.assertAlmostEqual(test_gas.a.value, 340.4, places=1)
            # Check US units.
            self.assertAlmostEqual(test_gas.cp.convert('Btu/lbm/°R').value,
                                   0.240, places=3)

        # Check standard temp. properties (Ps = 1 bar i.e. not relevant to
        # this).  Default gas is air.
        gas = GasFlowWF(P=Dim(1, 'bar'), T=Dim(15, '°C'))
        stp_checks(gas)
        # Check stagnation.
        gas = GasFlowWF(P0=Dim(1, 'bar'), T0=Dim(15, '°C'), M=0.0)
        stp_checks(gas)

        # Check air at higher temp (poly range). First setup room temperature.
        gas = GasFlowWF(P=Dim(1, 'bar'), h=Dim(1.937741828959338, 'MJ/kg'),
                        gas='air')
        # Assigning h = 1.937741828959338 MJ/kg should set T = 1400 K and
        # leave everything else alone.
        self.assertAlmostEqual(gas.T.value, 1400.0, places=1)
        self.assertAlmostEqual(gas.R.value, 287.05287, places=5)  # Unchanged.
        self.assertAlmostEqual(gas.cp.value, 1200, places=0)
        self.assertAlmostEqual(gas.gamma, 1.314, places=2)

        # Check h, s terms across a compressor (W&F Ex. C3.2(iii).
        inlet_flow = GasFlowWF(P=Dim(1, 'atm'), T=Dim(288.15, 'K'), M=0,
                               gas='air')
        h_in = inlet_flow.h.convert('kJ/kg').value
        s_in = inlet_flow.s.convert('kJ/kg/K').value

        isen_exit = GasFlowWF(P=inlet_flow.P * 20, T=Dim(667.36, 'K'))
        h_out_i = isen_exit.h.convert('kJ/kg').value
        s_out_i = isen_exit.s.convert('kJ/kg/K').value
        self.assertAlmostEqual(h_out_i - h_in, 389.997, places=2)
        self.assertAlmostEqual(s_out_i - s_in, 0.0, places=3)

        act_exit = GasFlowWF(P=inlet_flow.P * 20, T=Dim(731.42, 'K'))
        h_out_a = act_exit.h.convert('kJ/kg').value
        self.assertAlmostEqual(h_out_a - h_in, 458.82, places=2)

        # Check high mach atmospheric flow, US units.
        # standard day altitude = 40,000' at 400 KTAS
        # -> P0 = 544.0402 psf, T0 = 427.9023 °R, M = ‭0.69738 gamma = 1.4015
        cruise_flow = GasFlowWF(T0=Dim(428.0438, '°R'),
                                P0=Dim(542.2000, 'psf'),
                                M=0.69738)

        # Check ambient P, T.
        self.assertAlmostEqual(cruise_flow.P.convert('psf').value,
                               391.686, places=1)
        self.assertAlmostEqual(cruise_flow.T.convert('°F').value,
                               -69.70, places=1)

        # Check back computed airspeed.
        v_ktas = (cruise_flow.a * cruise_flow.M).convert('kt').value
        self.assertAlmostEqual(v_ktas, 400, places=0)

        # Check h, s in from out
        gas_in = GasFlowWF(P=Dim(1, 'bar'), T=Dim(15, '°C'), M=0.8)
        gas_out = GasFlowWF(h=gas_in.h, s=gas_in.s, M=0.8)
        self.assertAlmostEqual(float(gas_in.P), float(gas_out.P), places=5)
        self.assertAlmostEqual(float(gas_in.T), float(gas_out.T), places=5)


# noinspection PyPep8Naming
class TestPerfectGasFlow(TestCase):
    def test_all(self):
        from gasflow import PerfectGasFlow
        from units import Dim, Units

        def stp_checks(test_gas):
            self.assertAlmostEqual(test_gas.gamma, 1.400, places=3)
            self.assertEqual(test_gas.R.units, Units('J/kg/K'))
            self.assertAlmostEqual(test_gas.R.value, 287.05287, places=5)
            self.assertEqual(test_gas.cp.units, Units('J/kg/K'))
            self.assertAlmostEqual(test_gas.cp.value, 1004.685, places=2)
            self.assertEqual(test_gas.cv.units, Units('J/kg/K'))
            self.assertAlmostEqual(test_gas.cv.value, 717.632, places=2)
            self.assertAlmostEqual(test_gas.gamma, 1.40, places=2)
            self.assertAlmostEqual(test_gas.a.value, 340.3, places=1)
            # Check US units.
            self.assertAlmostEqual(test_gas.cp.convert('Btu/lbm/°R').value,
                                   0.240, places=3)

        # Check standard temp. properties (Ps = 1 bar i.e. not relevant to
        # this).  Default gas is air.
        gas = PerfectGasFlow(P=Dim(1, 'bar'), T=Dim(15, '°C'))
        stp_checks(gas)

        # Test entropy in = entropy out.
        gas = PerfectGasFlow(P=Dim(20, 'bar'), T=Dim(800, 'K'))
        P, s = gas.P, gas.s
        newgas = PerfectGasFlow(P=P, s=s)
        self.assertAlmostEqual(float(gas.P), float(newgas.P), places=6)
        self.assertAlmostEqual(float(gas.T), float(newgas.T), places=6)
