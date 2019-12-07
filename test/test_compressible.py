from unittest import TestCase


class TestCompressibleGas(TestCase):
	def test_all(self):
		from compressible import ComprFlow
		from units import Dim, Units

		def stp_checks(test_gas):
			self.assertEqual(test_gas.gas, 'air')
			self.assertEqual(test_gas.R.units, Units('J/kg/K'))
			self.assertAlmostEqual(test_gas.R.value, 287.05287, places=5)
			self.assertEqual(test_gas.Cp.units, Units('J/kg/K'))
			self.assertAlmostEqual(test_gas.Cp.value, 1003.33, places=1)
			self.assertEqual(test_gas.Cv.units, Units('J/kg/K'))
			self.assertAlmostEqual(test_gas.Cv.value, 716.28, places=1)
			self.assertAlmostEqual(test_gas.gamma, 1.40, places=2)
			self.assertAlmostEqual(test_gas.a.value, 340.4, places=1)
			# Check US units.
			self.assertAlmostEqual(test_gas.Cp.convert('Btu/lbm/°R').value,
			                       0.240, places=3)

		# Check standard temp. properties (Ps = 0 i.e. not relevant to this).
		# Default gas is air.
		gas = ComprFlow(P=Dim(0, 'Pa'), T=Dim(15, '°C'), M=0.0)
		stp_checks(gas)
		# Check stagnation.
		gas = ComprFlow(Pt=Dim(0, 'Pa'), Tt=Dim(15, '°C'), M=0.0)
		stp_checks(gas)

		# Check air at higher temp (poly range). First setup room temperature.
		gas = ComprFlow(P=Dim(1, 'bar'), h=Dim(1.937741828959338, 'MJ/kg'),
		                M=0, gas='air')
		# Assigning h = 1.937741828959338 MJ/kg should set T = 1400 K and
		# leave everything else alone.
		self.assertEqual(gas.Tt.units, Units('K'))
		self.assertAlmostEqual(gas.Tt.value, 1400.0, places=1)
		self.assertAlmostEqual(gas.R.value, 287.05287, places=5)  # Unchanged.
		self.assertAlmostEqual(gas.Cp.value, 1200, places=0)
		self.assertAlmostEqual(gas.gamma, 1.314, places=2)

		# Check air at even higher temp (eqn range).
		gas = ComprFlow(P=Dim(1, 'atm'), T=Dim(2500, 'K'), M=0, gas='air')
		self.assertAlmostEqual(gas.R.value, 287.05287, places=5)  # Unchanged.
		self.assertAlmostEqual(gas.Cp.value, 1259.8, places=1)
		self.assertAlmostEqual(gas.gamma, 1.294, places=2)
