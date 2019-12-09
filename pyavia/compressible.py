"""
Equations and constants relating to gases and compressible flow.

Contains:
	ComprFlow     Class giving fixed properties of a flowing compressible gas.
"""
# Last updated: 9 December 2019 by Eric J. Whitney

from math import exp, log
from typing import Set, Tuple

from units import Dim, make_total_temp
from util import iterate_fn, bisect_root

__all__ = ['ComprFlow']


# -----------------------------------------------------------------------------


# noinspection PyPep8Naming,NonAsciiCharacters
class ComprFlow:

	"""
	Class representing a flowing compressible gas, including non-linear
	thermal values.  Once initialised, properties are fixed.

	Notes:
		- cp and γ vary with temperature.  For higher temperatures the
		interpolating polynomial method of Walsh & Fletcher Chapter 3 is
		used to calculate cp and h.

		- The following properties may not be defined for extreme / high
		temperature flow states, although the object will successfully
		initialise:
			- h
			- cp_integral_term
		In this case these properties will return None.

		- Static (ambient) temperature and pressure are	used internally for
		convenience of cp and h calculation.

		- As expected, specific enthalpy has an arbitrary baseline and
		should never be used as a standalone value.
	"""

	# Magic methods -----------------------------------------------------------

	def __init__(self, *, P: Dim = None, Pt: Dim = None, T: Dim = None,
	             Tt: Dim = None, h: Dim = None, s: Dim = None, M: float,
	             W: Dim = None, gas: str = 'air', FAR: float = 0.0):
		"""
		Constructs a representation of a compressible gas flowing in 1-D.
		Notes:
			- Set either T, Tt, h or s (note the h and s baselines are
			arbitrary; refer to formulation used).
			- Set either P or Pt.
			- Mach number must be supplied.

		Note: Any combination of total / stagnation (Pt, Tt) and static (P, T)
		pressure or temperature can be set, but of course only one at a time.

		Args:
			Pt: (Dim) Total / stagnation pressure (also P0).
			Tt: (Dim) Total /stagnation temperature (also T0).
			Pt: (Dim) Stream / static pressure.
			T: (Dim) Stream / static temperature.
			h: (Dim) Specific enthalpy.
			s: (Dim) Specific entropy.
			M: (float) Mach number.
			W: (dim) Mass flowrate.
			gas: (str) identifier (default = 'air').  Available 'gas' values
				are:
					air             Dry air
					burned_kero     Products of combustion for kerosene in
									dry air
					burned_diesel   Products of combustion for diesel in
									dry air
			FAR: (float) Fuel-Air Ratio (default = 0.0)

		Raises:
			ValueError on unknown gas.  TypeError on invalid arguments.
		"""
		self._M, self._W, self._gas, self._FAR = M, W, gas, FAR
		if self._gas not in {'air', 'burned_kero', 'burned_diesel'}:
			raise ValueError(f"Unknown gas: {gas}")

		# Most other properties depend on T so this is set next.

		self._P, self._T = 0, 0   # Dummies required for Tt / h functions.
		if T and not any([Tt, h, s]):
			self._T = make_total_temp(T)  # Straightforward case.
		elif Tt and not any([T, h, s]):
			self._T = self._get_T_from_Tt(Tt)  # From stagnation.
		elif h and not any([T, Tt, s]):
			self._T = self._get_T_from_h(h)  # From enthalpy.
		elif s and not any([T, Tt, h]):
			self._T = self._get_T_from_s(s)  # From entropy.
		else:
			raise TypeError(f"Invalid temperature arguments.")

		# Precompute remaining properties.
		if self._T.units.θ[0] == '°R':  # Try and match units intent.
			self._output_units = 'US'
		else:
			self._output_units = 'SI'

		# self._set_cp_from_T()
		T_K = self._T.convert('K').value
		try:
			# Set via polynomial.
			cp_, h_, s_ = _wfpoly_cp_h_s(T_K, self._gas, self._FAR)
			self._cp = Dim(cp_*1000, 'J/kg/K')
			self._h = Dim(h_*1000, 'kJ/kg')
			self._s = Dim(s_, 'kJ/kg/K')

		except ValueError:
			# Set via vibrator model.
			self._cp = Dim(_vib_cp(T_K)*1000, 'J/kg/K')
			self._h, self._s = None, None

		self._set_R()
		self._γ = self._cp / (self._cp - self._R)
		self._a = (self._γ * self._R * self._T)**0.5
		# self._set_h_from_T()
		# self._set_s_from_T()

		if P and not Pt:
			self._P = P
		elif Pt and not P:
			self._P = Pt / self.Pt_on_P
		else:
			raise TypeError(f"Invalid pressure arguments.")

	# Normal methods ----------------------------------------------------------

	@staticmethod
	def props() -> Set[str]:
		"""Returns a set containing strings of all the independent
		parameters required to fully initialise the object."""
		return {'P', 'T', 'M', 'W', 'gas', 'FAR'}

	def replace(self, **kwargs) -> 'ComprFlow':
		"""
		Makes a copy of this object except with nominated parameters replaced
		by the given values, e.g. similar to a partial __init__.

		Args:
			**kwargs: As per __init__.

		Returns:
			New ComprFlow object.
		"""
		new_kwargs = {key: getattr(self, key) for key in self.props()}

		for new_key, value in kwargs.items():
			if new_key in ('P', 'Pt'):
				new_kwargs.pop('P', None)
			if new_key in ('T', 'Tt', 'h'):
				new_kwargs.pop('T', None)
			new_kwargs[new_key] = value

		return ComprFlow(**new_kwargs)

	# Properties --------------------------------------------------------------

	@property
	def a(self) -> Dim:
		"""Local speed of sound a = (γRT)**0.5."""
		return self._a

	@property
	def cp(self) -> Dim:
		"""Specific heat capacity at constant pressure of the gas."""
		return self._cp

	@property
	def s(self) -> Dim:
		"""Entropy.  Arbitrary zero basis."""
		return self._s

	@property
	def cv(self) -> Dim:
		"""Specific heat capacity at constant volume of the gas."""
		return self._cp - self._R

	@property
	def FAR(self) -> float:
		"""Fuel-Air Ratio."""
		return self._FAR

	@property
	def gamma(self) -> float:
		"""Ratio of specific heats γ = cp / c_v."""
		return self._γ

	@property
	def gas(self) -> str:
		return self._gas

	@property
	def h(self) -> Dim:
		"""Specific enthalpy of the gas.  Note: The baseline is arbitrary."""
		return self._h

	@property
	def M(self) -> float:
		"""Mach number."""
		return self._M

	@property
	def Pt_on_P(self) -> float:
		"""Ratio of total (stagnation) pressure to static pressure."""
		return (1 + 0.5 * (self._γ - 1) * self._M ** 2) ** (self._γ / (
				self._γ - 1))

	@property
	def Pt(self) -> Dim:
		"""Total / stagnation pressure."""
		return self.Pt_on_P * self.P

	@property
	def P(self) -> Dim:
		"""Stream / ambient pressure."""
		return self._P

	@property
	def R(self) -> Dim:
		"""Gas constant R."""
		return self._R

	@property
	def s(self) -> Dim:
		"""Entropy"""
		return self._s

	@property
	def Tt_on_T(self) -> float:
		"""Ratio of total (stagnation) pressure to static temperature."""
		return 1 + 0.5 * (self._γ - 1) * self._M ** 2

	@property
	def Tt(self) -> Dim:
		"""Total / stagnation temperature."""
		return self.Tt_on_T * self.T

	@property
	def T(self) -> Dim:
		"""Stream / ambient temperature."""
		return self._T

	@property
	def W(self) -> Dim:
		"""Mass flowrate."""
		return self._W

	# Internal methods --------------------------------------------------------

	def _get_T_from_h(self, value: Dim) -> Dim:
		"""Find the stream temperature of the gas given the specific
		enthalpy, using a bisection method.  The initial T range is assumed
		to be 200 K -> 2000 K."""

		def h_error(T_try: Dim) -> Dim:
			return self.replace(P=Dim(0, 'kPa'), T=T_try).h - value

		T_L, T_R = Dim(200, 'K'), Dim(2000, 'K')
		return bisect_root(h_error, T_L, T_R, maxits=50,
		                   tol=Dim(1, 'J/kg'))  # Approx tol=1e-6

	def _get_T_from_s(self, value: Dim) -> Dim:
		"""Find the stream temperature of the gas given the specific
		entropy, using a bisection method.  The initial T range is assumed
		to be 200 K -> 2000K."""

		def s_error(T_try: Dim) -> Dim:
			return self.replace(P=Dim(0, 'kPa'), T=T_try).s - value

		T_L, T_R = Dim(200, 'K'), Dim(2000, 'K')
		return bisect_root(s_error, T_L, T_R, maxits=50,
		                   tol=Dim(1e-4, 'J/kg/K'))

	def _get_T_from_Tt(self, Tt_reqd: Dim) -> Dim:
		"""Find the stream temperature of the gas given the total (stagnation)
		temperature via iteration..  This is due to the non-linear
		dependence of γ on Ts."""
		Tt_reqd = make_total_temp(Tt_reqd)
		if self._M == 0:  # Already at stagnation.
			return Tt_reqd

		# Iterate temperature due to non-linear dependence of γ on Ts.
		# Start iteration with Ts = T.
		def new_Ts(try_Ts):
			flow = ComprFlow(P=Dim(0, 'kPa'), T=try_Ts, M=self._M, W=self._W,
			                 gas=self._gas, FAR=self._FAR)
			return Tt_reqd / flow.Tt_on_T

		return iterate_fn(new_Ts, x_start=Tt_reqd, xtol=Dim(1e-6, 'K'))

	def _set_R(self) -> None:
		R_air = 287.05287  # J/kg/K.
		if self._gas == 'air':
			R_adj = R_air
		elif self._gas == 'burned_kero':
			R_adj = R_air - 0.00990 * self.FAR + 1e-7 * self.FAR ** 2
		elif self._gas == 'burned_diesel':
			R_adj = R_air - 8.0262 * self.FAR + 3e-7 * self.FAR ** 2
		else:
			raise RuntimeError(f"Reached unreachable point.")

		self._R = Dim(R_adj, 'J/kg/K')
		if self._output_units == 'US':
			self._R = self._R.convert('Btu/lbm/°R')


# -----------------------------------------------------------------------------
# Internal functions.

# noinspection PyPep8Naming
def _wfpoly_cp_h_s(T_K: float, gas: str, FAR: float) -> Tuple[float, float,
                                                              float]:
	"""
	Computes cp, h, s for a given temperature (K) using the interpolating
	polynomial method of Walsh & Fletcher.  Valid for gas in {
	'air', 'burned_kero', 'burned_diesel'}.  FAR is fuel-air ratio.

	Returned values:
		cp	kJ/kg/K
		h 	MJ/kg
		s	kJ/kg/K

	"""
	# Applicable range.
	if not (200 <= T_K <= 2000):
		raise ValueError(f"T = {T_K} K outside allowable range.")

	# Eqn F3.23 A coefficients.
	A_COEFF = {
		# Dry air with/without kerosene or diesel products of combustion.
		'air': (0.992313, 0.236688, -1.852148, 6.083152, -8.893933, 7.097112,
		        -3.234725, 0.794571, -0.081873, 0.422178, 0.001053), }

	if gas in ('air', 'burned_kero', 'burned_diesel'):
		coeff_a = A_COEFF['air']
	else:
		raise ValueError(f"Unknown gas: {gas}")

	# Multiply out the polynomials.
	Tz = T_K / 1000

	# Eqn F3.23
	cp = sum([a_i * Tz ** i for i, a_i in enumerate(coeff_a[0:9], 0)])

	# Eqn F3.26
	h = coeff_a[9] + sum(
		[(a_i / i) * Tz ** i for i, a_i in enumerate(coeff_a[0:9], 1)])

	# Eqn F3.28
	EJW_s_A0_corr_term = coeff_a[0] * log(1000)
	s = ((coeff_a[0] * log(Tz)) + sum(
		[(a_i / i) * Tz ** i for i, a_i in enumerate(coeff_a[1:9], 1)]) +
	     coeff_a[10]) + EJW_s_A0_corr_term

	# Add FAR correction if required.
	if gas in ('burned_kero', 'burned_diesel') and FAR > 0:
		# 'B' coefficients for corrections due to kerosene / diesel products of
		# combustion.
		coeff_b = (
			-0.718874, 8.747481, -15.863157, 17.254096, -10.233795, 3.081778,
		-0.361112, -0.003919, 0.0555930, -0.0016079)

		# Eqn F3.24
		cp += (FAR / (1 + FAR)) * sum(
			[b_i * Tz ** i for i, b_i in enumerate(coeff_b[0:8], 0)])

		# Eqn F3.27
		h += (FAR / (1 + FAR)) * (sum(
			[(b_i / i) * Tz ** i for i, b_i in enumerate(coeff_b[0:7], 1)]) +
		                          coeff_b[8])

		# Eqn F3.29
		EJW_s_B0_corr_term = coeff_b[0] * log(1000)
		print(f"XXX TODO VERIFY FAR CORR TERM WITH EXAMPLES")
		s += (FAR / (1 + FAR)) * (coeff_b[0] * log(Tz) + sum(
			[(b_i / i) * Tz ** i for i, b_i in enumerate(coeff_b[1:8], 1)]) +
		                          coeff_b[9] + EJW_s_B0_corr_term)

	return cp, h, s


# noinspection NonAsciiCharacters,PyPep8Naming
def _vib_cp(T_K: float) -> float:
	"""Compute the cp value for air using a model of a simple  harmonic
	vibrator, ref NACA TN 1135 Eqn 175, 176.  This covers temperatures above
	the polynomial range limit of 2000 K.   In any case the vibrator and
	polynomial model are within about 0.5% everywhere under 2000 K.

	Input value is K, returns cp in kJ/kg/K.
	"""
	c_p_perf = 1.0057  # American Meter. Society [kJ/kg/K]
	γ_perf = 1.4
	temp_R = Dim(T_K, 'K').convert('°R').value
	ratio = 5500 / temp_R  # Ratio (Theta) = 5,500°R / T
	return c_p_perf * (1 + ((γ_perf - 1) / γ_perf) * (
			(ratio ** 2) * exp(ratio) / (exp(ratio) - 1) ** 2))
