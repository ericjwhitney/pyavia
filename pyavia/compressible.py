"""
Equations and constants relating to gases and compressible flow.

Contains:
	ComprFlow     Class giving fixed properties of a flowing compressible gas.
"""
# Last updated: 9 December 2019 by Eric J. Whitney

from math import exp, log

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
			- h and h_flow
			- s
		In this case these properties will return None.

		- Static (ambient) temperature and pressure are	used internally for
		convenience of cp and h calculation.

		- As expected, specific enthalpy has an arbitrary baseline and
		should never be used as a standalone value.
	"""

	# Magic methods -----------------------------------------------------------

	def __init__(self, *, P: Dim = None, Pt: Dim = None, T: Dim = None,
	             Tt: Dim = None, h: Dim = None, s: Dim = None, M: float,
	             w: Dim = None, gas: str = 'air', FAR: float = 0.0):
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
			w: (dim) Mass flowrate.
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
		self._M, self._w, self._gas, self._FAR = M, w, gas, FAR

		# Most other properties depend on T so this is set next.  Also try to
		# guess appropriate units system.
		self._output_units = 'SI'
		if T and not any([Tt, h, s]):
			# Straightforward case.
			self._T = make_total_temp(T)
			if self._T.units.θ[0] == '°R':
				self._output_units = 'US'
		elif Tt and not any([T, h, s]):
			# From stagnation.
			_Tt = make_total_temp(Tt)
			self._T = Dim(_get_T_from_Tt(_Tt.convert('K').value, self._M,
			                         self._gas, self._FAR), _Tt.units)
			if self._T.units.θ[0] == '°R':
				self._output_units = 'US'
		elif h and not any([T, Tt, s]):
			# From enthalpy.
			self._T = Dim(_get_T_from_h(h.convert('MJ/kg').value, self._gas,
			                        self._FAR), 'K')
			if h.units.θ[0] == '°R':
				self._T.convert('°R')
		elif s and not any([T, Tt, h]):
			# From entropy.
			self._T = Dim(_get_T_from_s(s.convert('kJ/kg/K').value, self._gas,
			                        self._FAR), 'K')
			if s.units.θ[0] == '°R':
				self._T.convert('°R')
		else:
			raise TypeError(f"Invalid temperature arguments.")

		# Precompute remaining properties.
		if self._output_units == 'US':
			c_R_units = 'Btu/lbm/°R'
			s_units = 'Btu/lbm/°R'
			h_units = 'Btu/lbm'
			a_units = 'ft/s'
		else:
			c_R_units = 'J/kg/K'
			s_units = 'kJ/kg/K'
			h_units = 'kJ/kg'
			a_units = 'm/s'

		T_K = self._T.convert('K').value
		self._cp = Dim(1000 * _get_cp_from_T(T_K, self._gas, self._FAR),
			               c_R_units)

		try:
			# These may not be settable, assign None in that case.
			self._h = Dim(1000 * _get_h_from_T(T_K, self._gas, self._FAR),
			              h_units)
			self._s = Dim(_get_s_from_T(T_K, self._gas, self._FAR),
			              s_units)
		except ValueError:
			self._h = None
			self._s = None

		self._R = Dim(_get_R_from_T(T_K, self._gas, self._FAR), c_R_units)
		self._γ = self._cp / (self._cp - self._R)
		self._a = ((self._γ * self._R * self._T)**0.5).convert(a_units)

		if P and not Pt:
			self._P = P
		elif Pt and not P:
			self._P = Pt / self.Pt_on_P
		else:
			raise TypeError(f"Invalid pressure arguments.")

	def __format__(self, format_spec) -> str:
		prop_list = []
		for p in self.props():
			if p == 'gas':
				fmt = 's'
			else:
				fmt = format_spec
			prop_list += [f'{p}={getattr(self, p):{fmt}}']
		return ', '.join(prop_list)

	def __repr__(self):
		return 'ComprFlow(' + ', '.join([f'{p}={repr(getattr(self, p))}'
		                                 for p in self.props()]) + ')'

	def __str__(self):
		return self.__format__('')

	# Normal methods ----------------------------------------------------------

	@staticmethod
	def props():
		"""Returns a set containing strings of all the independent
		parameters required to fully initialise the object."""
		return 'P', 'T', 'M', 'w', 'gas', 'FAR'

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
			if new_key in ('T', 'Tt', 'h', 's'):
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
	def cv(self) -> Dim:
		"""Specific heat capacity at constant volume of the gas."""
		return self._cp - self._R

	@property
	def Q(self) -> Dim:
		"""Enthalpy flowrate Q = h * w.  Inherits units from h."""
		return self._h * self._w

	@property
	def FAR(self) -> float:
		"""Fuel-Air Ratio."""
		return self._FAR

	@property
	def gamma(self) -> float:
		"""Ratio of specific heats γ = cp / cv."""
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
		return _Pt_on_P(self._γ, self._M)

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
		"""Entropy.  Arbitrary zero basis."""
		return self._s

	@property
	def Tt_on_T(self) -> float:
		"""Ratio of total (stagnation) pressure to static temperature."""
		return _Tt_on_T(self._γ, self._M)

	@property
	def Tt(self) -> Dim:
		"""Total / stagnation temperature."""
		return self.Tt_on_T * self.T

	@property
	def T(self) -> Dim:
		"""Stream / ambient temperature."""
		return self._T

	@property
	def w(self) -> Dim:
		"""Mass flowrate."""
		return self._w


# -----------------------------------------------------------------------------


# noinspection PyPep8Naming
def _get_T_from_h(h: float, gas: str, FAR: float) -> float:
	"""
	Computes stream temperature T [K] by bisecting between T = 200 K -> 2000 K
	for a given h [MJ/kg].  Valid for gas 'air', 'burned_kero',
	'burned_diesel'. FAR is fuel-air ratio.
	"""

	# noinspection PyPep8Naming
	def h_error(guess_T_K: float) -> float:
		return _get_h_from_T(guess_T_K, gas, FAR) - h

	return bisect_root(h_error, 200, 2000, max_its=50, f_tol=1e-6)


# noinspection PyPep8Naming
def _get_T_from_s(s: float, gas: str, FAR: float) -> float:
	"""
	Computes stream temperature T [K] by bisecting between T = 200 K -> 2000 K
	for a given s [kJ/kg/K].  Valid for gas 'air', 'burned_kero',
	'burned_diesel'. FAR is fuel-air ratio.
	"""

	# noinspection PyPep8Naming
	def s_error(guess_T_K: float) -> float:
		return _get_s_from_T(guess_T_K, gas, FAR) - s

	return bisect_root(s_error, 200, 2000, max_its=50, f_tol=1e-6)


# Walsh & Fletcher Eqn F3.23 coefficients.
_A_COEFF = {
	# Dry air with/without kerosene or diesel products of combustion.
	'air': (0.992313, 0.236688, -1.852148, 6.083152, -8.893933, 7.097112,
	        -3.234725, 0.794571, -0.081873, 0.422178, 0.001053), }

# 'B' coefficients for corrections due to kerosene / diesel products of
# combustion.
_COEFF_B = (
	-0.718874, 8.747481, -15.863157, 17.254096, -10.233795, 3.081778,
	-0.361112, -0.003919, 0.0555930, -0.0016079)


# noinspection PyPep8Naming
def _get_cp_from_T(T_K: float, gas: str, FAR: float) -> float:
	"""
	Computes cp [kJ/kg/K] for a given temperature [K] using either the
	interpolating polynomial method of Walsh & Fletcher ('air',
	'burned_kero', 'burned_diesel') or a simple harmonic vibrator model
	('air' only).  FAR is fuel-air ratio (polynomial method	only).

	The simple harmonic	vibrator is per NACA TN 1135 Eqn 175, 176.  This
	covers temperatures above the polynomial range limit of 2000 K.   In any
	case the vibrator and polynomial model are within about 0.5% everywhere
	under 2000 K.

	Raises ValueError if cp cannot be computed.
	"""
	try:
		# Polynomial model.
		if not (200 <= T_K <= 2000):
			raise ValueError
		if gas in ('air', 'burned_kero', 'burned_diesel'):
			coeff_a = _A_COEFF['air']
		else:
			raise ValueError

		# Multiply out the polynomial, W&F Eqn F3.23.
		Tz = T_K / 1000
		cp = sum([a_i * Tz ** i for i, a_i in enumerate(coeff_a[0:9], 0)])

		# Add FAR correction if required.
		if gas in ('burned_kero', 'burned_diesel') and FAR > 0:
			# W&F Eqn F3.24
			cp += (FAR / (1 + FAR)) * sum(
				[b_i * Tz ** i for i, b_i in enumerate(_COEFF_B[0:8], 0)])

		return cp

	except ValueError:
		pass

	try:
		# Vibrator model.
		if not gas == 'air' and T_K > 2000:
			raise ValueError

		c_p_perf = 1.0057  # American Meter. Society [kJ/kg/K]
		γ_perf = 1.4
		ratio = 5500 / (T_K * 9/5)  # Ratio (Theta) = 5,500°R / T [°R]
		return c_p_perf * (1 + ((γ_perf - 1) / γ_perf) * (
				(ratio ** 2) * exp(ratio) / (exp(ratio) - 1) ** 2))

	except ValueError:
		pass

	raise ValueError(f"Cannot determine cp for gas '{gas}' at temperature "
	                 f"{T_K}K")


# noinspection PyPep8Naming
def _get_h_from_T(T_K: float, gas: str, FAR: float) -> float:
	"""
	Computes h [MJ/kg] for a given temperature [K] using the interpolating
	polynomial method of Walsh & Fletcher.  Valid for gas 'air',
	'burned_kero', 'burned_diesel'.  FAR is fuel-air ratio.

	Raises ValueError if h cannot be computed.
	"""
	# Applicable range.
	if not (200 <= T_K <= 2000):
		raise ValueError(f"T = {T_K} K outside allowable range.")

	if gas in ('air', 'burned_kero', 'burned_diesel'):
		coeff_a = _A_COEFF['air']
	else:
		raise ValueError(f"Unknown gas: {gas}")

	# Multiply out the polynomials, W&F Eqn F3.26
	Tz = T_K / 1000
	h = coeff_a[9] + sum(
		[(a_i / i) * Tz ** i for i, a_i in enumerate(coeff_a[0:9], 1)])

	# Add FAR correction if required.
	if gas in ('burned_kero', 'burned_diesel') and FAR > 0:
		# W&F Eqn F3.27.
		h += (FAR / (1 + FAR)) * (sum(
			[(b_i / i) * Tz ** i for i, b_i in enumerate(_COEFF_B[0:7], 1)]) +
		                          _COEFF_B[8])
	return h


# noinspection PyPep8Naming
def _get_R_from_T(T_K: float, gas: str, FAR: float) -> float:
	"""
	Computes R [J/kg/K] for a given temperature [K] and FAR using the
	method of
	Walsh & Fletcher (200 <= T_K <= 200) or as a constant for air.

	Raises ValueError if R cannot be computed.
	"""
	R_air = 287.05287  # J/kg/K.
	if gas == 'air':
		return R_air

	if 200 <= T_K <= 2000:
		if gas == 'burned_kero':
			return R_air - 0.00990 * FAR + 1e-7 * FAR ** 2

		if gas == 'burned_diesel':
			return R_air - 8.0262 * FAR + 3e-7 * FAR ** 2

	raise ValueError(f"Cannot determine R for gas '{gas}' at temperature "
	                 f"{T_K} K")


# noinspection PyPep8Naming
def _get_s_from_T(T_K: float, gas: str, FAR: float) -> float:
	"""
	Computes s [kJ/kg/K] for a given temperature [K] using the interpolating
	polynomial method of Walsh & Fletcher.  Valid for gas in 'air',
	'burned_kero', 'burned_diesel'.  FAR is fuel-air ratio.

	Raises ValueError if s cannot be computed.
	"""
	# Applicable range.
	if not (200 <= T_K <= 2000):
		raise ValueError(f"T = {T_K} K outside allowable range.")

	if gas in ('air', 'burned_kero', 'burned_diesel'):
		coeff_a = _A_COEFF['air']
	else:
		raise ValueError(f"Unknown gas: {gas}")

	# Multiply out the polynomials, W&F Eqn F3.28 with EJ correction term.
	Tz = T_K / 1000
	EJW_s_A0_corr_term = coeff_a[0] * log(1000)
	s = ((coeff_a[0] * log(Tz)) + sum(
		[(a_i / i) * Tz ** i for i, a_i in enumerate(coeff_a[1:9], 1)]) +
	     coeff_a[10]) + EJW_s_A0_corr_term

	# Add FAR correction if required.
	if gas in ('burned_kero', 'burned_diesel') and FAR > 0:
		# W&F Eqn F3.29
		EJW_s_B0_corr_term = _COEFF_B[0] * log(1000)
		# XXX TODO VERIFY FAR CORR TERM WITH EXAMPLES
		s += (FAR / (1 + FAR)) * (_COEFF_B[0] * log(Tz) + sum(
			[(b_i / i) * Tz ** i for i, b_i in enumerate(_COEFF_B[1:8], 1)]) +
		                          _COEFF_B[9] + EJW_s_B0_corr_term)
	return s


# noinspection PyPep8Naming
def _get_T_from_Tt(Tt_K: float, M: float, gas: str, FAR: float) -> float:
	"""Return the stream temperature [K] of the gas given the total
	(stagnation) temperature, Mach number, gas and FAR via iteration.  This is
	due to the non-linear dependence of γ on Ts."""
	if M == 0:  # Already at stagnation.
		return Tt_K

	# Iterate temperature due to non-linear dependence of γ on Ts.
	# Start iteration with Ts = T.
	# noinspection PyPep8Naming
	def new_Ts(guess_Ts):
		cp = _get_cp_from_T(guess_Ts, gas, FAR)
		R = _get_R_from_T(guess_Ts, gas, FAR)
		γ = cp / (cp - R)
		return Tt_K / _Tt_on_T(γ, M)

	return iterate_fn(new_Ts, x_start=Tt_K, x_tol=1e-6)


# noinspection PyPep8Naming
def _Pt_on_P(γ: float, M: float) -> float:
	"""Ratio of total (stagnation) pressure to static pressure at a
	given γ and Mach number."""
	return (1 + 0.5 * (γ - 1) * M ** 2) ** (γ / (γ - 1))


# noinspection PyPep8Naming
def _Tt_on_T(γ: float, M: float) -> float:
	"""Ratio of total (stagnation) pressure to static temperature at a
	given γ and Mach number."""
	return 1 + 0.5 * (γ - 1) * M ** 2
