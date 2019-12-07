"""
Equations and constants relating to gases and compressible flow.

Contains:
	ComprFlow     Class giving fixed properties of a flowing compressible gas.
"""
# Last updated: 7 December 2019 by Eric J. Whitney

from math import exp
from typing import Set

from units import Dim, make_total_temp
from util import iterate_fn, bisect_root

__all__ = ['ComprFlow']


# -----------------------------------------------------------------------------


# noinspection PyPep8Naming,NonAsciiCharacters
class ComprFlow:

	"""
	Class representing a flowing compressible gas.  Once initialised,
	properties are fixed.

	Notes:
		- Cp and γ vary with temperature.  For higher temperatures the
		interpolating polynomial method of Walsh & Fletcher Chapter 3 is
		used to calculate Cp and h.
		- Static (ambient) temperature and pressure are	used internally for
		convenience of Cp and h calculation.
		- As expected, specific enthalpy has an arbitrary baseline and
		should never be used as a standalone value.
	"""
	__slots__ = ('_P', '_T', '_M', '_W', '_gas', '_FAR')

	# Magic methods.

	def __init__(self, *, P: Dim = None, Pt: Dim = None, T: Dim = None,
	             Tt: Dim = None, h: Dim = None, M: float, W: Dim = None,
	             gas: str = 'air', FAR: float = 0.0):
		"""
		Constructs a representation of a compressible gas flowing in 1-D.
		Notes:
			- Set either T, Tt or h (note the h baseline is arbitrary;
			refer to formulation used).
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
		self._M, self._W, self._FAR = M, W, FAR

		if gas not in {'air', 'burned_kero', 'burned_diesel'}:
			raise ValueError(f"Unknown gas: {gas}")
		self._gas = gas

		# T, P set via properties so that any internal states
		# can be updated / checks made.

		self._P, self._T = 0, 0   # Dummies required for Tt / h functions.
		if T and not Tt and not h:
			self._T = T
		elif Tt and not T and not h:
			self._T = self._get_T_from_Tt(Tt)
		elif h and not T and not Tt:
			self._T = self._get_T_from_h(h)
		else:
			raise TypeError(f"Invalid temperature arguments.")

		if P and not Pt:
			self._P = P
		elif Pt and not P:
			self._P = Pt / self.Pt_on_P
		else:
			raise TypeError(f"Invalid pressure arguments.")

	# Normal methods.

	def props(self) -> Set[str]:
		"""Returns a set containing strings of all the directly stored gas
		properties, i.e. {'P', 'T', 'M', ...)"""
		return {_key[1:] for _key in self.__slots__}

	def replace(self, **kwargs) -> 'ComprFlow':
		"""
		Makes a copy of this object except with nominated parameters replaced
		by the given values, e.g. similar to a partial __init__.

		Args:
			**kwargs: As per __init__.

		Returns:
			New ComprFlow object.
		"""
		# Get properties directly from internal values.
		new_kwargs = {_key: getattr(self, _key) for _key in self.__slots__}

		# Strip underscores.
		new_kwargs = {_key[1:]: val for _key, val in new_kwargs.items()}

		for new_key, value in kwargs.items():
			if new_key in ('P', 'Pt'):
				new_kwargs.pop('P', None)
			if new_key in ('T', 'Tt', 'h'):
				new_kwargs.pop('T', None)
			new_kwargs[new_key] = value

		return ComprFlow(**new_kwargs)

	# Properties.

	@property
	def Cp(self) -> Dim:
		"""
		Specific heat capacity at constant pressure of the gas.  The
		interpolating polynomial method of Walsh & Fletcher Eqn F3.23,
		F3.24 is used.

		Note:  Very hot cases for air are permitted above the polynomial
		range limit of 2000 K.  In this case the Cp value for air is
		computed using a model of a simple  harmonic vibrator, ref NACA TN
		1135 Eqn 175, 176.  In any case the vibrator and polynomial model are
		within about 0.5% everywhere under 2000 K.

		Returns:
			Dim() object with Cp in J/kg/K (or Btu/lbm/°R).

		Raises:
			ValueError if the temperature is outside the allowable range.
		"""
		# Set gas properties.
		T_K = self._T.convert('K').value
		T_Range_K = _T_RANGE_K[self._gas]
		if self._gas in ('air', 'burned_kero', 'burned_diesel'):
			coeff_a = _EQN_COEFF['F3.23_A_Air'][0:9]  # Only A0..A8
		else:
			raise ValueError(f"Unknown gas when computing Cp: {self._gas}")

		# Check in valid range.
		if not (T_Range_K[0] <= T_K <= T_Range_K[1]):
			if T_K > T_Range_K[1] and self._gas == 'air':
				# Special high temperature case for air.
				c_p_perf = Dim(1005.7, 'J/kg/K')  # American Meter. Society
				γ_perf = 1.4
				temp_R = self._T.convert('°R').value
				ratio = 5500 / temp_R  # Ratio (Theta) = 5,500°R / T
				return c_p_perf * (1 + ((γ_perf - 1) / γ_perf) * (
						(ratio ** 2) * exp(ratio) / (exp(ratio) - 1) ** 2))
			else:
				raise ValueError(f"Temperature {T_K:.1f} K out of range "
				                 f" for Cp in {self._gas}. Allowable range "
				                 f"is {T_Range_K[0]:.1f} -> "
				                 f"{T_Range_K[1]:.1f}")

		# Multiply out the A polynomial.
		Tz = T_K / 1000
		c_p_val = sum([a_i * Tz ** i for i, a_i in enumerate(coeff_a)])

		# Add FAR correction if required.
		if self._gas in ('burned_kero', 'burned_diesel'):
			# Multiply out the B polynomial.
			coeff_b = _EQN_COEFF['F3.24_B'][0:8]  # Only B0..B7
			b_poly = sum([b_i * Tz ** i for i, b_i in enumerate(coeff_b)])
			c_p_val += (self.FAR / (1 + self.FAR)) * b_poly

		c_p_res = Dim(c_p_val * 1000, 'J/kg/K')  # x1000 to get J/kg/K
		if self._T.units.θ[0] == '°R':  # Try and match intent.
			return c_p_res.convert('Btu/lbm/°R')
		else:
			return c_p_res

	@property
	def Cv(self) -> Dim:
		"""Specific heat capacity at constant volume of the gas."""
		return self.Cp - self.R

	@property
	def FAR(self) -> float:
		"""Fuel-Air Ratio."""
		return self._FAR

	@property
	def gamma(self) -> float:
		"""Ratio of specific heats γ = Cp / c_v."""
		return self.Cp / (self.Cp - self.R)

	@property
	def gas(self) -> str:
		return self._gas

	@property
	def h(self) -> Dim:
		"""
		Specific enthalpy of the gas.  The interpolating polynomial method
		of Walsh & Fletcher Eqn F3.26, F3.27 is used.

		Returns:
			Dim() object with H in MJ/kg (or Btu/lbm).

		Raises:
			ValueError if the temperature is outside the allowable range.
		"""
		# Set dry gas properties.
		Ts_K = self._T.convert('K').value
		Ts_Range_K = _T_RANGE_K[self._gas]
		if self._gas in ('air', 'burned_kero', 'burned_diesel'):
			coeff_a = list(_EQN_COEFF['F3.23_A_Air'][0:10])  # A0..A9
			A9 = coeff_a.pop()  # Leaves only A0..A8
		else:
			raise ValueError(f"Unknown gas when computing H: {self._gas}")

		# Check in valid range.
		if not (Ts_Range_K[0] <= Ts_K <= Ts_Range_K[1]):
			raise ValueError(f"Temperature {Ts_K:.1f} K out of range for "
			                 f"H in {self._gas}. Allowable range is"
			                 f"{Ts_Range_K[0]:.1f} -> {Ts_Range_K[1]:.1f}")

		# Multiply out the polynomial.
		Tz = Ts_K / 1000
		h_val = A9 + sum([(a_i / i) * Tz ** i for i, a_i in
		                  enumerate(coeff_a, 1)])
		# Note divisors and Tz powers are all +1 compared to Cp.

		# Add FAR correction if required.
		if self._gas in ('burned_kero', 'burned_diesel'):
			# Multiply out the B polynomial.
			coeff_b = list(_EQN_COEFF['F3.24_B'][0:9])  # B0..B8
			B8 = coeff_b.pop()  # Leaves only B0..B7
			b_poly = B8 + sum([(b_i / i) * Tz ** i for i, b_i in
			                   enumerate(coeff_b, 1)])
			# Again divisors and Tz powers are all +1 compared to Cp.
			h_val += (self.FAR / (1 + self.FAR)) * b_poly

		h_res = Dim(h_val, 'MJ/kg')  # Equation gives MJ/kg.
		if self._T.units.θ[0] == '°R':  # Try and match intent.
			return h_res.convert('Btu/lbm')
		else:
			return h_res

	@property
	def M(self) -> float:
		"""Mach number."""
		return self._M

	@property
	def Pt_on_P(self) -> float:
		"""Ratio of total (stagnation) pressure to static pressure."""
		_γ = self.gamma  # One @property call.
		return (1 + 0.5 * (_γ - 1) * self._M ** 2) ** (_γ / (_γ - 1))

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
		"""Gas constant R [J/kg/K]."""
		_R_air = 287.05287  # J/kg/K.
		if self._gas == 'air':
			_R = _R_air
		elif self._gas == 'burned_kero':
			_R = _R_air - 0.00990 * self.FAR + 1e-7 * self.FAR ** 2
		elif self._gas == 'burned_diesel':
			_R = _R_air - 8.0262 * self.FAR + 3e-7 * self.FAR ** 2
		else:
			raise ValueError(f"Unknown gas getting R value: {self._gas}")
		R_res = Dim(_R, 'J/kg/K')
		if self._T.units.θ[0] == '°R':  # Try and match units to intent.
			return R_res.convert('Btu/lbm/°R')
		else:
			return R_res

	@property
	def Tt_on_T(self) -> float:
		"""Ratio of total (stagnation) pressure to static temperature."""
		_γ = self.gamma  # One @property call.
		return 1 + 0.5 * (_γ - 1) * self._M ** 2

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

	def _get_T_from_h(self, value: Dim) -> Dim:
		"""Find the stream temperature of the gas given the specific
		enthalpy, using a bisection method.  The initial T range is assumed
		to be the full range of the gas."""

		def h_error(T_try: Dim) -> Dim:
			return self.replace(P=Dim(0, 'kPa'), T=T_try).h - value

		T_L = Dim(_T_RANGE_K[self.gas][0], 'K')
		T_R = Dim(_T_RANGE_K[self.gas][1], 'K')
		return bisect_root(h_error, T_L, T_R, maxits=50,
		                   tol=Dim(1, 'J/kg'))  # Approx tol=1e-6

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
			flow = ComprFlow(P=Dim(0, 'kPa'), T=try_Ts, M=self._M,
			                 W=self._W, gas=self._gas, FAR=self._FAR)
			return Tt_reqd / flow.Tt_on_T

		return iterate_fn(new_Ts, x_start=Tt_reqd, xtol=Dim(1e-6, 'K'))

# -----------------------------------------------------------------------------


# Coefficiencts A0, A1, ... or B0, ... etc for polynomial fit equations in
# Walsh & Fletcher.
_EQN_COEFF = {
	# 'A' coefficients for dry air with/without kerosene or diesel products
	# of combustion.
	'F3.23_A_Air': (0.992313, 0.236688, -1.852148, 6.083152, -8.893933,
	               7.097112, -3.234725, 0.794571, -0.081873, 0.422178,
	                 0.001053),

	# 'B' coefficients for corrections due to kerosene / diesel products of
	# combustion.
	'F3.24_B': (-0.718874, 8.747481, -15.863157, 17.254096, -10.233795,
	              3.081778, -0.361112, -0.003919, 0.0555930, -0.0016079),
}

# Valid region for temperature polynomials (K).
_T_RANGE_K = {
	'air': (200, 2000),
	'burned_kero': (200, 2000),
	'burned_diesel': (200, 2000)
}
