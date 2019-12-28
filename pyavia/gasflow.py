"""
Equations and constants relating to gases and compressible flow.

Contains:
    GasFlow         Abstract class for compressible gas flows.
    GasFlowWF       Compressible gas, calorically imperfect, opt. fuel-air.
    PerfectGasFlow  Compressible gas, calorically perfect.
"""
# Last updated: 18 December 2019 by Eric J. Whitney

from abc import ABC, abstractmethod
from math import log, exp
from typing import Sequence

from units import Dim, make_total_temp
from util import fixed_point

__all__ = ['GasFlow', 'GasFlowWF', 'PerfectGasFlow']


# ----------------------------------------------------------------------------

# noinspection PyPep8Naming
class GasFlow(ABC):
    """
    Abstract base class representing common properties of gases flowing in
    1-D. Once any gas is initialised, properties are fixed.

    Note: This base class does not provide properties such as T0/T or P0/T
    because non-linearities in imperfect gas models may make these ratios
    moot.  Refer to the specific derived class.
    """

    # Magic methods ----------------------------------------------------------

    def __init__(self, *, R, w=None, gas: str = None):
        """
        Construct an abstract base representation of flowing compressible
        gas.  Once constructed, the properties of the gas are fixed.

        Args:
            R: (Opt Type) Gas constant for this particular gas (i.e. R =
            R_universal / n).
            w: (Opt Type) Mass flowrate
        """
        self._R, self._w, self._gas = R, w, gas

    def __format__(self, format_spec) -> str:
        return type(self).__name__ + ': ' + self._fmt_str(format_spec)

    def __repr__(self):
        return type(self).__name__ + '(' + ', '.join(
            [f'{p}={repr(getattr(self, p))}' for p in self.min_args()]) + ')'

    def __str__(self):
        return self.__format__('.5G')

    # Normal methods / properties --------------------------------------------

    @property
    def a(self):
        """Local speed of sound a = (γRT)**0.5."""
        return (self.gamma * self._R * self.T) ** 0.5

    @property
    def cv(self):
        """Specific heat capacity at constant volume of the gas."""
        return self.cp - self.R

    def _fmt_str(self, format_spec) -> str:
        """Generate formatted string of minimum properties for use by
        __format__, __str__, __repr__"""
        prop_list = []
        for p in self.min_args():
            val = getattr(self, p)
            if p == 'gas':
                fmt = 's'
            elif val is None:
                fmt = ''
            else:
                fmt = format_spec
            prop_list += [f'{p}={val:{fmt}}']
        return ', '.join(prop_list)

    @property
    def gas(self) -> str:
        """Identifier for the gas."""
        return self._gas

    @property
    def h0(self):
        """Total / stagnation enthalpy of the gas, assuming it is brought to
        rest with losses or heat transfer computed as h_0 = h + (1/2)*u**2.
        Like all enthalpy values the baseline is arbitrary and values from
        different formulations / classes should not be compared."""
        return self.h + 0.5 * self.u ** 2

    @property
    def Q(self):
        """Enthalpy flowrate Q = h * w.  Units determined by
        multiplication."""
        return self.h * self._w

    @property
    def R(self):
        """Gas constant R for the specific gas. R_specific = R_universal / M
        where R = 8.314462618 kg.m².s⁻².K⁻¹.mol⁻¹ and M is molar mass
        [kg/mol]. """
        return self._R

    def replace(self, **kwargs) -> 'GasFlow':
        """Return a GasFlow object initialised to match self except for
        keyword arguments from **kwargs.  If a new temperature or pressure
        is specified, a proper pair of P, T, h or s properties must be
        supplied (refer __init__ for the actual class)."""
        # Build keywords.
        cls = type(self)
        new_kwargs = {k: getattr(self, k) for k in cls.min_args()}

        # If new P, T, h, s type argument, clear existing P, T.
        if any([k in ('P', 'T', 'P0', 'T0', 'h', 's') for k in kwargs.keys()]):
            new_kwargs.pop('P', None)
            new_kwargs.pop('T', None)

        # Add / overwrite new args.
        for k, v in kwargs.items():
            new_kwargs[k] = v

        return cls(**new_kwargs)

    @property
    def w(self):
        """Mass flowrate."""
        return self._w

    # Abstract methods / properties - Override -------------------------------

    @property
    @abstractmethod
    def cp(self):
        """Specific heat capacity at constant pressure of the gas."""
        pass

    @property
    @abstractmethod
    def gamma(self) -> float:
        """Ratio of specific heats γ = cp / cv."""
        pass

    @property
    @abstractmethod
    def h(self):
        """Specific enthalpy of the gas. Note: The baseline is arbitrary and
        values from different formulations / classes should not be compared.
        """
        pass

    @property
    @abstractmethod
    def M(self) -> float:
        """Mach number."""
        pass

    @classmethod
    @abstractmethod
    def min_args(cls) -> Sequence[str]:
        """Return a sequence of strings giving a minimum list of keywords /
         properties required to fully initialise the object.  Used by
         __repr__, __str__, __format__, replace()."""
        pass

    @property
    @abstractmethod
    def P(self):
        """Stream / ambient pressure."""
        pass

    @property
    @abstractmethod
    def P0(self):
        """Total / stagnation pressure."""
        pass

    @property
    @abstractmethod
    def s(self):
        """Specific entropy of the gas.  Note: The baseline is arbitrary and
        values from different formulations / classes should not be compared.
        """
        pass

    @property
    @abstractmethod
    def T(self):
        """Stream / ambient temperature."""
        pass

    @property
    @abstractmethod
    def T0(self):
        """Total / stagnation temperature."""
        pass

    @property
    @abstractmethod
    def u(self):
        """Flow velocity."""
        pass

# ----------------------------------------------------------------------------


# noinspection PyPep8Naming,NonAsciiCharacters
class GasFlowWF(GasFlow):
    """
    Class representing airflow (with optional products of combustion) as a
    compressible, imperfect gas, based on interpolating polynomials in Walsh
    & Fletcher Chapter 3 and are fixed once initialised.

    Notes:
        - Temperature range is 200 K -> 2000 K.
        - Fuel-Air ratio 0.00 -> 0.05
        - Non-linear γ, cp, h, s dependent on temperature (i.e.
        calorically imperfect).  Stagnation enthalpy, temperature, pressure
        do not assume a perfect gas.
        - Uses Dim values. Internal values stored as SI [K, kPa, ...].
        - Stream / static temperature and pressure and Mach number are
        internal reference states.
    """

    # Magic methods ----------------------------------------------------------

    def __init__(self, *, P: Dim = None, P0: Dim = None, T: Dim = None,
                 T0: Dim = None, h: Dim = None, h0: Dim = None, s: Dim = None,
                 M: float = 0, w: Dim = None, gas: str = 'air',
                 FAR: float = 0.0):

        """
        Construct a compressible, imperfect gas, based on interpolating
        polynomials given in Walsh & Fletcher, "Gas Turbine Performance",
        Second Edition, Chapter 3.

        Notes:
            - Arguments must combine one temperature-like and one
            pressure-like:
                * T-like: T, T0, h, h0, s.
                * P-like: P, P0 or s.
            - Mach number must have the correct value for initialisation
            to give correct stagnation values.
            - Non-zero FAR normally accompanies gases including products of
            combustion but this is not enforced.  This is because this
            combinatinon may be useful in some hypothetical or debugging
            scenarios.

        Args:
            P: (Dim) Stream / static pressure.
            P0: (Dim) Total / stagnation pressure (also P0).
            T: (Dim) Stream / static temperature.
            T0: (Dim) Total /stagnation temperature (also T0).
            h: (Dim) Specific enthalpy.
            h0: (Dim) Total / stagnation enthalpy.
            s: (Dim) Specific entropy.
            M: (float) Mach number.
            w: (dim) Mass flowrate.
            gas: (str) identifier (default = 'air').  Available 'gas' values
                are:
                    air             Dry air
                    burned_kerosene Products of combustion for kerosene in
                                    dry air
                    burned_diesel   Products of combustion for diesel in
                                    dry air
            FAR: (float) Fuel-Air Ratio (default = 0.0)

        Raises:
            ValueError on invalid parameters.  TypeError on invalid arguments.
        """
        self._M, self._FAR = M, FAR
        if not (0 <= self._FAR <= 0.05):
            raise ValueError(f"Fuel-Air Ratio invalid: {self._FAR}")

        # Set R, simple fixed values and gas model coefficients first.
        self._coeff_a, self._coeff_b = None, None
        if gas in ('air', 'burned_kerosene', 'burned_diesel'):
            R = 287.05287  # J/kg/K.
            self._coeff_a = _WF_A_COEFF['air']

            if gas == 'burned_kerosene':
                R += -0.00990 * self._FAR + 1e-7 * self._FAR ** 2
                self._coeff_b = _WF_B_COEFF

            elif gas == 'burned_diesel':
                R += -8.0262 * self._FAR + 3e-7 * self._FAR ** 2
                self._coeff_b = _WF_B_COEFF

            R = Dim(R, 'J/kg/K')
        else:
            raise ValueError(f"Invalid gas: {gas}")

        super().__init__(R=R, w=w, gas=gas)

        # Initialise lazily evaluted internal parameters.
        self._cp, self._gamma, self._h, self._cptint = [None] * 4
        self._T0, self._P0 = [None] * 2

        # Set temperature first from T, T0, h, h0 or s.
        approx_cp = Dim(1005, 'J/kg/K')
        if T and not any([T0, h, h0]):
            self._T = make_total_temp(T)
            refine_T = None

        elif T0 and not any([T, h, h0]):
            self._P = Dim(1, 'bar')  # Dummy for _set_stagnation()

            def refine_T(try_T):
                self._lazy_reset()
                self._T = try_T
                return (T0 - self.T0) + try_T

        elif h and not any([T, T0, h0]):
            def refine_T(try_T):
                self._lazy_reset()
                self._T = try_T
                return (h - self.h) / approx_cp + try_T

        elif h0 and not any([T, T0, h]):
            def refine_T(try_T):
                self._lazy_reset()
                self._T = try_T
                return (h0 - self.h0) / approx_cp + try_T

        elif s and not any([T, T0, h, h0]):
            if P:
                self._P = P  # Required for entropy computation.
            else:
                raise TypeError(f"Entropy to set T also requires P argument.")

            def refine_T(try_T):
                self._lazy_reset()
                self._T = try_T
                # dT = (T/cp).ds for dp = 0.
                return (s - self.s) * try_T / approx_cp + try_T

        else:
            raise TypeError(f"Invalid temperature-type argument.")

        if refine_T:
            fixed_point(refine_T, x0=Dim(288.15, 'K'), xtol=Dim(1e-6, 'K'))

        # Set pressure last from P, P0 or s
        if P and not P0:
            # This also resets P when s is used to set temperature.
            self._P = P

        elif P0 and not P and not s:
            def refine_P(try_P):
                self._lazy_reset()
                self._P = try_P
                return (P0 - self.P0) + try_P

            # Softer relaxation to stabilise stagnation pressure.
            fixed_point(refine_P, x0=Dim(1, 'bar'), xtol=Dim(1e-6, 'bar'),
                        relax=0.5)

        elif s and not P and not P0:
            self._cptint = self._cptint_from_T(self._T)
            # P = P_ref * exp(int(cp/T).dT - s) / R
            self._P = _WF_P_REF_S * exp((self._cptint - s) / self.R)

        else:
            raise TypeError(f"Invalid pressure-type argument.")

        if not (Dim(200, 'K') <= self._T <= Dim(2000, 'K')):
            raise ValueError(f"Out of temperature range: {self._T}")
        if float(self._P) < 0:
            raise ValueError(f"Cannot set negative P: {self._P}")

    # Normal Methods / Properties --------------------------------------------

    @property
    def cp(self) -> Dim:
        if self._cp is None:
            self._cp = self._cp_from_T(self._T)
        return self._cp

    @property
    def FAR(self) -> float:
        """Fuel-Air Ratio FAR = w_f / w_total where w_f is the massflow of
        fuel products of combustion and w_total is the total massflow.  E.G.
        If the upstream flow was pure airflow of w_air then FAR = w_f / (w_f
        + w_air) """
        return self._FAR

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            self._gamma = self.cp / (self.cp - self.R)
        return self._gamma

    @property
    def h(self) -> Dim:
        if self._h is None:
            self._h = self._h_from_T(self._T)
        return self._h

    @property
    def M(self) -> float:
        return self._M

    @classmethod
    def min_args(cls):
        return 'P', 'T', 'M', 'w', 'gas', 'FAR'

    @property
    def P(self) -> Dim:
        return self._P

    @property
    def P0(self) -> Dim:
        if self._P0 is None:
            self._set_stagnation()
        return self._P0

    @property
    def s(self) -> Dim:
        if self._cptint is None:
            self._cptint = self._cptint_from_T(self._T)
        # s = int(cp/T).dT - R.ln(P/P_ref)
        return self._cptint - self.R * log(self._P / _WF_P_REF_S)

    @property
    def T(self) -> Dim:
        return self._T

    @property
    def T0(self) -> Dim:
        if self._T0 is None:
            self._set_stagnation()
        return self._T0

    @property
    def u(self) -> Dim:
        return self.a * self._M

    # Protected --------------------------------------------------------------

    def _cp_from_T(self, T: Dim) -> Dim:
        """Compute cp [kJ/kg/K] using W&F Eqn F3.23, given T and return
        J/kg/K.  self._coeff_a and self._coeff_b must be set prior to call."""
        Tz = T.convert('K').value / 1000
        cp = sum(
            [a_i * Tz ** i for i, a_i in enumerate(self._coeff_a[0:9], 0)])
        if self._coeff_b is not None:
            cp += (self._FAR / (1 + self._FAR)) * sum(
                [b_i * Tz ** i for i, b_i in
                 enumerate(self._coeff_b[0:8], 0)])
        return Dim(1000 * cp, 'J/kg/K')

    def _cptint_from_T(self, T: Dim) -> Dim:
        """Compute integral(cp/T) [kJ/kg/K] using W&F Eqn F3.28 with EJW
        correction term given T, return kJ/kg/K. self._coeff_a and
        self._coeff_b must be set prior to call."""
        Tz = T.convert('K').value / 1000
        EJW_A0_corr_term = self._coeff_a[0] * log(1000)
        cptint = ((self._coeff_a[0] * log(Tz)) + sum(
            [(a_i / i) * Tz ** i for i, a_i in
             enumerate(self._coeff_a[1:9], 1)]) + self._coeff_a[
                      10]) + EJW_A0_corr_term
        if self._coeff_b is not None:
            EJW_B0_corr_term = self._coeff_b[0] * log(1000)
            # XXX TODO VERIFY FAR CORR TERM WITH EXAMPLES
            cptint += (self._FAR / (1 + self._FAR)) * (
                    self._coeff_b[0] * log(Tz) +
                    sum([(b_i / i) * Tz ** i for i, b_i in
                         enumerate(self._coeff_b[1:8], 1)]) + self._coeff_b[
                        9] + EJW_B0_corr_term)
        return Dim(cptint, 'kJ/kg/K')

    def _h_from_T(self, T: Dim) -> Dim:
        """Compute enthalpy [MJ/kg] using W&F Eqn F3.26, given T and return
        kJ/kg/K.  self._coeff_a and self._coeff_b must be set prior to call."""
        Tz = T.convert('K').value / 1000
        h = self._coeff_a[9] + sum([(a_i / i) * Tz ** i for i, a_i in
                                    enumerate(self._coeff_a[0:9], 1)])
        if self._coeff_b is not None:
            h += (self._FAR / (1 + self._FAR)) * (sum(
                [(b_i / i) * Tz ** i for i, b_i in
                 enumerate(self._coeff_b[0:7], 1)]) + self._coeff_b[8])
        return Dim(1000 * h, 'kJ/kg')

    def _lazy_reset(self):
        self._cp, self._gamma, self._h, self._cptint = [None] * 4
        self._T0, self._P0 = [None] * 2

    def _set_stagnation(self) -> None:
        stopped_flow = self.replace(h=self.h0, s=self.s, M=0)
        self._T0, self._P0 = stopped_flow.T, stopped_flow.P


_WF_P_REF_S = Dim(1, 'bar')  # Reference pressure for entropy.

# Walsh & Fletcher Eqn F3.23 coefficients.
_WF_A_COEFF = {
    # Dry air with/without kerosene or diesel products of combustion.
    'air': (0.992313, 0.236688, -1.852148, 6.083152, -8.893933, 7.097112,
            -3.234725, 0.794571, -0.081873, 0.422178, 0.001053), }

# 'B' coefficients for corrections due to kerosene / diesel products of
# combustion.
_WF_B_COEFF = (
    -0.718874, 8.747481, -15.863157, 17.254096, -10.233795, 3.081778,
    -0.361112, -0.003919, 0.0555930, -0.0016079)


# ----------------------------------------------------------------------------


# noinspection PyPep8Naming
class PerfectGasFlow(GasFlow):
    """
    Class representing a simplified compressible gas which is thermally and
    calorically perfect. i.e. cp and R are constant.  Once initialised,
    properties are fixed.

    Notes:
        - Internally, stream pressure, temperature, Mach number and ratio of
         specific heats are the base flow properties (P, T, M, γ).
        - Sepcific enthalpy computation is h = h0 + cp.(T - T_ref)
        - Specific entropy computation is s = s0 + cp.ln(T / T_ref) - self.R *
            ln(P / P_ref).
    """
    # Magic methods ----------------------------------------------------------
    def __init__(self, *, P: Dim = None, P0: Dim = None, T: Dim = None,
                 T0: Dim = None, h: Dim = None, s: Dim = None, M: float = 0,
                 w: Dim = None, gas: str = 'air', gamma: float = 1.4):
        """
        Construct a thermally and calorifically perfect, compressible gas
        flowing in 1-D.

        Args:
            P0: (Dim) Total / stagnation pressure (also P0).
            T0: (Dim) Total /stagnation temperature (also T0).
            P: (Dim) Stream / static pressure.
            T: (Dim) Stream / static temperature.
            h: (Dim) Specific enthalpy.
            s: (Dim) Specific entropy.
            M: (float) Mach number.
            w: (dim) Mass flowrate.
            gas: (str) Gas type.  Supported values are: air.
            gamma: (float) Ratio of specific heats.  Commonly used
            values for air are:
                γ = 1.4     Atmospheric air, compressors (default).
                γ = 1.33    Hot air, burners, turbines.
        """
        # Set basic constants.
        self._M, self._gamma = M, gamma

        if gas == 'air':
            R = Dim(287.05287, 'J/kg/K')

            # Defaults below are selected to align with GasFlowWF model for
            # dry air with γ = 1.4.
            self._P_ref = Dim(1, 'bar')
            self._T_ref = Dim(298.15, 'K')  # WAS 1000
            self._h_ref = Dim(720.76, 'kJ/kg')
            self._s_ref = Dim(5.68226, 'kJ/kg/K')
        else:
            raise TypeError(f"Unknown gas: {gas}")

        super().__init__(R=R, w=w, gas=gas)

        # Set stream pressure (note γ, M fixed).
        if P and not P0:
            self._P = P
        elif P0 and not P:
            self._P = P0 / self.P0_on_P
        else:
            raise TypeError(f"Invalid pressure argument.")

        # Set stream temperature based on T, T0, h or s.
        if T and not any([T0, h, s]):
            self._T = make_total_temp(T).convert('K')
        elif T0 and not any([T, h, s]):
            self._T = make_total_temp(T0).convert('K') / self.T0_on_T
        elif h and not any([T, T0, s]):
            self._T = (h - self._h_ref) / self.cp + self._T_ref
        elif s and not any([T, T0, h]):
            lnterm = (s - self._s_ref + self.R *
                      log(self._P / self._P_ref)) / self.cp
            self._T = exp(lnterm) * self._T_ref
        else:
            raise TypeError(f"Invalid T, T0, h, s argument.")

        if float(self._T) <= 0 or float(self._P) <= 0:
            raise ValueError(f"Invalid temperature or pressure.")

    @classmethod
    def min_args(cls):
        return 'P', 'T', 'M', 'w', 'gas', 'gamma'

    # Properties --------------------------------------------------------------

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def cp(self) -> Dim:
        return self._R * self._gamma / (self._gamma - 1)

    @property
    def h(self) -> Dim:
        return self._h_ref + self.cp * (self._T - self._T_ref)

    @property
    def M(self) -> float:
        return self._M

    @property
    def P(self) -> Dim:
        return self._P

    @property
    def P0(self) -> Dim:
        return self._P * self.P0_on_P

    @property
    def P0_on_P(self) -> float:
        """Ratio of total (stagnation) pressure to static pressure computed
        using T0/T = (1 + 0.5 * (γ - 1) * M ** 2) ** (γ / (γ - 1)),
        which assumes a perfect gas."""
        return (1 + 0.5 * (self._gamma - 1) * self._M ** 2) ** (self._gamma /
                                                                (self._gamma - 1))

    @property
    def s(self) -> Dim:
        return (self._s_ref + self.cp * log(self._T / self._T_ref) -
                self.R * log(self._P / self._P_ref))

    @property
    def T(self) -> Dim:
        return self._T

    @property
    def T0(self) -> Dim:
        return self._T * self.T0_on_T

    @property
    def T0_on_T(self) -> float:
        """Ratio of total (stagnation) pressure to static temperature
        computed using T0/T = 1 + 0.5 * (γ - 1) * M ** 2, which assumes a
        perfect gas."""
        return 1 + 0.5 * (self._gamma - 1) * self._M ** 2

    @property
    def u(self) -> Dim:
        return self.a * self._M
