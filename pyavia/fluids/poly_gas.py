from abc import ABC
from typing import Final

import numpy as np
import numpy.typing as npt

from ._gas import Gas, _check_init_props
from ._make_gas import make_gas
from pyavia.numeric.solve import fixed_point, SolverError
from pyavia.util.function_ops import cached_property_readonly


# Original by Eric J. Whitney, December 2020.

# =============================================================================

# TODO general units (presently SI).

# - Suppress PyCharm warning about incorrect type for properties using
#   cached_property_readonly, this is valid.
# - Suppress PyCharm warning about overriding Final attributes, this is
#   wil be fixed in 3.15.
# noinspection protocol,final
class PolyGas(Gas, ABC):
    r"""
    Abstract model of real (imperfect) gas based on fitted polynomials,
    detailed in [1]_.  Specific gases are defined as subclasses that
    supply their own coefficients for the polynomials.

    Parameters
    ----------
    R : float or array_like of float, shape (N,)
        Gas constant for the specific gas.  This may be a scalar or
        array due to changes in `R` caused by real gase effects.


    Attributes
    ----------
    p_ref : float
        Reference pressure for the gas model in `Pa`.  

    T_ref : float
        Reference temperature for the gas model in `K`.

    h_ref : float
        Reference enthalpy for the gas at :math:`p_{ref}` and
        :math:`T_{ref}` in `kJ/kg`.

    s_ref : float
        Reference entropy for the gas model at :math:`p_{ref}` and
        :math:`T_{ref}` in `kJ/kg/K`. 

    Notes
    -----
    TODO update
    - Valid temperature range is `T` = 200 K → 2000 K.
    - Properties :math:`\gamma`, :math:`c_p`, `h`, `s` are non-linear
      and dependent on temperature (i.e. a calorically imperfect gas).
      Stagnation enthalpy, temperature and pressure also do not assume a
      perfect gas. Stagnation properties are computed by bringing the
      flow to rest isentropically.
    - Internal values stored as SI units [K, Pa, ...].
    - Internal reference states are stream pressure (`p`), temperature
      (`T`), Mach number (`M`) .
    - Stream / static temperature and pressure and Mach number are the
      internal reference states.

    References
    ----------
    .. [1] Walsh, P. & Fletcher, P., "Gas Turbine Performance", Second
       Edition, Chapter 3.
    """
    # -- Specific Gas Constants ----------------------------------------

    p_ref: Final[float]
    T_ref: Final[float]
    h_ref: Final[float]
    s_ref: Final[float]

    _A_COEFFS: Final[npt.NDArray[np.float64]]
    """
    Walsh & Fletcher Eqn F3.23 'A' coefficients for key gases. Note:
    These coefficients are used in equations for c_p, h, integrals, etc,
    and not all terms are used in each equation.
    """

    _ENTROPY_P_REF: Final[float] = 100_000  # 1 bar -> [Pa]
    """Reference pressure for entropy calculations."""

    # -- Magic ---------------------------------------------------------

    # TODO duplicated code here with PerfectGas.

    # noinspection PyMissingConstructor
    def __init__(self, *, R: float, **props: float):
        # Convert array-like arguments to numpy arrays (inc. copy) where
        # possible.
        self._R = R

        # Internally, polynomial gases are defined using fundamental
        # properties of 'p', 'T', 'M'.
        prop_keys = _check_init_props(
            props, ('h', 'M', 'p', 'p0', 'T', 's')
        )  # Alphabetical order (case insensitive).

        match prop_keys:
            # -- Two Parameters: Assume M = 0 --------------------------

            case ('p', 'T'):
                self.__p = props['p']
                self.__T = props['T']
                self.__M = 0.0

            case ('h', 'p'):
                self.__p, self.__T, self.__M = self._fit_pTM(
                    h=props['h'], p=props['p'], M=0.0)

            case ('p', 's'):
                self.__p, self.__T, self.__M = self._fit_pTM(
                    p=props['p'], s=props['s'], M=0.0)

            # Future work: More options.

            # -- Three Parameters: Fully Defined -----------------------

            case ('h', 'p', 'p0'):
                self.__p, self.__T, self.__M = self._fit_pTM(
                    h=props['h'], p=props['p'], p0=props['p0'])

            case ('M', 'p', 'T'):
                self.__p = props['p']
                self.__T = props['T']
                self.__M = props['M']

            # Future work: More options.

            case _:
                raise ValueError(f"Unknown property or combination: "
                                 f"{', '.join(prop_keys)}")

        # Check model limits.
        if np.any(self.__p <= 0):
            raise ValueError("Require p > 0.")

        if np.any(self.__T < 200) or np.any(self.__T > 2000):
            raise ValueError("Temperature outside range 200 K - 2000 K.")

        if np.any(self.__M < 0):
            raise ValueError("Require M >= 0.")

    # -- Properties ----------------------------------------------------

    @property
    def a(self) -> float:
        return (self._R * self.__T * self.γ) ** 0.5

    @cached_property_readonly
    def c_p(self) -> float:
        """
        Specific heat capacity at constant pressure, which is non-linear
        with temperature.  This is computed using Walsh & Fletcher Eqn
        F3.23 (see `References` for more information).
        """
        return self._c_p_poly(self.__T)

    @property
    def c_v(self) -> float:
        """
        Specific heat capacity at constant volume, which is non-linear
        with temperature.  This is compute using :math:`c_p - R`.
        """
        return self.c_p - self._R

    @cached_property_readonly
    def h(self) -> float:
        """
        Specific enthalpy, which is non-linear with temperature.  This
        is computed using Walsh & Fletcher Eqn F3.26 (see
        `References` for more information).

        .. note:: The enthalpy baseline is arbitrary and values from
           different formulations / classes should not be compared.
        """
        return self._enthalpy_poly(self.__T)

    @property
    def h0(self) -> float:
        r"""
        Total / stagnation enthalpy of the gas, assuming it is brought
        to rest without losses or heat transfer.  Defined as
        :math:`h_0 = h + \frac{1}{2}u^2`, ignoring graviational (height)
        change effects.
        """
        return self.h + 0.5 * self.V ** 2

    @property
    def M(self) -> float:
        return self.__M

    @property
    def p(self) -> float:
        return self.__p

    @cached_property_readonly
    def p0(self) -> float:
        r"""
        Total / stagnation pressure.  This is calculated to give equal
        entropy at :math:`T_0` as the flow state (`p`, `T`) using
        :math:`p_0 = p_{ref} * exp((\int (cp/T) dT - s_0) / R)`.
        """
        integral = self._entropy_poly(self.T0)
        s0_target = self.s
        return self._ENTROPY_P_REF * np.exp((integral - s0_target) / self._R)

    @property
    def R(self) -> float:
        """Gas constant for the specific gas."""
        return self._R

    @cached_property_readonly
    def s(self) -> float:
        r"""
        Specific entropy, computed using
        :math:`s = \int {c_p/T} dT - R \log (P/P_{ref})`.

        .. note:: The entropy baseline is arbitrary and values from
           different formulations / classes should not be compared.
        """
        # TODO should this have an s0 term?  Or is it covered by using a
        #  single sided term?
        integral = self._entropy_poly(self.__T)
        return integral - self._R * np.log(self.__p / self._ENTROPY_P_REF)

    @property
    def T(self) -> float:
        """Temperature (static / stream)."""
        return self.__T

    @cached_property_readonly
    def T0(self) -> float:
        """
        Total / stagnation temperature.  This value is computed by
        finding a value that gives correct stagnation enthalpy
        :math:`h_0` by converging using a fixed point method.
        """
        # All units are J, kg, K.
        h0_target = self.h0

        def T_next(T_):
            # Gradient dh/dT = c_p.
            dh_dT = self._c_p_poly(T_)  # Was fixed @ 1,005.
            ΔT = (h0_target - self._enthalpy_poly(T_)) / dh_dT
            return T_ + ΔT

        try:
            T0 = fixed_point(T_next, x0=self.__T, xtol=1e-9, maxits=15)

        except SolverError as e:
            raise SolverError(
                f"Failed to compute T0.", flag=1,
                details=f"Failed to set stagnation conditions: "
                        f"p = {self.p:.5G}, T = {self.T:.02f}, "
                        f"M = {self.M:.03f}, h0 = {h0_target:.5G}.")

        return T0

    @property
    def V(self) -> float:
        return self.a * self.__M

    @property
    def γ(self) -> float:
        r"""
        Ratio of specific heats :math:`\gamma = c_p/c_v.`  Computed
        using :math:`\gamma = c_p/(c_p - R)`.
        """
        return self.c_p / (self.c_p - self._R)

    @property  # TODO cached?
    def ρ(self) -> float:
        """Density. Compute from :math:`ρ = P/(RT)`."""
        return self.__p / (self._R * self.__T)

    # -- Protected Methods ---------------------------------------------

    def _c_p_poly(self, T: float) -> float:
        # Compute c_p from T in units kJ/kg/K using W&F Eqn F3.23.
        Tz = T / 1_000  # T [K] -> Eqn Tz [K/1000]
        c_p = np.polyval(self._A_COEFFS[8::-1], Tz) * 1_000  # kJ -> J
        return c_p

    def _enthalpy_poly(self, T: float) -> float:
        # Compute enthalpy in units MJ/kg using W&F Eqn F3.26.
        Tz = T / 1_000  # K -> Eqn [K/1000]

        # WAS
        # h = self._A_coeffs[9] + sum([(a_i / i) * Tz ** i for i, a_i in
        #                              enumerate(self._A_coeffs[0:9], 1)])

        # Coeffs: A9, A0, A1 / 2, ..., A8 / 9
        # Terms:   1,  x,  x**2,  ...,   x**8
        coeffs = np.r_[self._A_COEFFS[9], (self._A_COEFFS[0:9] /
                                           np.arange(1, 10))]
        h = np.polyval(coeffs[::-1], Tz) * 1_000_000  # Eqn MJ/kg -> [J/kg]

        return h

    def _entropy_poly(self, T: float) -> float:
        # Compute ∫cp/T dT term from T using W&F Eqn F3.28 (units
        # kJ/kg/K).  Add EJW correction term.
        Tz = T / 1_000  # K -> Eqn [K/1000]

        # WAS
        # EJW_A0_corr_term = self._A_COEFFS[0] * np.log(1000)
        # cptint = ((self._A_COEFFS[0] * np.log(Tz)) + sum(
        #     [(a_i / i) * Tz ** i
        #      for i, a_i in enumerate(self._A_COEFFS[1:9], 1)]) +
        #           self._A_COEFFS[10]) + EJW_A0_corr_term

        # TODO THIS WHOLE CORRECTION SHAMOZZLE MIGHT BE FIXED BY USING
        #  A0*log(T) instead of A0*log(Tz).  CHECK.

        # NOW

        # Coeffs: A10, A1, A2 / 2, ..., A8 / 8
        # Terms:    1,  x,   x**2, ...,   x**8
        coeffs = np.r_[self._A_COEFFS[10], (self._A_COEFFS[1:9] /
                                            np.arange(1, 9))]

        # Calculate integral including EJW correction for A0 term (*).
        integral = (np.polyval(coeffs[8::-1], Tz)
                    + self._A_COEFFS[0] * np.log(Tz)  # A0 × ln(Tz)
                    + (self._A_COEFFS[0] * np.log(1000))  # (*)
                    ) * 1_000  # Eqn kJ/kg/K -> [J/kg/K]

        return integral

    def _fit_pTM(self, **reqd_props: float) -> tuple[float, float, float]:
        # Compute core properties 'p', 'T', 'M' by fitting to given
        # properties.
        fitted = make_gas(type(self), ('p', 'T', 'M'), **reqd_props)
        return fitted.p, fitted.T, fitted.M

    @property
    def FAR(self) -> float:
        """
            # TODO push this down exclusively to air.

        Fuel-Air Ratio :math:`FAR = w_f / w_{total}` where :math:`w_f`
        is the massflow of fuel products of combustion and
        :math:`w_{total}` is the total massflow.  E.G. If the upstream
        flow was pure airflow of :math:`w_{air}` then
        :math:`FAR = w_f / (w_f + w_{air})`.
        """
        ...

# ----------------------------------------------------------------------

# # Specific gas constants.
# _R_spec_ref = {  # [J/kg/K]
#     'dry_air': 287.05287,  # ISO 2533-1975
#     'hydrogen': 4_124.201,  # (*) Circ. 564, U.S. Dept. of Commerce
# }
#
# # (*) The 'R' values found in Circular 564 from the U.S. Department of
# # Commerce use completely stupid units.  To convert to something
# # reasonable, take the T=[K], p=[atm], ρ=[g/cm³] value and multiply by
# # ×101.325 to get [J/kg/K].  A further /1000 can be applied to get
# # [kJ/kg/K] if desired.



