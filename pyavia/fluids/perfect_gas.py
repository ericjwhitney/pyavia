from abc import ABC
from typing import Final

import numpy as np

from ._gas import Gas, _check_init_props

from pyavia.util.function_ops import cached_property_readonly


# Original by Eric J. Whitney, December 2020.

# ======================================================================

# TODO Unit support - presently SI units.

# - Suppress PyCharm warning about incorrect type for properties using
#   cached_property_readonly, this is valid.
# - Suppress PyCharm warning about overriding Final attributes, this is
#   wil be fixed in 3.15.
# noinspection protocol,final
class PerfectGas(Gas, ABC):
    r"""
    A model of a thermally and calorically perfect gas, i.e.
    :math:`c_p` and `R` are constant.

    Parameters
    ----------
    γ : float
        Ratio of specific heats :math:`γ = c_p / c_v`.

    **props :  dict[str, float]
        Keyword arguments giving property names and values that
        properly define the gas state, e.g. ``p=101325, T=288.15``.
        Available properties are:

        - p: Stream / static pressure.
        - p0: Total / stagnation pressure.
        - T: Stream / static temperature.
        - T0: Total / stagnation temperature.
        - M: Mach number.
        - h: Specific enthalpy.
        - h0: Total / stagnation enthalpy.
        - s: Specific entropy.

        Either two or three properties must be supplied:

        - *If two parameters are supplied,* this is assumed to be a
          stationary gas, and only `p`, `T`, `h` or `s` can be used.
          In this case `M = 0` is assumed.

        - *If three parameters are supplied,* any combination of
          parameters can be used provided the gas state is fully
          specified including velocity (in any combination), e.g.
          :math:'P = 101.325 kPa`, :math:`T = 288.15 K` and :math:`M =
          0.5`.

    Attributes
    ----------
    R : float
        Gas constant for the specific gas.  This value can be computed
        from the universal gas constant using :math:`R = R_{univ} / M`
        where :math:`R_{univ}` = 8.314462618 kg.m².s⁻².K⁻¹.mol⁻¹ and M
        is molar mass [kg/mol].

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
    - All attributes are considered read-only.
    - For more information about relationships between different
      properties in ideal gases and relative accuracy see [1]_.
    - Internally, a `PerfectGas` is always defined by stream pressure
      (`p`), temperature (`T`), Mach number (`M`) and ratio of specific
      heats (`γ`).

    References
    ----------
    .. [1] Hilsenrath, J., et al, "Tables of Thermal Properties of
       Gases", Circular 564, U.S. Department of Commerce, National
       Bureau of Standards, 1955.
    """

    # -- Specific Gas Constants ----------------------------------------

    R    : Final[float]
    p_ref: Final[float]
    T_ref: Final[float]
    h_ref: Final[float]
    s_ref: Final[float]

    # -- Magic Methods -------------------------------------------------

    # FutureWork: Overzealous PyCharm warning, this is valid.
    # noinspection missing-constructor
    def __init__(self, *, γ: float, **props: float):
        # Internally, all perfect gases are defined using fundamental
        # properties of 'p', 'T', 'M', 'γ'.  We get property keys in
        # alphabetical order then find the fundamental properties that
        # result in 'props'.
        self.__γ = γ
        prop_keys = _check_init_props(
            props, ('h', 'M', 'p', 'p0', 'T', 's')
        )  # Alphabetical order (case insensitive).

        match prop_keys:  # TODO check for error, allows extra param?
            # -- Two Parameters: Assume M = 0 --------------------------

            case ('p', 'T'):
                self.__p = props['p']
                self.__T = props['T']
                self.__M = 0.0

            case ('h', 'p'):
                self.__p = props['p']
                self.__T = temperature_h(props['h'], h_ref=self.h_ref,
                                         c_p=self.c_p, T_ref=self.T_ref)
                self.__M = 0.0

            case ('p', 's'):
                self.__p = props['p']
                self.__T = temperature_ps(
                    props['p'], props['s'], p_ref=self.p_ref,
                    s_ref=self.s_ref, R=self.R, c_p=self.c_p,
                    T_ref=self.T_ref
                )
                self.__M = 0.0

            # FutureWork: More options.

            # -- Three Parameters: Fully Defined -----------------------

            case ('h', 'p', 'p0'):
                self.__p = props['p']
                self.__T = temperature_h(props['h'], h_ref=self.h_ref,
                                         c_p=self.c_p, T_ref=self.T_ref)
                self.__M = mach(props['p0'] / props['p'], self.__γ)

            case ('M', 'p', 'T'):
                self.__p = props['p']
                self.__T = props['T']
                self.__M = props['M']

            # FutureWork: More options.

            case _:
                raise ValueError(f"Unknown property or combination: "
                                 f"{', '.join(prop_keys)}")

        # Basic check on properties.
        if np.any(self.__p <= 0):
            raise ValueError("Requires p > 0.")

        if np.any(self.__T <= 0):
            raise ValueError("Requires T > 0.")

        if np.any(self.__M < 0):
            raise ValueError("Requires M >= 0.")

    # -- Properties  ---------------------------------------------------

    @cached_property_readonly
    def a(self) -> float:
        r"""
        Local speed of sound.  Computed using :math:`a = \sqrt{γRT}`.
        """
        return (self.__γ * self.R * self.__T) ** 0.5

    @property
    def c_p(self) -> float:
        """
        Specific heat capacity at constant pressure.  Computed using
        :math:`c_p = R.γ / (γ - 1)`.
        """
        return self.R * self.__γ / (self.__γ - 1)

    @property
    def c_v(self) -> float:
        """
        Specific heat capacity at constant volume.  Computed from
        :math:`c_v = c_p - R`.
        """
        return self.c_p - self.R

    @cached_property_readonly
    def h(self) -> float:
        return enthalpy(self.__T, T_ref=self.T_ref, c_p=self.c_p,
                        h_ref=self.h_ref)

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

    @property
    def p0(self) -> float:
        return self.__p * self.p0_p

    @property
    def p0_p(self) -> float:
        """
        Ratio of total (stagnation) pressure to static pressure.
        """
        return p0_p(self.__M, self.__γ)

    @cached_property_readonly
    def s(self) -> float:
        return entropy(
            self.__p, self.__T, p_ref=self.p_ref, T_ref=self.T_ref,
            c_p=self.c_p, R=self.R, s_ref=self.s_ref
        )

    @property
    def T(self) -> float:
        return self.__T

    @property
    def T0(self) -> float:
        return self.__T * self.T0_T

    @property
    def T0_T(self) -> float:
        return T0_T(self.__M, self.__γ)

    @property
    def V(self) -> float:
        """
        Flow velocity.  Computed from :math:`V = M.a`.
        """
        return self.M * self.a

    @property
    def γ(self) -> float:
        return self.__γ

    @property
    def ρ(self) -> float:
        """
        Density :math:`\rho = P/(RT)`.
        """
        return self.__p / (self.R * self.__T)

    @property
    def μ(self):
        raise NotImplementedError


# ======================================================================

def enthalpy(T: float, *, T_ref: float,
             c_p: float, h_ref: float) -> float:
    r"""
    Returns the specific enthalpy of a perfect gas given the
    temperature.  Calculate using:

    .. math:: h = h_{ref} + c_p (T - T_{ref})

    .. note:: Enthalpies can only be compared if they have common
       reference conditions.

    Parameters
    ----------
    T : float
        Temperature.

    T_ref : float
        Reference temperature.

    c_p : float
        Specific heat at constant pressure.

    h_ref : float
        Reference enthalpy.

    Returns
    -------
    float
        Specific enthalpy.
    """
    return h_ref + c_p * (T - T_ref)


# ----------------------------------------------------------------------

def entropy(p: float, T: float, *, p_ref: float, T_ref: float,
            c_p: float, R: float, s_ref: float) -> float:
    r"""
    Computes the specific entropy of a perfect gas given the pressure
    and temperature, using :math:`s = s_{ref} + c_p \log (T/T_{ref}) -
    R \log (p/p_{ref})`.

    Parameters
    ----------
    p : float
        Pressure/s (static / stream).

    T : float
        Temperature/s (static / stream).

    p_ref : float
        Reference pressure for the gas model in `Pa`.

    T_ref : float
        Reference temperature for the gas model in `K`.

    c_p : float
        Specific heat at constant pressure.

    R : float
        Gas constant for the specific gas.

    s_ref : float
        Reference specific entropy.

    Returns
    -------
    float
    """
    return s_ref + c_p * np.log(T / T_ref) - R * np.log(p / p_ref)

# ----------------------------------------------------------------------

def mach(p0_p_: float, γ: float) -> float:
    r"""
    Returns the Mach number of a perfect gas given the stagnation
    pressure ratio.  Calculated using:

    .. math:: M = \sqrt{ \frac{2}{γ-1} (1 - \frac{p_0}{p})^{\frac{γ-1}{γ} } }

    Parameters
    ----------
    p0_p_ : float
        Stagnation to stream pressure ratio.

    γ : float
        Ratio of specific heats.

    Returns
    -------
    float
        Mach number.
    """
    return np.sqrt(2 / (γ - 1) * (1 - p0_p_) ** ((γ - 1) / γ))

# ----------------------------------------------------------------------

def p0_p(M: float, γ: float) -> float:
    r"""
    Computes the ratio of total (stagnation) pressure to static pressure
    of a perfect gas from the Mach number.  Calculated using:

    .. math:: \frac{P_0}{P} = (1 + \frac{1}{2}(γ - 1)M^2)^{
       \frac{γ}{γ - 1} }

    Parameters
    ----------
    M : float
        Mach number.

    γ : float
        Ratio of specific heats.

    Returns
    -------
    float
        Ratio of stagnation to stream pressure.
    """
    return (1 + 0.5 * (γ - 1) * M ** 2) ** (γ / (γ - 1))

# ----------------------------------------------------------------------

def T0_T(M: float, γ: float) -> float:
    r"""
    Computes the ratio of total (stagnation) temperature to static
    temperature of a perfect gas from the Mach number.  Calculated
    using:

    .. math:: T_0 / T = 1 + \frac{1}{2} (γ - 1) M^2

    Parameters
    ----------
    M : float
        Mach number/s :math:`M = u/a`.

    γ : float
        Ratio of specific heats.

    Returns
    -------
    float
        Ratio of stagnation to stream temperature.
    """
    return 1 + 0.5 * (γ - 1) * M ** 2

# ----------------------------------------------------------------------

def temperature_h(h: float, *, h_ref: float, c_p: float, T_ref: float) -> float:
    r"""
    Returns the static temperature of a perfect gas given the specific
    enthalpy.  Calculated using:

     .. math:: T = T_{ref} + (h - h_{ref}) / c_p

    Parameters
    ----------
    h : float
        Specific enthalpy.

    h_ref : float
        Reference enthalpy.

    c_p : float
        Specific heat at constant pressure.

    T_ref : float
        Reference temperature.

    Returns
    -------
    float
    """
    return T_ref + (h - h_ref) / c_p

# ----------------------------------------------------------------------

def temperature_ps(p: float, s: float, *, p_ref: float, s_ref: float,
                   R: float, c_p: float, T_ref: float) -> float:
    r"""
    Computes the temperature of a perfect gas given the pressure
    and entropy.  Calculated using:

    .. math:: T = T_{ref} \exp [ ((s - s_{ref}) + R \log (p / p_{ref})) / c_p ]

    Parameters
    ----------
    p : float
        Static / stream pressure.

    s : float
        Specific entropy.

    p_ref : float
        Reference pressure.

    s_ref : float
        Reference specific entropy.

    R : float
        Gas constant for the specific gas.

    c_p : float
        Specific heat at constant pressure.

    T_ref : float
        Reference temperature.

    Returns
    -------
    float
        Static / stream temperature.
    """
    return T_ref * np.exp(((s - s_ref) + R * np.log(p / p_ref)) / c_p)
