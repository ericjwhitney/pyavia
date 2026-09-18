from typing import Protocol, Iterable


# Original by Eric J. Whitney, December 2020.

# ======================================================================

# TODO Unit support - presently SI units.
class Gas(Protocol):
    r"""
    A protocol for models of general gases in motion or at rest.

    Notes
    -----
    - Attributes are intended to be read-only.
    - `Gas` class instances are not intended to be created directly
      by users, but by factory functions.
    """

    def __init__(self, **props: float):
        """
        Gases are constructed by passing the required properties as
        keyword-only arguments to ``__init__``.
        """
        ...

    # -- Properties ----------------------------------------------------

    @property
    def a(self) -> float:
        r"""Local speed of sound :math:`a = \sqrt{{\gamma}RT}`."""
        ...

    @property
    def c_p(self) -> float:
        """Specific heat capacity at constant pressure."""
        ...

    @property
    def c_v(self) -> float:
        """Specific heat capacity at constant volume."""
        ...

    @property
    def h(self) -> float:
        """
        Specific enthalpy of the gas.

        .. note:: The enthalpy baseline is arbitrary and values from
           different formulations / classes should not be compared.
        """
        ...

    @property
    def h0(self) -> float:
        r"""
        Total / stagnation enthalpy of the gas, assuming it is brought
        to rest without losses or heat transfer.
        """
        ...

    @property
    def M(self) -> float:
        """Mach number :math:`M = V/a`."""
        ...

    @property
    def p(self) -> float:
        """Pressure (static / stream)."""
        ...

    @property
    def p0(self) -> float:
        """Total / stagnation pressure."""
        ...

    @property
    def R(self) -> float:
        """
        Gas 'constant' for the specific gas, as used in the equation
        :math:`P = ρRT`.  This value may vary (e.g. with temperature)
        depending on the actual gas model used.
        """
        ...

    @property
    def s(self) -> float:
        """
        Specific entropy of the gas.

        .. note:: The entropy baseline is arbitrary and values from
           different formulations / classes should not be compared.
        """
        ...

    @property
    def T(self) -> float:
        """Temperature (static / stream)."""
        ...

    @property
    def T0(self) -> float:
        """Total / stagnation temperature."""
        ...

    @property
    def V(self) -> float:
        """Flow speed (combined)."""
        ...

    @property
    def γ(self) -> float:
        r"""Ratio of specific heats :math:`\gamma = c_p/c_v.`"""
        ...

    @property
    def ρ(self) -> float:
        """Density."""
        ...

    @property
    def μ(self) -> float:
        """
        Dynamic viscosity (μ) (also called 'absolute viscosity' or just
        'viscosity').
        """
        ...

# ----------------------------------------------------------------------

# TODO Unit support - presently SI units.
class GasFlow(Gas):
    """
    Mixin for `Gas` objects to add 1-D flow properties.

    Parameters
    ----------
    m_dot : float
        Mass flowrate.
    """

    def __init__(self, *, m_dot: float, **kwargs):
        super().__init__(**kwargs)
        self._m_dot = m_dot

    # -- Properties ----------------------------------------------------

    @property
    def Q(self) -> float:
        r"""Enthalpy flowrate :math:`Q = h * \dot{m}`."""
        return self.h * self._m_dot

    @property
    def m_dot(self) -> float:
        """Mass flowrate."""
        return self._m_dot

# ======================================================================

def _check_init_props(props: dict[str, float], allowed: Iterable[str]
                      ) -> list[str]:
    """
    Return a list of property keywords in sorted alphabetical order
    (case insensitive).  Raise `ValueError` if any of them are not in
    the list of allowed properties (case sensitive).
    """
    sorted_keys = sorted(props.keys(), key=lambda alpha: alpha.lower())
    for key in sorted_keys:
        if key not in allowed:
            raise ValueError(
                f"Property '{key}' not in list of allowed properties: "
                f"{', '.join(allowed)}"
            )

    return sorted_keys
