
from __future__ import annotations

import numpy as np

from .gas import Gas, _solve_TPM
from pyavia.units import dim, Dim
from pyavia.util import split_dict


# Written by Eric J. Whitney, January 2021.

# =============================================================================

# noinspection PyPep8Naming
class PerfectGas(Gas):
    r"""
    A simplified thermally and calorically perfect gas, i.e. c_p and R are
    constant.  Once initialised, properties are fixed.

    .. note::
        - Internal working parameters are stream pressure, temperature,
          Mach number and ratio of specific heats  (`P`, `T`, `M`, `γ`).
        - Sepcific enthalpy computation is :math:`h = h_ref + c_p.(T -
          T_{ref})`.
        - Specific entropy computation is :math:`s = s_ref + c_p.ln(T /
          T_{ref}) - R.ln(P / P_{ref})`.

    Reference values for `air`:
        - P_ref = 100 kPa.  Note this is 1 bar, not an ISA standard atmosphere.
        - T_ref = 298.15 K.
        - h_ref = 720.76 kJ/kg.
        - s_ref = 5.68226 kJ/kg/K.
    """

    def __init__(self, *, gas: str = 'air', gamma: float = 1.4, **kwargs):
        """
        Construct a thermally and calorically perfect gas with 1-D flow
        properties.

        Parameters
        ----------
        gas : str
            Gas idenfitifer, used to set `R` and reference values.  Supported
            values for this model are:

                - 'air' (default).

        gamma : float
            Ratio of specific heats.  Commonly used values for air are:

                - γ = 1.4: Atmospheric air, compressors (default).
                - γ = 1.33: Hot air, burners, turbines.

        kwargs : {str: Dim | float}
            Remaining arguments must include three state properties that
            properly define the gas.  These can be supplied in any
            combination provided they fully define the state, e.g. {'P':
            dim(101.325, 'kPa'), 'T': dim(288.15, 'K'), 'M': 0.0}.

            Available properties are:
                - P0: Total / stagnation pressure.
                - T0: Total / stagnation temperature.
                - P: Stream / static pressure.
                - T: Stream / static temperature.
                - h: Specific enthalpy.
                - h0: Total / stagnation enthalpy.
                - s: Specific entropy.
                - M: Mach number.

            If `P`, `T` and `M` are supplied these are directly set and
            initialisation is complete.  Other combinations are converted
            into these values using perfect gas equations or iterative
            convergence of `T`, `P` and `M`.
        """
        # Set basic constants.
        self._gamma = gamma
        if gas == 'air':
            R = dim(287.05287, 'J/kg/K')

            # Fixed reference values below are selected to align with
            # GasFlowWF model for dry air with γ = 1.4.
            self._P_ref = dim(100, 'kPa')  # 1 bar, not ISA std atm.
            self._T_ref = dim(298.15, 'K')
            self._h_ref = dim(720.76, 'kJ/kg')
            self._s_ref = dim(5.68226, 'kJ/kg/K')
        else:
            raise ValueError(f"Unknown gas: {gas}")

        # Extract state definition from kwargs.
        state, kwargs = split_dict(kwargs, ('P0', 'T0', 'P', 'T', 'h', 'h0',
                                            's', 'M'))

        # Now we can initialise the superclass.
        super().__init__(R=R, **kwargs)

        # ---------------------------------------------------------------------

        # Where possible, try to massage the state provided into T, P, M to
        # avoid complex calculation.

        com_err = "Invalid combination of states: "
        com_err += ', '.join([f"'{st}'" for st in state])

        if 'h' in state:
            # Replace static enthalpy with static temperature.
            if 'T' in state:  # Can't already have T with h.
                raise ValueError(com_err)
            state['T'] = temp_from_enthalpy(state['h'], self.c_p, self._T_ref,
                                            self._h_ref)
            del state['h']

        if 'T0' in state and 'M' in state:
            # Replace stagnation temperature with static temperature.
            if 'T' in state:  # Can't already have T with T0, M.
                raise ValueError(com_err)

            state['T'] = state['T0'] / stag_temp_ratio(self._gamma, state['M'])
            del state['T0']

        if 'P0' in state and 'M' in state:
            # Replace stagnation pressure with static pressure.
            if 'P' in state:  # Can't already have P with P0, M.
                raise ValueError(com_err)

            state['P'] = state['P0'] / stag_press_ratio(self._gamma, state['M'])
            del state['P0']

        # TODO More opportunities exist: h0, s, s0, etc.

        # ---------------------------------------------------------------------

        # Initialise the gas using our best available starting point.
        if len(state) != 3:
            raise ValueError(f"Need three state properties to uniquely "
                             f"specify flow, got: "
                             f"{', '.join(list(state.keys()))}")

        if set(state.keys()) != {'T', 'P', 'M'}:
            # We didn't arrive at T, P, M directly.  Solve for these by
            # constructing trial flows until the state matches.
            self._T, self._P, self._M = _solve_TPM(
                PerfectGas, gas=gas, gamma=self._gamma, **state, **kwargs)
        else:
            # T, P, M directly available.
            self._T = state['T']
            self._P = state['P']
            self._M = state['M']

        # Specifically check gas temperatures are total and not delta.
        if not self._T.is_total_temp():
            raise ValueError(f"Gas temperatures must be total not Δ, got "
                             f"{self._T:.5G}")

    # -- Properties ---------------------------------------------------------

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def c_p(self) -> Dim:
        return self.R * self._gamma / (self._gamma - 1)

    @property
    def h(self) -> Dim:
        return enthalpy(self._T, self.c_p, self._T_ref, self._h_ref)

    @property
    def M(self) -> float:
        return self._M

    @property
    def P(self) -> Dim:
        return self._P

    @property
    def P0(self) -> Dim:
        return self._P * self.P0_P

    @property
    def P0_P(self) -> float:
        """
        Ratio of total (stagnation) pressure to static pressure.
        """
        return stag_press_ratio(self._gamma, self._M)

    @property
    def s(self) -> Dim:
        return (self._s_ref + self.c_p * np.log(self._T / self._T_ref) -
                self.R * np.log(self._P / self._P_ref))

    @property
    def T(self) -> Dim:
        return self._T

    @property
    def T0(self) -> Dim:
        return self._T * self.T0_T

    @property
    def T0_T(self) -> float:
        """
        Ratio of total (stagnation) pressure to static temperature.
        """
        return stag_temp_ratio(self._gamma, self._M)

    @property
    def u(self) -> Dim:
        return self.a * self._M


# =============================================================================

# General perfect gas equations.

def enthalpy(T: Dim, c_p: Dim, T_ref: Dim, h_ref: Dim) -> Dim:
    r"""
    Returns the specific enthalpy of a perfect gas, computed using
    :math:`{h} = h_{ref} + c_p (T - T_{ref})`

    .. note:: Enthalpies can only be compared if they have common reference
              conditions.
    """
    return h_ref + c_p * (T - T_ref)


def stag_press_ratio(gamma: float, M: float) -> float:
    r"""
    Returns the ratio of total (stagnation) pressure to static pressure of
    a perfect gas, computed using
    :math:`\frac{P_0}{P} = (1 + \frac{1}{2}(γ - 1)M^2)^{\frac{γ}{γ - 1}}`.
    """
    return (1 + 0.5 * (gamma - 1) * M ** 2) ** (gamma / (gamma - 1))


def stag_temp_ratio(gamma: float, M: float) -> float:
    r"""
    Returns the ratio of total (stagnation) pressure to static temperature of
    a perfect gas, computed using
    :math:`\frac{T_0}{T} = 1 + \frac{1}{2}(γ - 1)M^2`.
    """
    return 1 + 0.5 * (gamma - 1) * M ** 2


def temp_from_enthalpy(h_: Dim, c_p: Dim, T_ref: Dim, h_ref: Dim) -> Dim:
    r"""
    Returns the static temperature of a perfect gas, based on the specific
    enthalpy, computed using :math:`{T} = T_{ref} + (h - h_{ref}) / c_p`
    """
    return T_ref + (h_ - h_ref) / c_p

# -----------------------------------------------------------------------------
