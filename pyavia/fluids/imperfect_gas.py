# Written by Eric J. Whitney, December 2020.

from __future__ import annotations

import numpy as np

from .gas import Gas, _solve_TPM
from pyavia.numeric.solve import fixed_point
from pyavia.units import dim, Dim
from pyavia.util import split_dict


# =============================================================================


# noinspection PyPep8Naming
class ImperfectGas(Gas):
    r"""
    Imperfect gas using polynomials from Walsh & Fletcher §3. Includes
    optional products of combustion.  Properties are fixed once initialised.

    .. note::
        - Temperature range is 200 K → 2000 K.
        - Fuel-Air Ratio `FAR` = 0.00 → 0.05
        - Non-linear :math:`\gamma`, :math:`c_p`, `h`, `s` dependent on
          temperature (i.e. calorically imperfect).  Stagnation enthalpy,
          temperature and pressure also do not assume a perfect gas.
          Stagnation properties are computed by bringing the flow to rest
          isentropically.
        - Internal values stored as SI units [K, kPa, ...].
        - Stream / static temperature and pressure and Mach number are the
          internal reference states.
    """

    def __init__(self, *, gas: str = 'air', FAR: float = 0.0,
                 init_gas: Gas = None, **kwargs):
        """
        Construct a compressible, imperfect gas, based on interpolating
        polynomials given in Walsh & Fletcher, "Gas Turbine Performance",
        Second Edition, Chapter 3.

        Parameters
        ----------
        gas : str
            Gas identifier.  Available `gas` values are:

            - 'air': Dry air (default).
            - 'burned_kerosene': Products of combustion for kerosene in dry
              air.
            - 'burned_diesel': Products of combustion for diesel in dry air.

        FAR : float
            Refer to `GasFlow` for details.

        init_gas : Gas
            If T, P, M are not directly specified for the gas, init_gas can
            be provided which will be a gas with very similar properties to
            those requested.  For example, this can be a previous solution
            point. This can act as a starting point so as to accelerate
            convergence / solution of the gas equations.  If not required it is
            ignored.

        kwargs : {str: Dim | float}
            Required state properties of the gas flow, supplied as a dict.
            Refer to ``PerfectGasFlow`` initialisation for requirements.

        init_gasflow : GasFlow (Optional)
            If provided, relevant properties from this reference gasflow are
            used as a starting point to accelerate the construction /
            convergence of the new gasflow.
        """
        # Find state values provided.
        state, kwargs = split_dict(kwargs, ('P0', 'T0', 'P', 'T',
                                            'h', 'h0', 's', 'M'))

        if set(state.keys()) != {'T', 'P', 'M'}:
            # Direct T, P, M state not provided.  Solve for these by
            # constructing trial flows until the state matches.
            self._T, self._P, self._M = _solve_TPM(
                ImperfectGas, gas=gas, FAR=FAR, init_gas=init_gas,
                **state, **kwargs)

        else:
            # T, P, M directly available.
            self._T = state['T']
            self._P = state['P']
            self._M = state['M']

        # Specifically check gas temperatures are total and not delta.
        if not self._T.is_total_temp():
            raise ValueError(f"Gas temperatures must be total not Δ, got "
                             f"{self._T:.5G}")

        # Check we are within model limits.
        if not (dim(200, 'K') <= self._T <= dim(2000, 'K')):
            raise ValueError(f"T = {self._T:.5G} outside model limits of "
                             f"200 K -> 2000 K.")

        if not (0 <= FAR <= 0.05):
            raise ValueError(f"FAR = {FAR:.5G} outside model limits of "
                             f"0 -> 0.05.")

        # Set internal coefficients.
        self._coeff_a, self._coeff_b = None, None
        if gas in ('air', 'burned_kerosene', 'burned_diesel'):
            R = 287.05287  # J/kg/K.
            self._coeff_a = _WF_A_COEFF['air']

            if gas == 'burned_kerosene':
                R += -0.00990 * FAR + 1e-7 * FAR ** 2
                self._coeff_b = _WF_B_COEFF

            elif gas == 'burned_diesel':
                R += -8.0262 * FAR + 3e-7 * FAR ** 2
                self._coeff_b = _WF_B_COEFF

            R = dim(R, 'J/kg/K')
        else:
            raise ValueError(f"Invalid gas: {gas}")

        self._gas = gas

        # Now we can initialise the superclass.
        super().__init__(R=R, FAR=FAR, **kwargs)

        # Setup lazily evaluated attributes.
        self._cp, self._cptint, self._gamma = None, None, None
        self._h, self._P0, self._T0 = None, None, None

    # -- Properties ---------------------------------------------------------

    @property
    def c_p(self) -> Dim:
        if self._cp is None:
            self._cp = self._cp_from_T(self._T)
        return self._cp

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            self._gamma = self.c_p / (self.c_p - self.R)
        return self._gamma

    @property
    def gas(self) -> str:
        """
        Identifier for the gas.  Refer to ``__init__`` for details.
        """
        return self._gas

    @property
    def h(self) -> Dim:
        if self._h is None:
            self._h = self._h_from_T(self._T)
        return self._h

    @property
    def M(self) -> float:
        return self._M

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
        # Equation: s = int(cp/T).dT - R.ln(P/P_ref)
        return self._cptint - self.R * np.log(self._P / _WF_P_REF_S)

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

    # -- Private Methods ----------------------------------------------------

    def _cp_from_T(self, T: Dim) -> Dim:
        """
        Compute cp  using W&F Eqn F3.23, given T [units K required if plain
        scalar].  Returns units J/kg/K. self._coeff_a and self._coeff_b must
        be set prior to call.
        """
        Tz = T.to_value('K') / 1000  # noqa

        cp = sum([a_i * Tz ** i
                  for i, a_i in enumerate(self._coeff_a[0:9], 0)])
        if self._coeff_b is not None:
            cp += (self.FAR / (1 + self.FAR)) * sum(
                [b_i * Tz ** i
                 for i, b_i in enumerate(self._coeff_b[0:8], 0)])
        return dim(1000 * cp, 'J/kg/K')  # Eqn for kJ/kg/K, return J/kg/K.

    def _cptint_from_T(self, T: Dim) -> Dim:
        """
        Compute integral(cp/T) [kJ/kg/K] using W&F Eqn F3.28 with EJW
        correction term given T, return kJ/kg/K. self._coeff_a and
        self._coeff_b must be set prior to call.
        """
        Tz = T.to_value('K') / 1000  # noqa

        EJW_A0_corr_term = self._coeff_a[0] * np.log(1000)
        cptint = ((self._coeff_a[0] * np.log(Tz)) + sum(
            [(a_i / i) * Tz ** i
             for i, a_i in enumerate(self._coeff_a[1:9], 1)]) +
                  self._coeff_a[10]) + EJW_A0_corr_term
        if self._coeff_b is not None:
            EJW_B0_corr_term = self._coeff_b[0] * np.log(1000)
            # XXX TODO VERIFY FAR CORR TERM WITH EXAMPLES
            cptint += (self.FAR / (1 + self.FAR)) * (
                    self._coeff_b[0] * np.log(Tz) +
                    sum([(b_i / i) * Tz ** i
                         for i, b_i in enumerate(self._coeff_b[1:8], 1)]) +
                    self._coeff_b[9] + EJW_B0_corr_term)
        return dim(cptint, 'kJ/kg/K')

    def _h_from_T(self, T: Dim) -> Dim:
        """Compute enthalpy [MJ/kg] using W&F Eqn F3.26, given T and return
        kJ/kg/K.  self._coeff_a and self._coeff_b must be set prior to call.
        """
        Tz = T.to_value('K') / 1000  # noqa

        h = self._coeff_a[9] + sum([(a_i / i) * Tz ** i for i, a_i in
                                    enumerate(self._coeff_a[0:9], 1)])
        if self._coeff_b is not None:
            h += (self.FAR / (1 + self.FAR)) * (sum(
                [(b_i / i) * Tz ** i
                 for i, b_i in enumerate(self._coeff_b[0:7], 1)]) +
                                                self._coeff_b[8])
        return dim(1000 * h, 'kJ/kg')

    def _set_stagnation(self) -> None:
        h0_target = self.h0
        s0_target = self.s

        # Find T that gives h0 using fixed point method and approx cp.  Note:
        # Units are K in this routine (removed for solver).
        def update_T(trial_T_K):
            ΔT_K = ((h0_target - self._h_from_T(dim(trial_T_K, 'K'))) /
                    dim(1005, 'J/kg/K')).to_value('K')

            return trial_T_K + ΔT_K

        try:
            # noinspection PyTypeChecker
            T0_K = fixed_point(update_T, x0=self.T.to_value('K'),  # Plain K.
                               xtol=1e-5, maxits=20)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to set stagnation conditions: h0 = "
                               f"{h0_target:.5G}, s0 = {s0_target:.5G}.\n"
                               f"Solver message: {str(e)}")

        self._T0 = dim(T0_K, 'K')

        # Calculate P to give equal entropy.
        # P0 = P_Ref * exp((int(cp/T).dT - s0) / R)
        cptint0 = self._cptint_from_T(self._T0)
        self._P0 = _WF_P_REF_S * np.exp((cptint0 - s0_target) / self.R)


# -----------------------------------------------------------------------------

_WF_P_REF_S = dim(1, 'bar')  # Reference pressure for entropy.

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
