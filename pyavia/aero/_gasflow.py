"""
Equations and constants relating to gases and compressible flow.
"""
# Last updated: 11 January by Eric J. Whitney

from __future__ import annotations
from pyavia.solve import fixed_point, solve_dqnm
from pyavia.core.units import Dim, make_total_temp
from pyavia.core.util import all_not_none, all_none
from abc import ABC, abstractmethod
from math import log, exp, inf
from typing import Sequence, Tuple, List, Dict

__all__ = ['GasError', 'GasFlow', 'PerfectGasFlow', 'GasFlowWF']


class GasError(Exception):
    """Exception class specific to gas computations."""
    pass

# ----------------------------------------------------------------------------


# noinspection PyPep8Naming
class GasFlow(ABC):
    """
    Abstract base class representing common properties of gases flowing in
    1-D. Once any gas is initialised, properties are fixed.

    .. note:: This base class does not provide properties such as `T0/T` or
       `P0/T` because non-linearities in imperfect gas models may make these
       ratios moot.  Refer to the specific derived class.
    """

    # Magic methods ----------------------------------------------------------

    def __init__(self, *, R, w=None, gas: str = None):
        """
        Construct an abstract base representation of flowing compressible
        gas.  Once constructed, the properties of the gas are fixed.

        Parameters
        ----------
        R :
            Gas constant for this particular gas (i.e.
            :math:`R = R_{univ}/n`).
        w :
            Mass flowrate
        """
        self._R, self._w, self._gas = R, w, gas

    def __format__(self, format_spec) -> str:
        """Formatted output string.  The defining state properties are
        assumed to be T, P, M, w by default.  Format is applied to numeric
        values only."""
        arg_list = ['T', 'P', 'M', 'w'] + list(self.special_args())
        prop_strs = []
        for prop in arg_list:
            val = getattr(self, prop)
            try:
                float(val)  # Try conversion.
                fmt = format_spec
            except (TypeError, ValueError):
                fmt = ''  # Treat as plain value.
            prop_strs += [f'{prop}={val:{fmt}}']
        return type(self).__name__ + ': ' + ', '.join(prop_strs)

    def __repr__(self):
        arg_list = ['T', 'P', 'M', 'w'] + list(self.special_args())
        return type(self).__name__ + '(' + ', '.join(
            [f'{p}={repr(getattr(self, p))}' for p in arg_list]) + ')'

    def __str__(self):
        return self.__format__('.5G')

    # Normal methods / properties --------------------------------------------

    @property
    def a(self):
        r"""Local speed of sound :math:`a = \sqrt{{\gamma}RT}`."""
        return (self.gamma * self._R * self.T) ** 0.5

    def new_props(self, **kwargs) -> GasFlow:
        """Return a new GasFlow object specialised to match `self` except
        without any state properties `T`, `T0`, `P`, `P0`, `h`, `h0`, `s` or
        `M`, which are to be supplied by the user in `**kwargs`. This is used
        for making a new GasFlow object from the current one but with
        changed properties.

        .. note::

            - Any arguments in `**kwargs` supersedes any matching defaults
              supplied by new_props.
            - Mass flowrate `w` is also copied automatically.
            - Refer to __init__ for the derived class for valid property
              combinations."""
        # Build keywords.
        cls = type(self)
        spec_kwargs = {k: getattr(self, k) for k in cls.special_args()}
        spec_kwargs['w'] = self.w

        # Add / overwrite new args.
        for k, v in kwargs.items():
            spec_kwargs[k] = v

        return cls(**spec_kwargs)

    @property
    def cv(self):
        """Specific heat capacity at constant volume of the gas."""
        return self.cp - self.R

    @property
    def gas(self) -> str:
        """Identifier for the gas."""
        return self._gas

    @property
    def h0(self):
        r"""Total / stagnation enthalpy of the gas, assuming it is brought to
        rest with losses or heat transfer computed as :math:`h_0 = h +
        \frac{1}{2}u^2`. Like all enthalpy values the baseline is arbitrary
        and values from different formulations / classes should not be
        compared."""
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
        r"""Ratio of specific heats :math:`\gamma = c_p/c_v.`"""
        pass

    @property
    @abstractmethod
    def h(self):
        """Specific enthalpy of the gas.

        .. note:: The baseline is arbitrary and values from different
           formulations / classes should not be compared.
        """
        pass

    @property
    @abstractmethod
    def M(self) -> float:
        """Mach number :math:`M = u/a`."""
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
    def rho(self):
        """Density ρ = P/(R.T)."""
        return self.P / (self.R * self.T)

    @property
    @abstractmethod
    def s(self):
        """Specific entropy of the gas.

        .. note:: The baseline is arbitrary and values from different
           formulations / classes should not be compared.
        """
        pass

    @classmethod
    @abstractmethod
    def special_args(cls) -> Sequence[str]:
        """Return a sequence of strings listing any additional arguments
        specific to the derived class needed to fully fully initialise this
        object.  Standard state properties `T`, `T0`, `P`, `P0`, `h`, `h0`,
        `s` or `M` and mass flowrate `w` are not to be included.  Used by
        __repr__, __str__, __format__, new_props().
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


# noinspection PyPep8Naming
class PerfectGasFlow(GasFlow):
    r"""
    A simplified compressible gas, thermally and calorically perfect.
    I.E. cp and R are constant.  Once initialised, properties are fixed.

.. note::
    - Internally, stream pressure, temperature, Mach number and ratio of
      specific heats are the stored flow properties (`P`, `T`, `M`, `γ`).
    - Sepcific enthalpy computation is :math:`h = h_0 + c_p.(T - T_{ref})`.
    - Specific entropy computation is :math:`s = s_0 + c_p.ln(T / T_{ref}) -
      R.ln(P / P_{ref})`.
    """

    def __init__(self, *, P: Dim = None, P0: Dim = None, T: Dim = None,
                 T0: Dim = None, h: Dim = None, h0: Dim = None, s: Dim =
                 None, M: float = None, w: Dim = None, gas: str = 'air',
                 gamma: float = 1.4):
        """
        Construct a thermally and calorically perfect, compressible gas
        flowing in 1-D.  Three state property arguments must be given that
        completely specify the gas state.

        Parameters
        ----------
        P0 : Dim, optional
            Total / stagnation pressure.
        T0 : Dim, optional
            Total / stagnation temperature.
        P : Dim, optional
            Stream / static pressure.
        T : Dim, optional
            Stream / static temperature.
        h : Dim, optional
            Specific enthalpy.
        h0 : Dim, optional
            Total / stagnation enthalpy.
        s : Dim, optional
            Specific entropy.
        M : float, optional
            Mach number.
        w : Dim, optional
            Mass flowrate.  Default = None.
        gas : str, optional
            Gas idenfitifer.  Supported values are:
                - 'air'.
        gamma : float, optional
            Ratio of specific heats.  Commonly used values for air are:
                - γ = 1.4: Atmospheric air, compressors (default).
                - γ = 1.33: Hot air, burners, turbines.
        """
        # Set basic constants.
        self._gamma = gamma
        if gas == 'air':
            R = Dim(287.05287, 'J/kg/K')

            # Defaults below are selected to align with GasFlowWF model for
            # dry air with γ = 1.4.
            self._P_ref = Dim(1, 'bar')
            self._T_ref = Dim(298.15, 'K')  # WAS 1000
            self._h_ref = Dim(720.76, 'kJ/kg')
            self._s_ref = Dim(5.68226, 'kJ/kg/K')
        else:
            raise GasError(f"Unknown gas: {gas}")

        super().__init__(R=R, w=w, gas=gas)

        # If T, P, M are supplied, set them directly.
        if all_not_none(T, P, M) and all_none(P0, T0, h, h0, s):
            self._T, self._P, self._M = make_total_temp(T), P, M
            if float(self._T) < 0.0 or float(self._P) < 0.0 or self._M < 0.0:
                raise GasError(f"Cannot set negative T, P or M.")
            return

        # Otherwise, set stream T, P, M by converging given arguments.
        reqd_props = {}
        for prop_id in ('T', 'h', 'T0', 'h0', 's', 'P', 'P0', 'M'):
            val = locals()[prop_id]
            if val is not None:
                reqd_props[prop_id] = val
        TPM_bounds = ([0.0, 0.0, 0.0],  # [K], [kPa], (float)
                      [+inf, +inf, +inf])
        self._T, self._P, self._M = _converge_TPM(self, TPM_bounds,
                                                  reqd_props)

    @classmethod
    def special_args(cls):
        return 'gas', 'gamma'

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
        r"""Ratio of total (stagnation) pressure to static pressure computed
        using :math:`\frac{P_0}{P} = (1 + \frac{1}{2}(γ - 1)M^2)^{\frac{γ}{γ -
        1}}`, which assumes a perfect gas."""
        return (1 + 0.5 * (self._gamma - 1) *
                self._M ** 2) ** (self._gamma / (self._gamma - 1))

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
        r"""Ratio of total (stagnation) pressure to static temperature
        computed using :math:`\frac{T_0}{T} = 1 + \frac{1}{2}(γ - 1)M^2`,
        which assumes a perfect gas."""
        return 1 + 0.5 * (self._gamma - 1) * self._M ** 2

    @property
    def u(self) -> Dim:
        return self.a * self._M


# ----------------------------------------------------------------------------


# noinspection PyPep8Naming
class GasFlowWF(GasFlow):
    r"""
    Compressible, imperfect gas using polynomials from Walsh & Fletcher §3.
    Includes optional products of combustion.  Properties are fixed once
    initialised.

.. note::
    - Temperature range is 200 K → 2000 K.
    - Fuel-Air Ratio `FAR` = 0.00 → 0.05
    - Non-linear :math:`\gamma`, :math:`c_p`, `h`, `s` dependent on
      temperature (i.e. calorically imperfect).  Stagnation enthalpy,
      temperature and pressure also do not assume a perfect gas.  Stagnation
      properties are computed by bringing the flow to rest isentropically.
    - Internal values stored as SI [K, kPa, ...].
    - Stream / static temperature and pressure and Mach number are the
      internal reference states.
    """

    def __init__(self, *, P: Dim = None, P0: Dim = None, T: Dim = None,
                 T0: Dim = None, h: Dim = None, h0: Dim = None, s: Dim = None,
                 M: float = None, w: Dim = None, gas: str = 'air',
                 FAR: float = 0.0):
        """
        Construct a compressible, imperfect gas, based on interpolating
        polynomials given in Walsh & Fletcher, "Gas Turbine Performance",
        Second Edition, Chapter 3.

        .. note::
            - Three state property arguments (`T`, `h`, `T0`, `h0`, `s`,
              `P`, `P0`, `M`) are required, but can be supplied in any
              combination that properly defines the gas.
            - If `P` and `T` are supplied these are directly set and
              initialisation is complete.  All other combinations result in
              an iterative convergence using the values of `T` [K],
              `P` [kPa] and `M`.
            - Non-zero `FAR` normally accompanies gases including products of
              combustion but this is not enforced.  This is because this
              combinatinon may be useful in some hypothetical or
              debugging scenarios.

        Parameters
        ----------
        P : Dim, optional
            Stream / static pressure.
        P0: Dim, optional
            Total / stagnation pressure (also P0).
        T : Dim, optional
            Stream / static temperature.
        T0 : Dim, optional
            Total /stagnation temperature (also T0).
        h : Dim, optional
            Specific enthalpy.
        h0 : Dim, optional
            Total / stagnation enthalpy.
        s : Dim, optional
            Specific entropy.
        M : float, optional
            Mach number.
        w: Dim, optional
            Mass flowrate.  Default = None.
        gas : str, optional
            Gas identifier.  Default = 'air'.  Available `gas` values are:

            - 'air': Dry air
            - 'burned_kerosene': Products of combustion for kerosene in dry
              air.
            - 'burned_diesel': Products of combustion for diesel in dry air.
        FAR : float, optional
            Fuel-Air Ratio.  Default = 0.0.

        Raises
        ------
        GasError
            On invalid arguments.
        """
        # Set simple fixed values and gas model coefficients first.
        self._FAR = FAR
        if not (0 <= self._FAR <= 0.05):
            raise GasError(f"Fuel-Air Ratio invalid: {self._FAR}")

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
            raise GasError(f"Invalid gas: {gas}")

        super().__init__(R=R, w=w, gas=gas)

        # Setup lazily evaluted internal parameters.
        self._cp, self._gamma, self._h, self._cptint = [None] * 4
        self._T0, self._P0 = [None] * 2

        # If T, P, M are supplied, set them directly.
        if all_not_none(T, P, M) and all_none(T0, P0, h, h0, s):
            self._T, self._P, self._M = make_total_temp(T), P, M

            if not (Dim(200, 'K') <= self._T <= Dim(2000, 'K')):
                raise GasError(f"Out of temperature range (200 K -> 2000 "
                               f"K): {self._T}")
            if float(self._P) < 0:
                raise GasError(f"Cannot set negative P or M.")
            return

        # All other cases are solved by converging T, P, M.
        reqd_props = {}
        for prop_id in ('T', 'h', 'T0', 'h0', 's', 'P', 'P0', 'M'):
            val = locals()[prop_id]
            if val is not None:
                reqd_props[prop_id] = val
        TPM_bounds = ([200.0, 0.0, 0.0],  # [K], [kPa], (float)
                      [2000.0, +inf, +inf])
        self._T, self._P, self._M = _converge_TPM(self, TPM_bounds,
                                                  reqd_props)

    # Normal Methods / Properties --------------------------------------------

    @property
    def cp(self) -> Dim:
        if self._cp is None:
            self._cp = self._cp_from_T(self._T)
        return self._cp

    @property
    def FAR(self) -> float:
        """Fuel-Air Ratio :math:`FAR = w_f / w_{total}` where :math:`w_f` is
        the massflow of fuel products of combustion and :math:`w_{total}` is
        the total massflow.  E.G. If the upstream flow was pure airflow of
        :math:`w_{air}` then :math:`FAR = w_f / (w_f + w_{air})`."""
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

    @classmethod
    def special_args(cls):
        return 'gas', 'FAR'

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
        cp = sum([a_i * Tz ** i
                  for i, a_i in enumerate(self._coeff_a[0:9], 0)])
        if self._coeff_b is not None:
            cp += (self._FAR / (1 + self._FAR)) * sum(
                [b_i * Tz ** i
                 for i, b_i in enumerate(self._coeff_b[0:8], 0)])
        return Dim(1000 * cp, 'J/kg/K')

    def _cptint_from_T(self, T: Dim) -> Dim:
        """Compute integral(cp/T) [kJ/kg/K] using W&F Eqn F3.28 with EJW
        correction term given T, return kJ/kg/K. self._coeff_a and
        self._coeff_b must be set prior to call."""
        Tz = T.convert('K').value / 1000
        EJW_A0_corr_term = self._coeff_a[0] * log(1000)
        cptint = ((self._coeff_a[0] * log(Tz)) + sum(
            [(a_i / i) * Tz ** i
             for i, a_i in enumerate(self._coeff_a[1:9], 1)]) +
                  self._coeff_a[10]) + EJW_A0_corr_term
        if self._coeff_b is not None:
            EJW_B0_corr_term = self._coeff_b[0] * log(1000)
            # XXX TODO VERIFY FAR CORR TERM WITH EXAMPLES
            cptint += (self._FAR / (1 + self._FAR)) * (
                    self._coeff_b[0] * log(Tz) +
                    sum([(b_i / i) * Tz ** i
                         for i, b_i in enumerate(self._coeff_b[1:8], 1)]) +
                    self._coeff_b[9] + EJW_B0_corr_term)
        return Dim(cptint, 'kJ/kg/K')

    def _h_from_T(self, T: Dim) -> Dim:
        """Compute enthalpy [MJ/kg] using W&F Eqn F3.26, given T and return
        kJ/kg/K.  self._coeff_a and self._coeff_b must be set prior to call."""
        Tz = T.convert('K').value / 1000
        h = self._coeff_a[9] + sum([(a_i / i) * Tz ** i for i, a_i in
                                    enumerate(self._coeff_a[0:9], 1)])
        if self._coeff_b is not None:
            h += (self._FAR / (1 + self._FAR)) * (sum(
                [(b_i / i) * Tz ** i
                 for i, b_i in enumerate(self._coeff_b[0:7], 1)]) +
                                                  self._coeff_b[8])
        return Dim(1000 * h, 'kJ/kg')

    def _reset_lazy(self):
        self._cp, self._gamma, self._h, self._cptint = [None] * 4
        self._T0, self._P0 = [None] * 2

    def _set_stagnation(self) -> None:
        h0_target = self.h0
        s0_target = self.s

        # Find T that gives h0 using fixed point method and approx cp.
        def update_T(try_T):
            return ((h0_target - self._h_from_T(try_T)) / Dim(1005, 'J/kg/K')
                    + try_T)

        self._T0 = fixed_point(update_T, x0=self._T, xtol=Dim(1e-5, 'K'),
                               maxits=20)
        # Calculate P to give equal entropy.
        # P0 = P_Ref * exp((int(cp/T).dT - s0) / R)
        cptint0 = self._cptint_from_T(self._T0)
        self._P0 = _WF_P_REF_S * exp((cptint0 - s0_target) / self.R)


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

_last_TPM = None  # Last converged gas solution [K, kPa, M].


# noinspection PyPep8Naming,PyUnresolvedReferences
def _converge_TPM(proto_gas: GasFlow, TPM_bounds: Tuple[List, List],
                  reqd_props: Dict[str, Dim]):
    """
    Function to converge T, P, M for a given prototype gas by satisfying the
    required flow properties using an iterative equation solver.

    Any T value in reqd_props will be converted to a total temperature.

    Because it is  likely that the next requested gasflow will be similar to
    the last, if the last convergence was successful we use the previous T,
    P and M and use this as a starting point.  We also try the standard
    atmosphere, M=0.5 starting point if that does not converge.
    """
    global _last_TPM

    # Make up pools with all the possible flow properties that can be
    # affected by T, P or M.  This algorithm was inspired by Zoe Whitney
    # (3 January 2020).
    reqd_ids = set(reqd_props.keys())
    draft_ids = [reqd_ids & {'T', 'T0', 'h', 'h0', 's'},
                 reqd_ids & {'P', 'P0', 's'},
                 reqd_ids & {'M', 'T0', 'P0', 'h0'}]

    # Move first element from shortest pool to final prop list until done.
    prop_id = [None, None, None]
    while any(draft_ids):
        draft_sizes = [len(x) for x in draft_ids]
        short_idx = draft_sizes.index(max(min(draft_sizes), 1))
        next_prop = draft_ids[short_idx].pop()
        prop_id[short_idx] = next_prop
        for idx in 0, 1, 2:  # Remove from all pools.
            draft_ids[idx].discard(next_prop)

    argreq_str = ', '.join(x if x else 'None' for x in prop_id)
    if len(reqd_props) != 3 or not all(prop_id):
        raise GasError(f"Could not make required property vector from "
                       f"arguments (in TPM order): {argreq_str}")

    # Subtractions require total temperatures.
    for k in reqd_props:
        if k == 'T' or k == 'T0':
            reqd_props[k] = make_total_temp(reqd_props[k])

    # noinspection PyPep8Naming
    def prop_residual(try_TPM):
        try_gas = proto_gas.new_props(T=Dim(try_TPM[0], 'K'),
                                      P=Dim(try_TPM[1], 'kPa'),
                                      M=try_TPM[2])
        return [float(getattr(try_gas, prop_id[i]) - reqd_props[prop_id[i]])
                for i in (0, 1, 2)]

    # Build a list of possible starting points.
    x0_list = []
    if _last_TPM is not None:
        # Add last point if converged.  Refine it using direct values from
        # required properties if available.
        if 'T' in reqd_props:
            _last_TPM[0] = reqd_props['T'].convert('K').value
        if 'P' in reqd_props:
            _last_TPM[1] = reqd_props['P'].convert('kPa').value
        if 'M' in reqd_props:
            _last_TPM[2] = reqd_props['M']
        x0_list.append(_last_TPM)
    x0_list.append([288.15, 101.325, 0.2])  # SSL (1st iter or fallback).

    for x0 in x0_list:
        try:
            # Most points only require a few iterations.  The maxits value
            # used represents a few weakly convergent combinations e.g.
            # h0, s, P, etc.
            final_TPM = solve_dqnm(prop_residual, x0=x0, ftol=1e-3, xtol=1e-4,
                                   jacob_diag=[1.0, 1.0, 5.0], order=2,
                                   bounds=TPM_bounds, maxits=50)
            # Converged.  Move the start point and return.
            _last_TPM = final_TPM
            return (Dim(final_TPM[0], 'K'), Dim(final_TPM[1], 'kPa'),
                    final_TPM[2])
        except (RuntimeError, GasError):
            _last_TPM = None  # Failed, reset start point.

    raise GasError(f"Could not converge gas with requested properties "
                   f"(in TPM order): {argreq_str}")
