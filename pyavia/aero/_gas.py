"""
Models for gases including 1-D flow analysis.
"""
# Last updated: 15 January 2021 by Eric J. Whitney

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from scipy import optimize

from pyavia.core.units import dim, Dim
from pyavia.core.util import split_dict


# -----------------------------------------------------------------------------


# noinspection PyPep8Naming
class Gas(ABC):
    """
    Abstract Base class representing common properties of gases, including
    1-D flow properties. ``Gas`` objects are not intended to be modified
    after initialisation - they simply allow access to all the gas
    properties, which are fixed.
    """

    def __init__(self, *, R: Dim, w: Dim = dim(0.0, 'kg/s'), FAR: float = 0.0):
        """
        Parameters
        ----------
        R : Dim
            Gas constant for the gas (i.e. :math:`R = R_{univ}/n`).
        w : Dim
            Mass flowrate.
        FAR: float
            Fuel-Air Ratio :math:`FAR = w_f/w_{total}`.  Non-zero `FAR` occurs
            when gases include products of combustion.
        """
        self._R, self._w, self._FAR = R, w, FAR

        # Greek letter aliases are created here to allow derived classes to
        # override the properties.
        cls = type(self)
        cls.γ = cls.gamma
        cls.ρ = cls.rho

    # -- Properties ---------------------------------------------------------

    @property
    def a(self) -> Dim:
        r"""
        Local speed of sound :math:`a = \sqrt{{\gamma}RT}`.
        """
        # Note the multipliation order: Dim objects on LHS and gamma on RHS
        # allows Dim.__mul__ to be called instead of NumPy ufunc.
        return (self._R * self.T * self.gamma) ** 0.5

    @property
    @abstractmethod
    def c_p(self) -> Dim:
        """
        Specific heat capacity at constant pressure of the gas.
        """
        raise NotImplementedError

    @property
    def c_v(self) -> Dim:
        """
        Specific heat capacity at constant volume of the gas.
        """
        return self.c_p - self.R

    @property
    def FAR(self) -> float:
        """
        Fuel-Air Ratio :math:`FAR = w_f / w_{total}` where :math:`w_f`
        is the massflow of fuel products of combustion and :math:`w_{total}`
        is the total massflow.  E.G. If the upstream flow was pure
        airflow of :math:`w_{air}` then :math:`FAR = w_f / (w_f + w_{air})`.
        The base class implementation simply returns the initialisation
        value."""
        return self._FAR

    @property
    @abstractmethod
    def gamma(self) -> float:
        r"""
        Ratio of specific heats :math:`\gamma = c_p/c_v.`
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def h(self) -> Dim:
        """
        Specific enthalpy of the gas.

        .. note:: The baseline is arbitrary and values from different
           formulations / classes should not be compared.
        """
        raise NotImplementedError

    @property
    def h0(self) -> Dim:
        r"""
        Total / stagnation enthalpy of the gas, assuming it is brought to
        rest with losses or heat transfer computed as :math:`h_0 = h +
        \frac{1}{2}u^2`. Like all enthalpy values the baseline is arbitrary
        and values from different formulations / classes should not be
        compared.
        """
        return self.h + 0.5 * self.u ** 2

    @property
    @abstractmethod
    def M(self) -> float:
        """Mach number :math:`M = u/a`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def P(self) -> Dim:
        """
        Stream / static pressure.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def P0(self):
        """
        Total / stagnation pressure.
        """
        raise NotImplementedError

    @property
    def Q(self):
        """
        Enthalpy flowrate Q = h * w.  Units determined by multiplication.
        """
        return self.h * self._w

    @property
    def R(self) -> Dim:
        """
        Gas constant R for the specific gas. R_specific = R_universal / M
        where R = 8.314462618 kg.m².s⁻².K⁻¹.mol⁻¹ and M is molar mass
        [kg/mol].
        """
        return self._R

    @property
    def rho(self) -> Dim:
        """
        Density ρ = P/(R.T).
        """
        return self.P / (self.R * self.T)

    @property
    @abstractmethod
    def s(self) -> Dim:
        """
        Specific entropy of the gas.

        .. note:: The baseline is arbitrary and values from different
           formulations / classes should not be compared.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def T(self) -> Dim:
        """
        Stream / static temperature.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def T0(self) -> Dim:
        """
        Total / stagnation temperature.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def u(self) -> Dim:
        """
        Flow velocity.
        """
        raise NotImplementedError

    @property
    def w(self) -> Dim:
        """
        Mass flowrate.
        """
        return self._w


# =============================================================================


def _solve_TPM(flow: Type[Gas], *, init_gas: Gas = None,
               **kwargs) -> (Dim, Dim, float):
    """
    Return `T`, `P` and `M` values that produce a flow with the required
    state defined as part of `kwargs`.  This is done by creating a series of
    `flow` objects with trial values, until a match is achieved.

    `init_gas` can act (when provided) acts as an initialisation point to
    help convergence.
    """
    state, kwargs = split_dict(kwargs, ('P0', 'T0', 'P', 'T',
                                        'h', 'h0', 's', 'M'))

    if len(state) != 3:
        raise ValueError(f"Need three state properties to uniquely specify "
                         f"flow, got: {', '.join(list(state.keys()))}")

    # Separate out the remaining target state from the known and
    # unknown T, P, M values.
    target_state = {}
    known_vars = {}
    unknown_vars = {'T', 'P', 'M'}
    for name, value in state.items():
        if name in unknown_vars:
            # If T, P or M were provided directly, move it to the knowns.
            known_vars[name] = value
            unknown_vars.remove(name)

        else:
            # Otherwise add it to the target state.
            target_state[name] = value

    # Shouldn't reach this point with no unknowns or target state.
    if not target_state or not unknown_vars:
        raise ValueError(f"Invalid state specified - no target state or "
                         f"unknown values resulted.")

    # Get unknown starting values, units and scale factors.
    unknown_vars = list(unknown_vars)  # Fix order.
    x0, x_units = [], []
    for name in unknown_vars:
        if name == 'P':
            var_ref = init_gas.P.convert('kPa') if init_gas else dim(100, 'kPa')
        elif name == 'T':
            var_ref = init_gas.T.convert('K') if init_gas else dim(298.15, 'K')
        else:
            var_ref = init_gas.M if init_gas else dim(0.50)

        x0.append(var_ref.value)
        x_units.append(var_ref.units)

    x0 = np.array(x0, dtype=np.float64)

    def make_TPM(x: [float]) -> {str, Dim}:
        # Function to dimensionalise / complete the trial point 'x' including
        # the known values.
        all_x = known_vars.copy()
        for name_, x_i, units in zip(unknown_vars, x, x_units):
            all_x[name_] = dim(x_i, units)
        return all_x

    def state_err(x: [float]) -> [float]:
        # This function computes the difference / residual between target and
        # current states.

        # Create a flow using the trial T, P, M values and remaining knowns.
        dim_x = make_TPM(x)
        trial_flow = flow(**dim_x, **kwargs)

        # Now compute state errors.
        err = []
        for name_, f_target in target_state.items():
            # Get the current value of the state.  Note the conversion as in
            # the case of temperature subtraction this requires the same basis
            # be used to remove ambiguity.  This also does a sanity check.
            f_i = getattr(trial_flow, name_).convert(f_target.units)
            Δf_i = f_target - f_i
            err.append(Δf_i.to_real())

        return err

    # Solve the equation/s.  NOTE: These solver parameters are quite touchy
    # and need to be finely adjusted to give fast convergence to the result
    # without bouncing past model limits.
    sol = optimize.root(state_err, x0=x0, method='hybr', options={
        'xtol': 1e-9, 'maxfev': 50, 'factor': 0.01, 'diag': 1 / x0,
        'eps': 1e-3})

    if sol.success and all([abs(x) < 1e-6 for x in sol.fun]):
        # We correct tiny negative Mach numbers which can result from
        # 'stepping over' the origin during convergence. We flip these (not
        # zero them) to preserve M² terms.  Larger excursions are cause for
        # concern and these are trapped later.
        final_x = make_TPM(sol.x)
        if -1e-6 <= final_x['M'] < 0:
            final_x['M'] = -final_x['M']

        return final_x['T'], final_x['P'], final_x['M']

    # Something went wrong.
    raise RuntimeError(f"Could not converge gas state.\n"
                       f"\tSolver message: {sol.message}\n"
                       f"\tSolver function evaluations = {sol.nfev}\n"
                       f"\tLast x = {sol.x}\n"
                       f"\tLast f(x) = {sol.fun}")
