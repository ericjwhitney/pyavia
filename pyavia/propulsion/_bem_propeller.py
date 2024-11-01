from __future__ import annotations

from functools import partial
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyavia.util.print_styles import AddDotStyle, val2str
from pyavia.numeric.solve import SolverError
from pyavia.state import InvalidStateError
from pyavia.propulsion import BEPropeller, CSPropeller


# Written by Eric J. Whitney, January 2023.

# ======================================================================

# Future work: 'Buhl' model seems flawed here.
class BEMPropeller(BEPropeller):
    """
    Blade Element Momentum (BEM) propeller model that converges induced
    inflow and swirl factors separately.

    Force and momentum equilibrium conditions are used to give local
    velocities and angles of attack along the blade (method similar to
    [1]_).  See :ref:`Notes <BEM_J_limitation>` for advance ratio
    limitations of this method.

    Parameters
    ----------
    blade_stall : 'du-selig' or None
        Make corrections to the model in cases of blade stall:

        - `None`: No correction applied.
        - 'du-selig' (Default): Correction of Du and Selig is applied
          [3]_.  At this stage the internal constants are fixed as
          ``a = b = d = 1``.

    high_inflow : 'buhl', 'wilson' or None
        Make corrections to the model in cases of high induced inflow:

        - `None`: No correction applied.
        - 'buhl': Correction of Buhl is applied ([2]_ Table 1).
        - 'wilson': Correction of Wilson et al is applied ([2]_
          Table 1).

    tip_losses : 'glauert' or None
            Include Prandtl's tip loss factor `F` when computing the
            momentum balance:

            - 'None': No tip loss correction is applied.
            - 'glauert':  Typical correction of Glauert [4]_ Chapter VII
              Equation 5.6 is used.  At each blade element `i`:

                :math:`f_i = (B/2) (R - r_i) / (r_i sin(ϕ_i))`
                :math:`F_i = (2/π) cos^{-1}(e^{-f})`

              An upper limit of 20 is applied to the exponent argument
              `f` in a similar fasion to `QPROP` and `XROTOR` codes.

    Notes
    -----
    .. _BEM_J_limitation:

    - This is a simple technique that gives reasonably accurate
      performance predictions in forward flight.  The static thrust /
      low velocity case is not handled well however, and advance ratio
      is restricted to `J > 0.0025`.

    - This class requires specification of `β0` directly.

    .. todo:: Version with constant speed operation.

    - See parent class `BEPropeller` for more general workings and
      limitations of Blade Element Methods.

    References
    ----------
    .. [1] This method was originally based on a MATLAB code by D. Auld,
           although it is now significantly modified.  For the original
           see: http://www.aerodynamics4students.com/propulsion/blade-element-propeller-theory.php
    .. [2] J.Ledoux, S. Riffo and J. Salomon, "Analysis of the Blade
       Element Momentum Theory", SIAM Journal on Applied Mathematics,
       2021, 81 (6), pp.2596-2621.
    .. [3] Z. Du and M. Selig, "A 3-D Stall-Delay Model for Horizontal
       Axis Wind Turbine Performance Prediction", 1998 ASME Wind Energy
       Symposium, Reno, NV, U.S.A.
    .. [4] W. F. Durand (ed.), "Aerodynamic Theory", Julius Springer,
       1935.
    """  # noqa

    # -- Inner Classes -------------------------------------------------

    class DefPoints(BEPropeller.DefPoints):
        pass  # To synchronise with outer class.

    class Elements(BEPropeller.Elements):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Initialise additional element values.
            self._a = np.full(self.n, np.nan)
            self._b = np.full(self.n, np.nan)

        # -- Protected Methods -----------------------------------------

        def _init_ab(self, *, init_α: float = 0.0,
                     init_b: float = 0.01):
            """
            For all blade elements initialise the swirl factor `b` to
            `init_b` and the inflow factor `a` to produce the requested
            angle of attack along the blade at the first iteration. `V0`
            and `Ω` must be properly defined in the main propeller for
            this to work.
            """
            self._b = np.full(self.n, init_b)
            self._a = ((self._prop.Ω * self.r / self._prop.V0) *
                       np.tan(self.β - init_α) * (1 - self._b) - 1)

        def _reset(self):
            super()._reset()
            self._a[:] = np.nan
            self._b[:] = np.nan

    # -- Main Class ----------------------------------------------------

    _J_MIN: float = 0.0025  # Smallest advance ratio |J| allowed.

    def __init__(self, *,
                 # Options.
                 maxits_flow: int = 100, tol_flow: float = 1e-4,
                 blade_stall: Literal['du-selig', None] = 'du-selig',
                 high_inflow: Literal['buhl', 'wilson', None] = 'wilson',
                 tip_losses: Literal['glauert', None] = 'glauert',
                 **kwargs):
        super().__init__(maxits_flow=maxits_flow, tol_flow=tol_flow,
                         **kwargs)

        # Setup solution options.
        self.blade_stall = blade_stall
        self.high_inflow = high_inflow
        self.tip_losses = tip_losses

        # Setup formatting for _solve_flow() printed output.
        self.pstyles.add('BEMPropeller', AddDotStyle(),
                         parent='BEPropeller')

        # Set and check state inputs at this level.
        self._bem_propeller_set_state()

    # -- Public Methods ------------------------------------------------

    def set_state(self, **kwargs) -> frozenset[str]:
        changed = super().set_state(**kwargs)
        changed |= self._bem_propeller_set_state()
        return changed

    # -- Private Methods -----------------------------------------------

    def _bem_propeller_set_state(self) -> frozenset[str]:
        # Internal implementation of `set_state` that sets, checks and
        # actions all state variables for only this level in the class
        # hiearchy.
        #
        # 'BEMPropeller' doesn't add any new state parameters, but we
        # add a specific 'J' check.
        if (not self.stopped) and (np.abs(self.J) < self._J_MIN):
            raise InvalidStateError(
                f"{self.__class__.__name__} can't compute near-static "
                f"conditions.  |J| = {np.abs(self.J)} < {self._J_MIN}.")

        return frozenset()

    def _ds_stall_factors(self, a: float = 1.0, b: float = 1.0,
                          d: float = 1.0) -> tuple[npt.NDArray[float],
                                                   npt.NDArray[float]]:
        """
        Returns element-wise factors f_l and f_d from equation 12a and
        12b in Du and Selig (see ``__init__`` for details).  Requires
        the following to be valid: `self.c`, `self.r`, `self.R`,
        `self.Ω`, `self.V0`.

        `a`, `b` and `d` are empirical correction factors can be
        adjusted to best reproduce the requied behaviour.  Defaults
        presently follows the original paper ``a = b = d = 1``.
        """
        r, R = self.elements.r, self.R  # For convenience.
        Λ = ((self.Ω * R) / np.hypot(self.V0, self.Ω * R))  # Mod. TSR.
        c_r = self.elements.c / r  # Chord-radius effect.

        f_l = (0.5 / np.pi) * (
                ((1.6 * c_r * (a - np.power(c_r, (d * R) / (Λ * r)))) /
                 (0.1267 * (b + np.power(c_r, (d * R) / (Λ * r))))) - 1)

        f_d = (0.5 / np.pi) * (
                ((1.6 * c_r * (a - np.power(c_r, (d * R) / (2 * Λ * r)))) /
                 (0.1267 * (b + np.power(c_r, (d * R) / (2 * Λ * r))))) - 1)

        return f_l, f_d

    # noinspection PyProtectedMember
    def _solve_flow(self):
        """
        Calculate performance at the current state by converging inflow
        and swirl factors `a` and `b` at each blade element using
        momentum conditions.  This method is not valid for
        near-stationary  operation (low `|J|`), see `Notes`.

        Raises
        ------
        SolverError
            If the solver fails to converge.
        """
        # Note: All working quantities in this method are per-element
        # unless noted otherwise.

        # Setup quick references.
        ps_print = partial(self.pstyles.print, 'BEMPropeller')
        elems: BEMPropeller.Elements = self._elements
        n_elems = elems.n

        # Initialise swirl / inflow if J or β0 have changed
        # significantly or if any are NaN (tpyically on first use).
        if (np.abs(self.J - self._last_J) > 0.25 or
                np.abs(self.β0 - self._last_β0) > np.deg2rad(5.0) or
                np.any(np.isnan(elems._a)) or np.any(np.isnan(elems._b))):
            ps_print("-> (Re-)Initialising inflow / swirl factors.")
            # noinspection PyProtectedMember
            elems._init_ab()

        # Setup blade stall model paramaters.
        if self.blade_stall == 'du-selig':
            # For later improvement: Set a, b, d here.
            f_l_ds, f_d_ds = self._ds_stall_factors()

        # Setup adaptive relaxation factor and residual recording.
        # Future work: Investigate expanging to relax = 1.0 for easy
        # cases.
        relax = 0.50
        relax_min, relax_max = 0.25, 0.50

        if self.high_inflow and self.J < 0.1:
            # The high inflow model gives poor convergence at low J, so
            # the maximum relaxation factor is reduced to improve
            # stability.
            relax_max = 0.33

        Δa_max, Δb_max = [], []  # Convergence history.

        # -- Main Loop -------------------------------------------------

        for it in range(self.maxits_flow):

            # -- Flow State --------------------------------------------

            Vp_a = self.V0 * (1 + elems._a)
            Vp_t = self.Ω * elems.r * (1 - elems._b)
            Vp_mag = np.hypot(Vp_a, Vp_t)
            ϕ = np.arctan2(Vp_a, Vp_t)  # Flow angle to disc.
            α = elems.β - ϕ  # AoA.
            M = Vp_mag / self.a0  # Mach No.
            if self.μ0 != 0.0:
                Re = self.ρ0 * Vp_mag * elems.c / self.μ0
            else:
                Re = np.full(n_elems, np.inf)  # Inviscid Re = ∞.

            # On first few iterations only, we catch a potential
            # supersonic flow error that can occur if the previous
            # converged point had high inflow making the inherited
            # startup values too severe. In this case inflow/swirl
            # factors are reset to near-zero and the loop restarted.
            if it < 3 and (np.any(M >= 0.99)):
                ps_print("-> Initial conditions supersonic, restarting.")

                # noinspection PyProtectedMember
                elems._init_ab(init_b=0.001)
                continue

            # -- Aerodynamic Coefficients ------------------------------

            # Compute cl, cd at the blade elements.
            for i, foil in enumerate(elems.foils):
                foil.set_state(α=α[i], M=M[i], Re=Re[i])
                elems._cl[i] = foil.cl
                elems._cd[i] = foil.cd

            # Adjust according to stall model.
            if self.blade_stall is None:
                pass

            elif self.blade_stall == 'du-selig':
                # Du-Selig stall model also requires α0 and drag at
                # α = 0.
                α0 = np.full(n_elems, np.nan)
                cd_α0 = np.full(n_elems, np.nan)
                for i, foil in enumerate(elems.foils):
                    # Resetting state here covers the case where a
                    # single aerofoil reference might be used.  It
                    # should be cheap if the state hasn't changed.
                    foil.set_state(α=0, M=M[i], Re=Re[i])
                    α0[i] = foil.α0
                    cd_α0[i] = foil.cd

                # Calculate lift and drag corrections assuming the basic
                # aerofoils are operating in the linear thin inviscid
                # region.
                cl_p = 2 * np.pi * (α - α0)
                Δcl = f_l_ds * (cl_p - elems._cl)  # noqa: F823
                Δcd = f_d_ds * (elems._cd - cd_α0)  # noqa: F823

                # Future improvement:  Add a check to see if each
                # aerofoil is actually stalled before combining.
                elems._cl += Δcl
                elems._cd += Δcd

            else:
                raise ValueError(f"Unknown blade stall model "
                                 f"'{self.blade_stall}'.")

            # -- Tip Losses --------------------------------------------

            if self.tip_losses is None:
                F = np.ones(n_elems)

            elif self.tip_losses == 'glauert':
                # Glauert Chapter VII Eqn 5.6.
                f = ((self.B / 2) * (self.R - elems.r) /
                     (elems.r * np.sin(ϕ)))
                # Future work: Check if clipping is still necessary.
                f = np.clip(f, None, 20.0)  # Per XROTOR, QPROP.
                F = 2 / np.pi * np.arccos(np.exp(-f))

            else:
                raise ValueError(f"Unknown tip loss model "
                                 f"'{self.tip_losses}'.")

            # -- Element and Total Forces ------------------------------

            self._update_forces(Vp_mag, ϕ)

            # -- High Inflow Model -------------------------------------

            if self.high_inflow is None:
                # Variable 'ψ' in Ledoux renamed 'H' to avoid confusion.
                H = np.zeros(n_elems)

            elif self.high_inflow == 'buhl':
                # See Ledoux Table 1.
                a_c = 0.4
                Δa_upr = np.maximum(elems._a - a_c, 0.0)
                H = (1 / (2 * F)) * (Δa_upr / (1 - a_c)) ** 2

            elif self.high_inflow == 'wilson':
                # See Ledoux Table 1.
                a_c = 1 / 3
                Δa_upr = np.maximum(elems._a - a_c, 0.0)
                H = Δa_upr ** 2

            else:
                raise ValueError(f"Unknown high inflow model "
                                 f"'{self.high_inflow}'.")

            # -- Inflow / Swirl Residual -------------------------------

            # Get revised estimates for the inflow and swirl
            # coefficients using rearranged momentum equations.
            a_new = ((elems._dT_dr /
                      (4.0 * np.pi * elems.r * self.ρ0 *
                       (self.V0 ** 2) * F) - H) / (1 + elems._a))

            b_new = (elems._dQs_dr /
                     (4.0 * np.pi * elems.r ** 3 * self.ρ0 * self.V0 *
                      F * (1 + elems._a) * self.Ω))

            # Calculate residuals and check convergence.
            Δa = a_new - elems._a
            Δb = b_new - elems._b
            Δa_max.append(np.max(np.abs(Δa)))
            Δb_max.append(np.max(np.abs(Δb)))

            if Δa_max[-1] <= self.tol_flow and Δb_max[-1] <= self.tol_flow:
                return

            # More iterations are required.
            ps_print(f"Iteration {it + 1:4d} of {self.maxits_flow:4d}: "
                     f"T = {val2str(self.T)}, Qs = {val2str(self.Qs)}, "
                     f"Max Resid. Inflow (Δa) = {Δa_max[-1]:11.05E}, "
                     f"Swirl (Δb) = {Δb_max[-1]:11.05E}, "
                     f"Relax = {relax:5.03f}")

            # -- Solution Correction -----------------------------------

            # Adapt relaxation based on the relative improvement in the
            # residuals. Future improvements: See M. Jin, X. Yang, "A
            # new fixed-point algorithm to solve the blade element
            # momentum equations with high robustness", 2021 for a
            # method which may help the corner cases.

            if len(Δa_max) > 2 and len(Δb_max) > 2:
                # Calculate historic step ratios.
                Δa_facs = [Δa_1 / Δa_2 for Δa_1, Δa_2 in
                           zip(Δa_max[:-1], Δa_max[1:])]
                Δb_facs = [Δb_1 / Δb_2 for Δb_1, Δb_2 in
                           zip(Δb_max[:-1], Δb_max[1:])]

                # Use weighted factor over last two steps.
                wt = (0.5, 0.5)
                Δa_fac = wt[0] * Δa_facs[-1] + wt[1] * Δa_facs[-2]
                Δb_fac = wt[0] * Δb_facs[-1] + wt[1] * Δb_facs[-2]
                max_Δ_fac = max(Δa_fac, Δb_fac)

                # Convergence:
                #   - Average: Hold steady.
                #   - Poor: Reduce factor in steps.
                #   - Good: Increase factor in steps.
                if max_Δ_fac > 1.00:
                    relax /= 1.1

                elif max_Δ_fac < 0.90:
                    relax *= 1.1

                relax = np.clip(relax, relax_min, relax_max)

            # Adjust inflow / swirl for next pass.
            elems._a += relax * Δa
            elems._b += relax * Δb

        # --------------------------------------------------------------

        raise SolverError("Reached iteration limit without converging "
                          "inflow / swirl.")


# ----------------------------------------------------------------------

class BEMPropellerCS(CSPropeller, BEMPropeller):
    """Constant speed operation of `BEMPropeller`."""
    pass
