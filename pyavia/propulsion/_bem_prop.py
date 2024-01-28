from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pyavia.iter import any_none
from pyavia.propulsion._generic_be_prop import GenericBEProp
from pyavia.type_ext import make_sentinel
from pyavia.solve.exception import SolverError
from pyavia.util import disp_enter, disp_print, disp_exit

_NOT_READY = make_sentinel('_NOT_READY')


# ===========================================================================

class BEMPropeller(GenericBEProp):
    """
    Blade Element Momentum (BEM) propeller model that converges induced
    inflow and swirl factors separately under force and momentum
    equilibrium conditions [1]_.  This gives local velocities and angles of
    attack along the blade. This is a simple technique that still gives
    reasonably accurate performance predictions. Note the following:

        - The static thrust case is not handled well, and advance ratio
          is restricted to `J > 0.0025'.

        - Various stall and high inflow correction models can be used,
          see ``__init__`` for details.

        - See `GenericBEProp` for more general workings and limitations of
          Blade Element Methods.

    Notes
    -----
    .. [1] This method was originally based on a MATLAB code by D. Auld,
       although is now significantly modified.  For the original see:
       http://www.aerodynamics4students.com/propulsion/blade-element
       -propeller-theory.php
    """

    def __init__(self, *args,
                 tip_losses: Optional[str] = 'glauert',
                 maxits_flow: int = 100, tol_flow: float = 1e-4,
                 high_inflow: Optional[str] = 'wilson',
                 blade_stall: Optional[str] = 'du-selig',
                 **kwargs):
        """
        Parameters
        ----------
        r, c, βr, B, foils, β0, pitch_control, redist
            See ``GenericBEProp.__init__`` for general geometry parameters.

        ρ0, a0, μ0
            See ``GenericBEProp.__init__`` for atmospheric parameters.

        tip_losses : str, default = 'glauert'
            Include Prandtl's tip loss factor `F` when computing the
            momentum balance:
            - None: No tip loss correction is applied.
            - 'glauert':  Typical correction of Glauert [3]_ Equation 5.6 is
              applied.  The exponendt argument `f` is limited to 20 (
              similar to `QPROP` / `XROTOR`).

        maxits_cs, tol_cs
            See ``GenericBEProp.__init__`` for pitch control convergence
            parameters.

        maxits_flow : int, default = 100
            Maximum number of iterations for convergence of inflow / swirl
            factors `a` and `b`.  For normal conditions generally < 10
            iterations should be required (only unusual cases should
            require > 50).

        tol_flow : float, default = 1e-4
            Flow is considered converged when the maximum change in inflow
            and swirl parameters (`Δa`, `Δb`) is below this value.

        high_inflow : str | None, default = 'wilson'
            Make corrections to the model in cases of high induced inflow:
            - `None`: No correction applied.
            - 'buhl': Correction of Buhl is applied ([1]_ Table 1).
            - 'wilson': Correction of Wilson et al is applied ([1]_ Table 1).

        blade_stall : str | None, default = 'du-selig'
            Make corrections to the model in cases of blade stall:
            - `None`: No correction applied.
            - 'du-selig': Correction of Du and Selig is applied [2]_.  At
              this stage the internal constants are fixed as ``a = b = d =
              1``.

        Notes
        -----
        .. [1] J.Ledoux, S. Riffo and J. Salomon, "Analysis of the Blade
           Element Momentum Theory", SIAM Journal on Applied Mathematics,
           2021, 81 (6), pp.2596-2621.
        .. [2] Z. Du and M. Selig, "A 3-D Stall-Delay Model for Horizontal
           Axis Wind Turbine Performance Prediction", 1998 ASME Wind Energy
           Symposium, Reno, NV, U.S.A.
        .. [3] W. F. Durand (ed.), "Aerodynamic Theory", Julius Springer,
           1935.
        """
        super().__init__(*args, maxits_flow=maxits_flow, tol_flow=tol_flow,
                         **kwargs)

        # Setup formulation specific parameters.
        self._tip_losses = tip_losses  # Checked in _run_fixed().
        self._high_inflow = high_inflow
        self._blade_stall = blade_stall

        # Saved values for quick restart.
        self._a, self._b = _NOT_READY, _NOT_READY  # Element-wise.
        self._last_J_fixed, self._last_β0_fixed = np.inf, self._β0

    # -- Private Methods ----------------------------------------------------

    def _calc_ds_const(self, a: float = 1.0, b: float = 1.0,
                       d: float = 1.0) -> (NDArray, NDArray):
        """
        Returns element-wise factors f_l and f_d from equation 12a and 12b
        in Du and Selig (see ``__init__`` for details).  Requires the
        following to have been updated / valid: self._c, self._r, self.R,
        self._Ω, self._V0.

        `a`, `b` and `d` are empirical correction factors set to best
        reproduce the requied behaviour.  Defaults presently follows the
        original paper ``a = b = d = 1``.
        """
        r, R = self._r, self.R  # For convenience.
        Λ = ((self._Ω * R) /  # Modified tip speed ratio.
             np.sqrt(self._V0 ** 2 + (self._Ω * R) ** 2))
        c_r = self._c / r  # Chord-radius effect.

        f_l = (0.5 / np.pi) * (
                ((1.6 * c_r * (a - np.power(c_r, (d * R) / (Λ * r)))) /
                 (0.1267 * (b + np.power(c_r, (d * R) / (Λ * r))))) - 1)

        f_d = (0.5 / np.pi) * (
                ((1.6 * c_r * (a - np.power(c_r, (d * R) / (2 * Λ * r)))) /
                 (0.1267 * (b + np.power(c_r, (d * R) / (2 * Λ * r))))) - 1)

        return f_l, f_d

    def _run_fixed(self, disp: int | bool = None):
        """
        Calculate performance assuming fixed pitch (i.e. holding β0
        constant), using momentum conditions computing inflow and swirl
        factors `a` and `b`.  Not valid for near-stationary operation e.g.
        |J| < 0.0025.
        """
        # All working quantities in this method are per-element unless noted
        # otherwise.
        if any_none(self._V0, self._Ω, self._ρ0, self._a0):
            raise ValueError("Flow state not initialised, requires "
                             "V, Ω, ρ, a.")

        if np.abs(self.J) < 0.0025:
            raise ValueError("Can't compute stationary propellers using "
                             "this method.")

        disp_enter(disp)
        disp_print(f"Solving blade inflow / swirl (J = {self.J:.3f}, "
                   f"β0 = {np.rad2deg(self.β0):.01f}°):")

        # Setup adaptive relaxation factor and residual recording.
        if np.abs(self.J) <= 0.1:
            # Low-speed props can be difficult to converge; start cautiously.
            relax = 0.15
        else:
            relax = 0.5

        Δa_max, Δb_max = np.inf, np.inf

        if (self._a is _NOT_READY or
                np.abs(self.J - self._last_J_fixed) > 0.25 or
                np.abs(self._β0 - self._last_β0_fixed) > np.deg2rad(5.0)):
            disp_print("\t-> (Re-)Initialising inflow / swirl factors.")
            self._set_ab_factors(init_a=0.1)

        # Setup blade stall model paramaters.
        if self._blade_stall == 'du-selig':
            # For later improvement: Set a, b, d here.
            f_l_ds, f_d_ds = self._calc_ds_const()

        # -- Main Loop ------------------------------------------------------

        for it in range(self.maxits_flow):

            # -- Flow State -------------------------------------------------

            Vp_a = self._V0 * (1 + self._a)  # Axial vel.
            Vp_t = self._Ω * self._r * (1 - self._b)  # Tangent vel.
            Vp_mag = np.sqrt(Vp_a ** 2 + Vp_t ** 2)
            self._M = Vp_mag / self._a0  # Mach No.
            ϕ = np.arctan2(Vp_a, Vp_t)  # Flow angle to disc.
            self._α = self._β0 + self._βr - ϕ  # AoA.

            # On first few iterations only, we catch a potential supersonic
            # flow error that can occur if the previous converged point had
            # high inflow making the inherited startup values too severe.
            # In this case inflow/swirl factors are reset to near-zero and
            # the loop restarted.
            if it < 3 and (np.any(self._M >= 0.99)):
                disp_print(f"\t-> Initial conditions supersonic, restarting.")
                self._set_ab_factors(init_a=0.001)
                continue

            if self._μ0 != 0.0:
                self._Re = self._ρ0 * Vp_mag * self._c / self._μ0
            else:
                self._Re = np.full_like(Vp_mag, np.inf)

            # -- Aerodynamic Coefficients -----------------------------------

            if self._blade_stall is None:
                # Standard solution only requires cl, cd.
                self._cl, self._cd = self._calc_aero(['cl', 'cd'])

            elif self._blade_stall == 'du-selig':
                # Du-Selig stall model also requires α0 and drag at α = 0.
                # 'cd_α0' is listed last to avoid having to re-run the
                # aerofoil at the original α.
                self._cl, self._cd, α0, cd_α0 = self._calc_aero(
                    ['cl', 'cd', 'α0', 'cd_α0'])

                # Calculate lift and drag corrections assuming the basic
                # aerofoils are operating in the linear thin inviscid region.
                cl_p = 2 * np.pi * (self._α - α0)
                Δcl = f_l_ds * (cl_p - self._cl)  # noqa: F823
                Δcd = f_d_ds * (self._cd - cd_α0)  # noqa: F823

                # Future improvement:  Add a check to see if each aerofoil
                # is actually stalled before combining.
                self._cl += Δcl
                self._cd += Δcd

            else:
                raise ValueError(f"Unknown blade stall model "
                                 f"'{self._blade_stall}'.")

            # -- Tip Losses -------------------------------------------------

            if self._tip_losses is None:
                F = np.ones_like(self._r)

            elif self._tip_losses == 'glauert':
                # Glauert Eqn 5.6
                f_tip = ((self._B / 2) * (self.R - self._r) /
                         (self.R * np.sin(ϕ)))
                f_tip = np.clip(f_tip, None, 20.0)  # Per XROTOR / QPROP.
                F = 2 / np.pi * np.arccos(np.exp(-f_tip))

            else:
                raise ValueError(f"Unknown tip loss model "
                                 f"'{self._tip_losses}'.")

            # -- Element and Total Forces -----------------------------------

            self._calc_forces(Vp_mag, ϕ)

            # -- High Inflow Model ------------------------------------------

            if self._high_inflow is None:
                # Variable 'ψ' in Ledoux renamed 'H' to avoid confusion.
                H = np.zeros_like(self._r)

            elif self._high_inflow == 'buhl':
                # See Ledoux Table 1.
                a_c = 0.4
                Δa_upr = np.maximum(self._a - a_c, 0.0)
                H = (1 / (2 * F)) * (Δa_upr / (1 - a_c)) ** 2

            elif self._high_inflow == 'wilson':
                # See Ledoux Table 1.
                a_c = 1/3
                Δa_upr = np.maximum(self._a - a_c, 0.0)
                H = Δa_upr ** 2

            else:
                raise ValueError(f"Unknown high inflow model "
                                 f"'{self._high_inflow}'.")

            # -- Inflow / Swirl Residual ------------------------------------

            # Get revised estimates for the inflow and swirl coefficients
            # using rearranged momentum equations.
            a_new = ((self._dT_dr /
                      (4.0 * np.pi * self._r * self._ρ0 *
                       (self._V0 ** 2) * F) - H) / (1 + self._a))

            b_new = self._dQs_dr / (
                    4.0 * np.pi * self._r ** 3 * self._ρ0 * self._V0 *
                    F * (1 + self._a) * self._Ω)

            # Calculate residuals and check convergence.
            Δa, Δb = a_new - self._a, b_new - self._b
            Δa_max_prev, Δb_max_prev = Δa_max, Δb_max
            Δa_max, Δb_max = np.max(np.abs(Δa)), np.max(np.abs(Δb))

            if Δa_max <= self.tol_flow and Δb_max <= self.tol_flow:
                # Converged - Save state and return.
                self._last_J_fixed, self._last_β0_fixed = self.J, self._β0
                disp_print(f"\t-> Converged: T = {self._T:+.1f}, "
                           f"Qs = {self._Qs:+.1f}")
                disp_exit()
                return

            # More iterations are required.
            disp_print(f"\t... Iteration {it + 1:4d} of"
                       f" {self.maxits_flow:4d}: "
                       f"T = {self._T:+8.01f}, Qs = {self._Qs:+8.01f}, "
                       f"Max Resid. Inflow (Δa) = {Δa_max:11.05E}, "
                       f"Swirl (Δb) = {Δb_max:11.05E}, "
                       f"Relax = {relax:5.03f}")

            # -- Solution Correction ----------------------------------------

            # Adapt relaxation based on the relative improvement in the
            # residuals. Future improvements: See M. Jin, X. Yang, "A new
            # fixed-point algorithm to solve the blade element momentum
            # equations with high robustness", 2021 for a method which may
            # help the corner cases.
            max_Δ_fac = np.max([Δa_max / Δa_max_prev, Δb_max / Δb_max_prev])
            if max_Δ_fac >= 1.0:
                relax = 0.15  # Drop relaxation factor for wrong direction.
            else:
                relax *= 1.25  # Creep back up over 5 iterations.

            relax = np.clip(relax, 0.15, 0.50)

            # Adjust inflow / swirl for next pass.
            self._a += relax * Δa
            self._b += relax * Δb

        # -------------------------------------------------------------------

        disp_exit()
        raise SolverError("Reached iteration limit without converging "
                            "inflow / swirl.")

    def _set_ab_factors(self, init_α: float = 0.0, init_a: float = 0.1):
        """
        The inflow factor `a` is initialised to `init_a` and the swirl
        factor `b` is initialised to produce the requested angle of attack
        along the blade at the first iteration. self._V0 and self._Ω must
        have been set for this to work.
        """
        self._a = np.full_like(self._r, init_a)
        self._b = 1 - ((self._V0 * (1 + self._a)) /
                       (self._Ω * self._r *
                        np.tan(self._β0 + self._βr - init_α)))
