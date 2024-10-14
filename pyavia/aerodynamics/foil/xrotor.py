import warnings

import numpy as np
from scipy.optimize import root_scalar

from .base import Foil2DBasic
from .stall import PostStall2DMixin
from pyavia.numeric.solve import step_bracket_root, SolverError


# Written by Eric J. Whitney, January 2023.

# ======================================================================

class XROTOR2DAero(Foil2DBasic):
    """
    Simplified aerofoil model suitable for propeller analysis.  The
    model is derived from the one used in XROTOR [1]_.  Main features
    are:

    - Lift has a linear region and specified non-linear limit and slope
      at each end.
    - A quadratic drag function is used which includes Mach divergence
      and stall effects.
    - The model is applicable between α = -90° and +90°.  Outisde this
      range, `NaN` is returned for all properties. See
      `XROTOR2DAeroPostStall` for a version including a more detailed
      post-stall allowing for very high and low angles of attack.

    Notes
    -----
    .. [1] XROTOR Download Page. See
       https://web.mit.edu/drela/Public/web/xrotor/
    """
    warn_α_range = True
    """If ``warn_α_range==True``, attempts to extrapolate for values 
    outside the `α` range will produce a warning. Note that properties 
    return `NaN` in this situation. """

    def __init__(self, α0: float, clα0: float, clα_stall: float,
                 cl_max: float, cl_min: float, cl_inc_stall: float,
                 cd_min: float, cl_cdmin: float, cd_cl2: float,
                 Re_ref: float, Re_exp: float, M_crit: float,
                 cm_qc: float, **kwargs):
        """
        Except where noted, all parameters apply to low subsonic
        (incompressible) conditions.

        Parameters
        ----------
        α0 : float
            Zero-lift angle (:math:`α_0`).
        clα0 : float
            Lift curve slope in the linear / low angle of attack regime
            (:math:`c_{lα0}`).
        clα_stall : float
            Lift curve slope at stall (:math:`c_{lα,STALL}`).
        cl_max : float
            Maximum `cl`.
        cl_min : float
            Minimum `cl`.
        cl_inc_stall : float
            `cl` increment up to stall.
        cd_min : float
            Minimum `cd`.
        cl_cdmin : float
            `cl` at minimum `cd`.
        cd_cl2 : float
            Parabolic drag parameter (:math:`dc_d/d_{cl}^2`).
        Re_ref : float
            Reference Reynolds number.
        Re_exp : float
            Reynolds number scaling exponent.
        M_crit : float
            Critical Mach number.
        cm_qc : float
            Pitching moment coefficient (1/4 chord, constant).
        """
        # Init base aerofoil.
        super().__init__(**kwargs)

        # Setup performance parameters.
        self._α0 = α0
        self._clα0 = clα0
        self._a_stall = clα_stall
        self._cl_max = cl_max
        self._cl_min = cl_min
        self._cl_inc_stall = cl_inc_stall
        self._cd_min = cd_min
        self._cl_cdmin = cl_cdmin
        self._cd_cl2 = cd_cl2
        self._Re_ref = Re_ref
        self._Re_exp = Re_exp
        self._M_crit = M_crit
        self._cm_qc_M0 = cm_qc  # For M = 0.

        # Setup private states and compute initial coefficients.
        self._cl, self._cd, self._cm_qc = None, None, None  # Init only.
        self._clα = None
        self._α_stall_max, self._α_stall_min = None, None  # Lazy eval.
        self._calc_coeffs()

    # -- Public Methods ------------------------------------------------

    @property
    def α0(self) -> float:
        return self._α0

    @property
    def α_stall_neg(self) -> float:
        """Refer to `XROTOR2DAero.α_stall_pos` for details."""
        if self._α_stall_min is None:
            if self._a_stall < 0.0:
                self._α_stall_min = super().α_stall_neg  # Peak search.
            else:
                self._α_stall_min = self._find_clα(
                    (0.05 * self._clα0 + 0.95 * self._a_stall) / self.β,
                    side=-1)

        return self._α_stall_min

    @property
    def α_stall_pos(self) -> float:
        """
        Returns
        -------
        float
            Positive stalling angle of attack (above :math:`α_0`).

        Notes
        -----
        - If the post-stall lift gradient (`clα`) is negative, the peak
          `cl` value is found to give `α`.
        - If the post-stall lift gradient is zero or positive, `α` is
          set as the value where `clα` reaches 95% of its post-stall
          value (including Prandtl-Glauert effects).
        """
        # If it has not yet been calculated, the stalling angle is
        # found by searching using the base class method.
        if self._α_stall_max is None:
            if self._a_stall < 0.0:
                self._α_stall_max = super().α_stall_pos  # Peak search.
            else:
                self._α_stall_max = self._find_clα(
                    (0.05 * self._clα0 + 0.95 * self._a_stall) / self.β,
                    side=+1)

        return self._α_stall_max

    @property
    def cd(self) -> float:
        return self._cd

    @property
    def cl(self) -> float:
        return self._cl

    @property
    def clα(self) -> float:
        return self._clα

    @property
    def clα0(self) -> float:
        """
        Lift curve slope (:math:`c_{lα} = dc_l/dα`) in the linear lift
        regime (low angles of attack) for incompressible flow (`M` = 0).
        """
        return self._clα0

    @property
    def cm_qc(self) -> float:
        return self._cm_qc

    def set_state(self, *, Re: float = None, M: float = None,
                  α: float = None) -> frozenset[str]:
        changed = super().set_state(α=α, Re=Re, M=M)

        # If anything changed, recompute the coefficients.
        if changed:
            self._calc_coeffs()

        # If either Re or M has changed, also flag the stall angles for
        # recalculation.
        if 'Re' in changed or 'M' in changed:
            self._α_stall_max = None
            self._α_stall_min = None

        return changed

    # @property
    # def valid_state(self) -> bool:
    #     """`XROTOR2DAero.valid_state` always returns `True`."""
    #     return True

    # -- Private Methods -----------------------------------------------

    def _calc_coeffs(self):
        """
        Compute the aerodynamic coefficients using the `XROTOR` code
        aerofoil model.
        """
        Re, M, α = self.Re, self.M, self.α  # One-time property access.

        # -- Check Valid Flow Conditions -------------------------------

        make_nan = False  # Return NaN for invalid conditions.

        # Check for very large (invalid) angles of attack.
        if not (-0.5 * np.pi <= α <= 0.5 * np.pi):
            if self.warn_α_range:
                warnings.warn(f"α = {np.rad2deg(α):+.02f} outside range "
                              f"[-90°, +90°].")
            make_nan = True

        if M >= 1.0:
            warnings.warn(f"Not valid for supersonic flow, got M = {M}.")
            make_nan = True

        if make_nan:
            self._cl, self._cd, self._cm_qc = np.nan, np.nan, np.nan
            self._clα = np.nan
            return

        # -- Compressible Flow Corrections -----------------------------

        # Constants.
        cl_M_factor = 0.25
        cd_M_factor = 10.0
        M_exp = 3.0
        cd_M_dd = 0.0020
        cd_M_stall = 0.1000  # Drag at start of compressible stall.
        PG = 1.0 / self.β  # Prandtl-Glauert compressibility factor.

        # -- Lift ------------------------------------------------------

        # Generate cl from d(cl)/d(alpha) and Prandtl-Glauert scaling.
        clα_eff = self._clα0 * PG
        cl_eff = clα_eff * (α - self._α0)

        # Effective cl_max is limited by Mach effects. We reduce cl_max
        # to match the cl of the onset of serious compressible drag.
        ΔM_stall = (cd_M_stall / cd_M_factor) ** (1.0 / M_exp)
        cl_max_M = (max(0.0, (self._M_crit + ΔM_stall - M) / cl_M_factor)
                    + self._cl_cdmin)
        cl_max_eff = min(self._cl_max, cl_max_M)
        cl_min_M = (min(0.0, -(self._M_crit + ΔM_stall - M) / cl_M_factor)
                    + self._cl_cdmin)
        cl_min_eff = max(self._cl_min, cl_min_M)

        # Apply a cl limiter function that turns on after +ve / -ve
        # stall.
        ec_max = np.exp(min(200.0, (cl_eff - cl_max_eff) /
                            self._cl_inc_stall))
        ec_min = np.exp(min(200.0, (cl_min_eff - cl_eff) /
                            self._cl_inc_stall))
        cl_lim = self._cl_inc_stall * np.log((1.0 + ec_max) /
                                             (1.0 + ec_min))
        dcl_lim_da = ec_max / (1.0 + ec_max) + ec_min / (1.0 + ec_min)

        # Subtract off a (nearly unity) fraction of the limited cl
        # function.  This sets the d(cL)/d(alpha) in the stalled regions
        # to 1 - f_stall of that in the linear lift range.  Note: This
        # is the standard XROTOR stall model.
        f_stall = self._a_stall / self._clα0
        self._cl = cl_eff - (1.0 - f_stall) * cl_lim
        self._clα = clα_eff - (1.0 - f_stall) * dcl_lim_da * clα_eff

        # -- Drag ------------------------------------------------------

        # Reynolds number scaling factor.
        if np.isinf(Re):
            Re_factor = 1.0
        else:
            Re_factor = (Re / self._Re_ref) ** self._Re_exp

        # cd is the sum of profile drag, stall drag and compressibility
        # drag.  In the basic linear lift range profile drag is a
        # function of lift i.e. cd = cd0 (constant) + (quadratic with
        # CL)
        cd_profile = (self._cd_min + self._cd_cl2 * (
                self._cl - self._cl_cdmin) ** 2) * Re_factor

        # Post-stall drag increment.
        Δcd_stall = 2.0 * ((1.0 - f_stall) * cl_lim /
                           (PG * self._clα0)) ** 2

        # Compressibile drag increment, accounting for drag rise
        # above M_crit:
        #   - ΔM_dd is the ΔM giving a cd rise of cd_M_dd at M_crit.
        #   - M_crit_eff = M_crit - cl_M_factor * (cl - cl_cdmin) - ΔM_dd
        #   - Δcd_c = cd_M_factor * (M - Mcrit_eff) ** M_exp
        ΔM_dd = (cd_M_dd / cd_M_factor) ** (1.0 / M_exp)
        M_crit_eff = (self._M_crit - cl_M_factor *
                      np.abs(self._cl - self._cl_cdmin) - ΔM_dd)
        if M < M_crit_eff:
            Δcd_c = 0.0
        else:
            Δcd_c = cd_M_factor * (M - M_crit_eff) ** M_exp

        self._cd = cd_profile + Δcd_stall + Δcd_c

        # -- Pitching Moment -------------------------------------------

        self._cm_qc = PG * self._cm_qc_M0

        # -- Post-Stall Limits -----------------------------------------

        # This is beyond the original XROTOR formulation and just
        # removes some numerical artifacts (e.g. rising post-stall cl,
        # etc).  It is not a post-stall model per se.

        if cl_eff > cl_max_eff:
            stall = +1  # Positive stall region.
        elif cl_eff < cl_min_eff:
            stall = -1  # Negative stall region.
        else:
            stall = 0

        # If post-stall, limit the cl from falling back through zero.
        # Set slope accordingly.
        if stall and np.sign(self._cl) != stall:
            self._cl = 0.0
            self._clα = 0.0

    def _find_clα(self, clα: float, side: int) -> float:
        """
        Finds the `α` that produces the given `clα` value.  The
        starting point for the search is set by `side`, where
        ``side = 1`` means above `α0` and ``side = -1`` means below
        `α0`.
        """
        # Define error function as f(clα) = clα - clα* = 0.
        def clα_err(α_: float) -> float:
            self.set_state(α=α_)
            return self.clα - clα

        with self.restore_state():
            # First find an initial bracket by stepping along the curve.
            try:
                x1, x2 = step_bracket_root(
                    clα_err, x1=self.α0,
                    x2=self.α0 + side * 0.0349,  # Step 2° -> [rad].
                    x_limit=np.pi * np.sign(side))  # -180° or 180°.

            except SolverError as e:
                raise SolverError(f"Failed to bracket clα = {clα:.03f} - "
                                  f"{e.details}") from e

            # Solve for the value within the bracket.
            sol = root_scalar(clα_err, bracket=(x1, x2), xtol=1e-3,
                              maxiter=20)

            if not sol.converged:
                raise SolverError(f"Failed to find α giving clα = "
                                  f"{clα:.03f}: {sol.message}")

        return sol.root


# ======================================================================


class XROTOR2DAeroPostStall(PostStall2DMixin, XROTOR2DAero):
    """
    `XROTOR2DAeroPostStall` extends the `XROTOR2DAero` class by adding
    the `PostStall2DMixin` mixin to give estimated properties at large
    positive and negative (post-stall) angles of attack.

    Default post-stall model behaviour is to replace base model `NaN`
    values, and other values if post-stall lift is higher (unless
    overridden).
    """
    def __init__(self, **kwargs):
        super().__init__(**({'replace_nan': True, 'high_α': True} | kwargs))
