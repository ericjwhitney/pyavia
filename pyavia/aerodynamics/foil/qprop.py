import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .base import Foil2DBasic
from pyavia.solve.exception import SolverError

# Written by Eric J. Whitney, January 2023.

d2r = np.deg2rad


# ======================================================================

class QPROP2DAero(Foil2DBasic):
    """
    Simplified aerofoil model suitable for propeller analysis.  The model
    is derived from the one used in QPROP [1]_.  Main features are:

    -  Lift consists of a linear region which is capped by a limit
       value at each end.

        - Lift coefficient is scaled by the Prandtl-Meyer
          compressibility factor in the linear region.
        - Stall uses a -cos(`α`) heuristic to reduce from the maximum.

    -  A two piece quadratic drag function is used for normal angles of
       attack.

        - Drag is increased beyond stall and drag-divergence Mach
          number using heuristics.
        - Reynolds number effects are incorporated by exponential
          scaling vs. a reference value.

    Notes
    -----
    .. [1] M. Drela, "QPROP Formulation", MIT Aero & Astro, June 2006.
       See https://web.mit.edu/drela/Public/web/qprop/
    """

    def __init__(self, cl0: float, clα0: float, cl_max: float,
                 cl_min: float, cd_min: float, cl_cd_min: float,
                 cd_cl2_pos: float, cd_cl2_neg: float,
                 Re_ref: float, Re_exp: float, M_crit: float = 0.7,
                 **kwargs):
        """
        Except where noted, all parameters apply to low subsonic /
        incompressible conditions (`M` ≈ 0).  Reynolds and Mach number
        scaling effects are incorporated in all properties.

        Parameters
        ----------
        cl0 : float
            `cl` when `α` = 0 (:math:`c_{l,α=0}`).
        clα0 : float
            Lift curve slope (:math:`c_{lα0}`) at low (linear) angles of
            attack.
        cl_max : float
            Maximum `cl`.
        cl_min : float
            Minimum `cl`.
        cd_min : float
            Minimum `cd`.
        cl_cd_min : float
            `cl` at minimum `cd`.
        cd_cl2_pos : float
            Parabolic drag parameter (:math:`dc_d^{+}/d_{cl}^2`) above
            `cl_cd_min`.
        cd_cl2_neg : float
            Parabolic drag parameter (:math:`dc_d^{-}/d_{cl}^2`) below
            `cl_cd_min`.
        Re_ref : float
            Reference Reynolds number.
        Re_exp : float
            Reynolds number scaling exponent.
        M_crit : float, default=0.7
            Critical Mach number.  Default is ``M_crit=0.7`` as this is
            the hardcoded value used in `QPROP`.
        """
        super().__init__(**kwargs)

        # Setup model parameters (M ≈ 0).
        self._α0 = -cl0 / clα0  # Use more familiar form internally.
        self._Δα_cd_min = (cl_cd_min - cl0) / clα0  # For drag model.
        self._clα0 = clα0
        self._cl_max, self._cl_min = cl_max, cl_min
        self._cd_min = cd_min
        self._cl_cd_min = cl_cd_min
        self._cd_cl2_pos, self._cd_cl2_neg = cd_cl2_pos, cd_cl2_neg
        self._Re_ref, self._Re_exp = Re_ref, Re_exp
        self._M_crit = M_crit

    # -- Public Methods ------------------------------------------------

    @property
    def α0(self) -> float:
        return self._α0

    @property
    def α_stall_pos(self) -> float:
        # Note: Using clα0 incorporates compressibility correction.
        return (self._cl_max / self.clα0) + self._α0

    @property
    def α_stall_neg(self) -> float:
        # Note: Using clα0 incorporates compressibility correction.
        return (self._cl_min / self.clα0) + self._α0

    @property
    def cd(self) -> float:
        # Subsonic drag equation.
        Δcl = self.cl - self._cl_cd_min
        if Δcl >= 0.0:
            cd_cl2 = self._cd_cl2_pos
        else:
            cd_cl2 = self._cd_cl2_neg

        Re_factor = (self.Re / self._Re_ref) ** self._Re_exp
        cd = (self._cd_min + cd_cl2 * Δcl ** 2) * Re_factor

        # Add MDD effect.
        ΔMdd = self.M - self._M_crit
        k_Mdd, exp_Mdd = 10.0, 3  # Constants.
        if ΔMdd > 0.0:
            cd += k_Mdd * ΔMdd ** exp_Mdd

        # Add stall effect.
        if not (self.α_stall_neg <= self.α <= self.α_stall_pos):
            cd += 2.0 * np.sin(self.α - self._Δα_cd_min) ** 2

        return cd

    @property
    def cl(self) -> float:
        # Note 1: We use a (+) sign in 'self.α + self._α0' to match the
        # behaviour of original which didn't include the usual (-) sign
        # when computing α0.  Original code snippet:
        #   ACL0 = CL0 / DCLDA          <- No (-) sign.
        #   CL = CLMAX * COS(A - ACL0)  <- Reversed (-) to (+) below.
        #
        # Note 2: The stall model ignores Mach number.

        if self.α > self.α_stall_pos:
            # Positive stall.
            return self._cl_max * np.cos(self.α + self._α0)  # <- (+)

        elif self.α < self.α_stall_neg:
            # Negative stall.
            return self._cl_min * np.cos(self.α + self._α0)  # <- (+)

        else:
            # Linear lift region.
            return self.clα0 * (self.α - self._α0)

    @property
    def cl0(self) -> float:
        """Lift coefficient :math:`c_l` when `α` = 0."""
        return -self.α0 * self.clα0

    @property
    def clα(self) -> float:
        # See 'cl' property regarding the reason for the (+) sign in the
        # stalled computation here.
        if self.α > self.α_stall_pos:
            # Positive stall.
            return -self._cl_max * np.sin(self.α + self._α0)  # <- (+)

        elif self.α < self.α_stall_neg:
            # Negative stall.
            return -self._cl_min * np.sin(self.α + self._α0)  # <- (+)

        else:
            # Linear lift region.
            return self.clα0

    @property
    def clα0(self) -> float:
        """
        Lift curve slope (:math:`c_{lα} = dc_l/dα`) in the linear lift
        regime (low lift region around α0).
        """
        return self._clα0 / self.β  # Comprresiblity correction.

    @property
    def cm_qc(self) -> float:
        """QPROP2DAero.cm_qc always returns `np.nan`."""
        return np.nan

    # @property
    # def valid_state(self) -> bool:
    #     """`QPROP2DAero.valid_state` always returns `True`."""
    #     return True


# ======================================================================

# TODO In development
def QPROP_from_model(model_foil: Foil2DBasic, test_α: ArrayLike,
                     Re_ref: float = 70000.0, Re_exp: float = -0.7,
                     M_crit: float = 0.7) -> QPROP2DAero:
    """
    Build a QPROP2DAero object as a best representation of foil
    characteristics given by `model_foil`.

    Parameters
    ----------
    model_foil
    test_α
    Re_ref
    Re_exp
    M_crit

    Returns
    -------

    """
    # -- Run Model Aerofoil --------------------------------------------

    restore = model_foil.def_state_values()
    test_Re, test_M = restore['Re'], restore['M']

    if (n_pts := len(test_α)) < 5:
        raise ValueError("Require at least 5 test α values.")

    # Get model fixed parameters.
    model_α0 = model_foil.α0
    model_α_stall_neg = model_foil.α_stall_neg
    model_α_stall_pos = model_foil.α_stall_pos
    model_cl_stall_neg = model_foil.cl_stall_neg
    model_cl_stall_pos = model_foil.cl_stall_pos

    # Run α-sweep of model.
    model_cl, model_cd = [], []
    for i in range(n_pts):
        model_foil.set_states(α=test_α[i])
        model_cl.append(model_foil.cl)
        model_cd.append(model_foil.cd)

    model_foil.set_states(**restore)

    # -- Setup Model Variables ----------------------------------------------

    # Establish default values and bounds for model parameters.
    lift_defs = {
        # (low bound, default, high bound)
        'cl0': [-1.0, 0.0, +1.0],  # Core parameters.  TODO WAS 0.5
        'clα0': [3.0, 5.8, 9.0],
        'cl_min': [-2.0, -1.5, 0.0],  # TODO WAS -0.4
        'cl_max': [0.1, 1.5, 2.0]}  # TODO WAS 1.2

    drag_defs = {
        # (low bound, default, high bound)
        'cd_min': [0.000, 0.05, None],  # TODO WAS 0.028
        'cd_cl2_pos': [0.000, 0.10, None],  # TODO WAS 0.05
        'cd_cl2_neg': [0.000, 0.10, None],  # TODO WAS 0.05
        'cl_cd_min': [-0.8, 0.1, 0.8],  # TODO WAS 0.5
        'Re_ref': [1e3, Re_ref, 1e9],  # Optional parameters.
        'Re_exp': [-2.0, Re_exp, 0.0],
        'M_crit': [0.40, M_crit, 0.99]}

    # TODO We can add robustness thru all of this later, check for NaN or
    #  fail, etc.

    lift_vars = set(lift_defs.keys())
    drag_vars = set(drag_defs.keys()) - {'Re_ref', 'Re_exp', 'M_crit'}

    # Get parameters that can be determined heuristically.
    lift_defs['cl_max'][1] = model_cl_stall_pos
    lift_vars.discard('cl_max')

    lift_defs['cl_min'][1] = model_cl_stall_neg
    lift_vars.discard('cl_min')

    # Assemble default values.
    all_defs = lift_defs | drag_defs

    # -- Common Function Definitions  ---------------------------------------

    def make_x0(use_vars_: {str}) -> NDArray:
        """
        Return 'x' starting values for any 'use_vars' appearing in 'all_defs',
        preserving its order.
        """
        return np.array([v_[1] for k_, v_ in all_defs.items()
                         if k_ in use_vars_])

    def make_x_bounds(use_vars_: {str}) -> [(float, float)]:
        """
        Return (min, max) boundary values values for any 'use_vars'
        appearing in 'all_defs', preserving its order.
        """
        return [(v_[0], v_[2]) for k_, v_ in all_defs.items()
                if k_ in use_vars_]

    def build_kwargs(x_: NDArray, use_vars_: {str}) -> {str: float}:
        """
        Build **kwargs for a bespoke QPROP2DAero object, given 'x_'
        values. Note: 'x_' should be ordered to match 'all_defs'.
        """
        # Pass over all keywords, taking a values either from 'x_' (if in
        # 'use_vars_') or 'all_defs'.
        kwargs_, i_ = {}, 0
        for k_, v_ in all_defs.items():
            if k_ in use_vars_:
                kwargs_[k_] = x_[i_]  # Set from 'x_'.
                i_ += 1
            else:
                kwargs_[k_] = v_[1]  # Default value.

        # Check we consumed entire 'x_'.
        if i_ != len(x_):
            raise ValueError("Mismatch in number of variables.")

        return kwargs_

    # -- Fit Lift Curve ------------------------------------------------

    def cl_fitfunc(x_: NDArray, use_vars_: {str}) -> float:
        """
        Compare QPROP trial foil to the model lift.
        TODO At this stage this only uses unstalled points +/- 7° either side
         of α0.
        """
        try_foil_ = QPROP2DAero(**build_kwargs(x_, use_vars_))
        cl_err2_ = []
        for i_ in range(n_pts):
            α_test_ = test_α[i_]

            # TODO unstalled region only.
            if ((α_test_ < model_α_stall_neg) or
                    (α_test_ > model_α_stall_pos)):
                continue

            # TODO linear region only.
            if (α_test_ < (model_α0 - d2r(5.0)) or
                    α_test_ > (model_α0 + d2r(5.0))):
                continue

            try_foil_.set_states(α=α_test_, Re=test_Re, M=test_M)
            cl_err2_.append((try_foil_.cl - model_cl[i_])**2)

        if len(cl_err2_) < 5:
            raise SolverError("Too few linear cl points.")

        return np.sqrt(np.sum(cl_err2_) / len(cl_err2_))

    lift_res = minimize(lambda x: cl_fitfunc(x, lift_vars),
                        x0=make_x0(lift_vars),
                        bounds=make_x_bounds(lift_vars),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-4,
                                 'fatol': 1e-5})

    # res = minimize(fitfunc, np.asarray(x0), bounds=xb,
    #                method='Powell', options={
    #         'disp': True, 'maxiter': 5000, 'xtol': 1e-5,
    #         'ftol': 1e-6})

    if not lift_res.success:
        raise SolverError(f"Failed to fit lift curve after {lift_res.nit} "
                          f"iterations: {lift_res.message}")

    if lift_res.fun > 0.2:  # TODO Working value.
        raise SolverError(f"Failed to fit lift curve to data with "
                          f"sufficient accuracy.")

    # Update default parameters with the solution values.
    for k, v in build_kwargs(lift_res.x, lift_vars).items():
        all_defs[k][1] = v

    # -- Fit Drag Curve -----------------------------------------------------

    def cd_fitfunc(x_: NDArray, use_vars_: {str}) -> float:
        """
        Compare QPROP trial foil to the model drag.
        """
        try_foil_ = QPROP2DAero(**build_kwargs(x_, use_vars_))
        cd_err2_ = []
        for i_ in range(n_pts):
            α_test_ = test_α[i_]

            # TODO unstalled region only.
            if ((α_test_ < model_α_stall_neg) or
                    (α_test_ > model_α_stall_pos)):
                continue

            # # TODO linear region only.
            # if ((α_test_ < model_α0 - d2r(7.0)) or
            #         (α_test_ > model_α0 + d2r(7.0))):
            #     continue

            try_foil_.set_states(α=α_test_, Re=test_Re, M=test_M)
            cd_err2_.append((try_foil_.cd - model_cd[i_]) ** 2)

        if len(cd_err2_) < 5:
            raise SolverError("Too few cd values to compute error.")

        return np.sqrt(np.sum(cd_err2_) / len(cd_err2_))

    drag_res = minimize(lambda x: cd_fitfunc(x, drag_vars),
                        x0=make_x0(drag_vars),
                        bounds=make_x_bounds(drag_vars),
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6,
                                 'fatol': 1e-6})

    if not drag_res.success:
        raise SolverError(f"Failed to fit drag curve after {drag_res.nit} "
                          f"iterations: {drag_res.message}")

    if drag_res.fun > 0.1:  # TODO ESTABLISH
        raise SolverError(f"Failed to fit drag curve to data with "
                          f"sufficient accuracy.")

    # Update default parameters with the solution values.
    for k, v in build_kwargs(drag_res.x, drag_vars).items():
        all_defs[k][1] = v

    # Build final object.
    qprop = QPROP2DAero(**build_kwargs(np.array([]), set()))

    # TODO DEBUGGING.

    return qprop
