from __future__ import annotations
from abc import ABC
from types import SimpleNamespace

import numpy as np
from scipy.optimize import root_scalar

from .base import Foil2DAero
from .flat import Flat2DAero
from pyavia.numeric.solve import step_bracket_root, SolverError


# Written by Eric J. Whitney, January 2023.

# ===========================================================================


class PostStall2DMixin(Foil2DAero, ABC):
    r"""
    Mixin class that adds a post-stall angle of attack performance
    model to another `Foil2DAero` type.  This allows properties to be
    evaluated in the complete range of :math:`α \in [-π, +π]` which is
    required in some cases such high-α models or sections of propellers
    / rotors operating at high pitch.

    Notes
    -----
    - At present the post-stall model used is `Flat2DAero` which gives a
      useful approximation of post-stall properties.
    - Post-stall values are substituted whenever `α` is not in the range
      :math:`[α^{-}_{stall}, α^{+}_{stall}]`, depending on
      `PostStall2DMixin.__init__` parameters.
    - For properties covered by the post-stall model, `α`-range warnings
      are disabled by default.
    """
    warn_α_range = False  # Override default aerofoil behaviour.

    def __init__(self, *, replace_nan: bool = True, high_α: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        replace_nan : bool, default = True
            If `True` and the base foil is stalled, any `NaN` value
            returned by the base model will be replaced by the
            corresponding post-stall model value.

        high_α : bool, default = False
            If `True` and the base foil is stalled, a check is made to
            determine if `α` is beyond the point where the base model
            :math:`c_l` crosses over (falls below) the post-stall model.
            If so, all properties are taken from the post-stall model.

            .. note::  Only the first crossover point is computed.
               Beyond this angle of attack the post-stall model values
               apply until `α` → ±`π`.
        """
        super().__init__(**kwargs)

        # Setup post-stall parameters in separate namespace.  Note that
        # 'active' is not an external attribute; the model is not
        # designed to be switched on and off.  This is only used for
        # temporarily disabling model when deriving certain parameters.
        self._post_stall = SimpleNamespace(
            model=Flat2DAero(Re=self.Re, M=self.M, α=self.α),
            active=True,
            replace_nan=replace_nan,
            high_α=high_α,
            base_α_stall=[None, None],  # -ve, +ve.
            post_α_cross=[None, None]  # -ve, +ve.
        )

    # -- Public Methods ------------------------------------------------

    @property
    def α_stall_neg(self) -> float:
        # Similar to α_stall_pos.
        ps = self._post_stall  # For compactness.
        if ps.base_α_stall[0] is None:
            restore = ps.active
            ps.active = False
            ps.base_α_stall[0] = super().α_stall_neg
            ps.active = restore

        return ps.base_α_stall[0]

    @property
    def α_stall_pos(self) -> float:
        # The stall angle is determined by the superclass method, after
        # the post-stall model has been temporarily disabled.
        ps = self._post_stall  # For compactness.
        if ps.base_α_stall[1] is None:
            restore = ps.active
            ps.active = False
            ps.base_α_stall[1] = super().α_stall_pos
            ps.active = restore

        return ps.base_α_stall[1]

    @property
    def cd(self) -> float:
        return self._stalled_prop('cd')

    @property
    def cl(self) -> float:
        return self._stalled_prop('cl')

    @property
    def clα(self) -> float:
        return self._stalled_prop('clα')

    @property
    def cm_qc(self) -> float:
        return self._stalled_prop('cm_qc')

    def set_state(self, **kwargs) -> frozenset[str]:
        # Update base model.
        changed = super().set_state(**kwargs)

        # Always update post-stall model even when inactive, to prevent
        # missing state changes, and always uses base model output
        # values.  The angle of attack is adjusted for the post-stall
        # model.
        ps = self._post_stall  # For compactness.
        ps.model.set_state(Re=self.Re, M=self.M, α=self.α - self.α0)

        if changed - {'α'}:
            # If anything other than 'α' changed, stall / crossover
            # points may need to be recomputed.
            ps.base_α_stall = [None, None]
            ps.post_α_cross = [None, None]

        return changed

    # -- Private Methods -----------------------------------------------

    def _find_α_cross(self, side: int) -> float:
        """
        Returns the first `α` value where the lift coefficient of the
        base and post-stall aerofoil models cross over (positive for
        side = +1, negative for side = -1).

        The crossover point is limited to +/-60° + ZLA.
        """
        # Find root of Δcl = cl_base - cl_post = f(α) -> 0.
        def Δcl_gap(α: float) -> float:
            self.set_state(α=α)
            cl_base = super(PostStall2DMixin, self).cl
            cl_post = self._post_stall.model.cl
            return cl_base - cl_post

        with self.restore_state():
            # First approximately bracket the crossover cl. Setup starting
            # bracket (α_stall,  α_stall +/- 2°).
            if side >= 0:
                α1 = self.α_stall_pos
                α_limit = 1.0471976 + self.α0  # 60° + ZLA.
            else:
                α1 = self.α_stall_neg
                α_limit = -1.0471976 + self.α0  # -60° + ZLA.

            α2 = α1 + side * 0.0349
            max_steps = 50
            try:
                x1, x2 = step_bracket_root(Δcl_gap, x1=α1, x2=α2,
                                           x_limit=α_limit,
                                           max_steps=max_steps)
            except SolverError as e:
                if e.flag == 2:
                    # 'α' limit reached; just return the limit.
                    return α_limit

                else:
                    raise SolverError(f"Failed to bracket the crossover cl - "
                                      f"{e.details}") from e

            # Converge the final crossover value.
            sol = root_scalar(Δcl_gap, bracket=(x1, x2), xtol=1e-3,
                              maxiter=20)
            if not sol.converged:
                raise SolverError(f"Failed to converge the crossover "
                                  f"cl: {sol.flag}")

        return sol.root

    def _stalled_prop(self, name: str) -> float:
        """
        Checks if the stall model is active and returns whichever
        property attribute is correct.
        """
        base_val = getattr(super(), name)

        # If inactive, return immediately.
        ps = self._post_stall  # For compactness.
        if not ps.active:
            return base_val

        # If α is post-stall and specified conditions are met,
        # substitute post-stall value.
        if self.α < self.α_stall_neg:
            stall = -1
        elif self.α > self.α_stall_pos:
            stall = +1
        else:
            return base_val  # Not stalled, return base value.

        # Find post-stall model value.
        post_val = getattr(ps.model, name)

        # Replace NaN values.
        if ps.replace_nan and np.isnan(base_val):
            return post_val

        # Replace if beyond cl crossover point.
        if ps.high_α:
            if stall < 0:
                # Negative side.
                if ps.post_α_cross[0] is None:
                    ps.post_α_cross[0] = self._find_α_cross(-1)

                if self.α < ps.post_α_cross[0]:
                    return post_val

            elif stall > 0:
                # Positive side.
                if ps.post_α_cross[1] is None:
                    ps.post_α_cross[1] = self._find_α_cross(+1)

                if self.α > ps.post_α_cross[1]:
                    return post_val

        # All other cases, return base (unstalled) value.
        return base_val

# ======================================================================
