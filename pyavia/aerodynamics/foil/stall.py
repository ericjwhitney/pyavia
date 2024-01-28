from __future__ import annotations
from abc import ABC

import numpy as np
from scipy.optimize import root_scalar

from .base import Foil2DAero
from .flat import Flat2DAero
from pyavia.solve import SolverError, step_bracket_root


# Written by Eric J. Whitney.

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

    def __init__(self, *, post_nan: bool = True,
                 # TODO Consider better names for these.
                 post_crossover: bool = False, **kwargs):
        """
        Parameters
        ----------
        post_nan : bool, default = True
            If `True` and the 'base' (underlying) foil is stalled, any
            `NaN` value returned by the base model will be replaced by
            the corresponding post-stall model value.
        post_crossover : bool, default = False
            If `True` and the base foil is stalled, a check is made to
            determine if `α` is beyond the point where the base model
            :math:`c_l` crosses over (falls below) the post-stall model.
            If so, all properties are taken from the post-stall model.

            .. note::  Only the first crossover point is computed.
               Beyond this angle of attack the post-stall model values
               apply until `α` → ±`π`.
        """
        super().__init__(**kwargs)

        # Setup post-stall parameters.  Note that _post_active is not an
        # external state; the model is not designed to be switched on and
        # off.  This is only used for temporarily disabling model when
        # deriving certain parameters.
        self._post_model = Flat2DAero(Re=self.Re, M=self.M, α=self.α)
        self._post_active = True
        self._post_nan, self._post_crossover = post_nan, post_crossover

        # Setup stalling and crossover 'α' of base model.
        self._base_α_stall_neg, self._base_α_stall_pos = None, None
        self._post_α_cross_neg, self._post_α_cross_pos = None, None

    # -- Public Methods ------------------------------------------------

    @property
    def α_stall_neg(self) -> float:
        # Similar to α_stall_pos.
        if self._base_α_stall_neg is None:
            restore, self._post_active = self._post_active, False
            self._base_α_stall_neg = super().α_stall_neg
            self._post_active = restore

        return self._base_α_stall_neg

    @property
    def α_stall_pos(self) -> float:
        # The stall angle is determined by the superclass method, after
        # the post-stall model has been temporarily disabled.
        if self._base_α_stall_pos is None:
            restore, self._post_active = self._post_active, False
            self._base_α_stall_pos = super().α_stall_pos
            self._post_active = restore

        return self._base_α_stall_pos

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

    def set_states(self, **kwargs) -> frozenset[str]:
        # Update base model.
        changed = super().set_states(**kwargs)

        # Always update post-stall model even when inactive, to prevent
        # missing state changes, and always uses base model output
        # values.  The angle of attack is adjusted for the post-stall
        # model.
        self._post_model.set_states(Re=self.Re, M=self.M,
                                    α=self.α - self.α0)

        if changed - {'α'}:
            # If anything other than 'α' changed, stall / crossover
            # points may need to be recomputed.
            self._base_α_stall_neg, self._base_α_stall_pos = None, None
            self._post_α_cross_neg, self._post_α_cross_pos = None, None

        return changed

    # -- Private Methods -----------------------------------------------

    def _find_α_cross(self, side: int) -> float:
        """
        Returns the first `α` value where the lift coefficient of the
        base and post-stall aerofoil models cross over (positive for
        side = +1, negative for side = -1).

        The crossover point is limited to +/-60° + ZLA.
        """
        restore = self.get_states()

        # Find root of Δcl = cl_foil - cl_stall = f(α) -> 0.
        def Δcl_models(α_: float) -> float:
            self.set_states(α=α_)
            return super(PostStall2DMixin, self).cl - self._post_model.cl

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
            x1, x2 = step_bracket_root(Δcl_models, x1=α1, x2=α2,
                                       x_limit=α_limit,
                                       max_steps=max_steps)
        except SolverError as e:
            if e.flag == 2:
                # 'α' limit reached; just return the limit.
                self.set_states(**restore)
                return α_limit

            else:
                raise SolverError(f"Failed to bracket the crossover cl - "
                                  f"{e.details}") from e

        # Converge the final crossover value.
        sol = root_scalar(Δcl_models, bracket=(x1, x2), xtol=1e-3, maxiter=20)
        if not sol.converged:
            raise SolverError(f"Failed to converge the crossover cl: "
                              f"{sol.flag}")

        self.set_states(**restore)
        return sol.root

    def _stalled_prop(self, prop: str) -> float:
        """
        Checks if the stall model is active and returns whichever
        property attribute is correct.
        """
        base_val = getattr(super(), prop)

        # If inactive, return immediately.
        if not self._post_active:
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
        post_val = getattr(self._post_model, prop)

        # Replace NaN values.
        if self._post_nan and np.isnan(base_val):
            return post_val

        # Replace if beyond cl crossover point.
        if self._post_crossover:
            if stall < 0:
                # Negative side.
                if self._post_α_cross_neg is None:
                    self._post_α_cross_neg = self._find_α_cross(-1)

                if self.α < self._post_α_cross_neg:
                    return post_val

            elif stall > 0:
                # Positive side.
                if self._post_α_cross_pos is None:
                    self._post_α_cross_pos = self._find_α_cross(+1)

                if self.α > self._post_α_cross_pos:
                    return post_val

        # All other cases, return base (unstalled) value.
        return base_val

# ======================================================================
