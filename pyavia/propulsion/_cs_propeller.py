from __future__ import annotations

from abc import ABC
from functools import partial

import numpy as np

from pyavia.util.print_styles import (
     PrintStylesMixin, LevelTabStyle, LevelDotStyle, rad2str, val2str)
from pyavia.numeric.solve import SolverError
from pyavia.propulsion import Propeller

from pyavia.state import InvalidStateError


# Written by Eric J. Whitney, February 2024.


# ======================================================================


class CSPropeller(PrintStylesMixin, Propeller, ABC):
    """
    TODO extending `Propeller` to allow for constant-speed
    operation (torque matching) to be used to set the blade angle (β0).

    CSPropeller must appears as the first (LH) parent in any
    concrete subclass to operate correctly.

    Initial β0 can be set during construction via kwarg TODO.

    A definite `β0_range` must be established otherwise `__init__` will
    raise `ValueError`.

    Parameters
    ----------
    β0_range : (float, float)
        Range of `β0` values `(min, max)` `[radians]` allowed.

        .. note::If `β0` is not provided `__init__` raises a
           `ValueError` exception.

    Ps_target : float, optional
        Initial value TODO
    Qs_target : float, optional
        Initial value TODO
        
    disp : xxxx

    maxits_cs : int, default = 20
        Maximum number of iterations for convergence of propeller blade
        angle in constant speed analyses.  Most normal points should
        only require 4-5 iterations.

    tol_cs : float, default = 1e-3
        Convergence tolerance applied to target variable (e.g. power,
        torque, thrust) in constant speed analyses
        TODO UPDATE TO REFLECT TORQUE ONLY 'Qs_tol'

        - If the target value is non-zero this represents a relative
          error - the error divided by the target value.  For example
          if the target power is 75 hp, then ``cs_tol = 0.01`` means
          the blade angle is converged if shaft power is within 1%,
          or within the range 74.25 hp <-> 75.75 hp.
        - If the target value is zero this is applied as an absolute
          tolerance.
    """
    def __init__(self, *,
                 # Initial operating state.
                 Ps: float = None, Qs: float = None,
                 # Options.
                 display_level: int = 0,
                 maxits_cs: int = 20, tol_cs: float = 1e-3,
                 **kwargs):
        super().__init__(display_level=display_level, **kwargs)
        if self.β0_range is None:
            raise ValueError("'β0_range' required.")

        # Setup formatting for _run_cs() printed output (two levels). If
        # other print styles are defined, insert these above them in the
        # hierarchy as the root style.
        top_style_names = list(self.pstyles.get_level(1).keys())
        self.pstyles.add('CSPropeller_2', LevelTabStyle(),
                         children=top_style_names)
        self.pstyles.add('CSPropeller_1', LevelDotStyle(),
                         children='CSPropeller_2')

        # Setup convergence parameters.
        self.maxits_cs = maxits_cs
        self.tol_cs = tol_cs
        self._Qs_target = np.nan  # NaN for type and error checking.

        # Set and check state inputs at this level.
        self._cs_propeller_set_state(Ps=Ps, Qs=Qs)

    # -- Public Methods ------------------------------------------------

    def set_state(self, *, Ps: float = None, Qs: float = None,
                  **kwargs) -> frozenset[str]:
        """
        Set constant speed propeller operating state using groups of
        related input values - for more information see `Notes`. Only a
        single value from each state group (primary or optional) may be
        selected as shown below:

        +-------------------------------+---------------------------+
        |           State Group         |                           |
        +---------------+---------------+          Affects          |
        |    Primary    |    Optional   |                           |
        +===============+===============+===========================+
        |      `V0`     |      `J`      | Freestream Airspeed       |
        +---------------+---------------+---------------------------+
        |      `Ω`      |   `n`, `RPM`  | Rotational Speed          |
        +---------------+---------------+---------------------------+
        |    *`Qs`*     |    *`Ps`*     | Collective Incidence      |
        +---------------+---------------+---------------------------+
        |     `ρ0`      |               | Freetream Density         |
        +---------------+---------------+---------------------------+
        |     `a0`      |               | Freestream Speed of Sound |
        +---------------+---------------+---------------------------+
        |     `μ0`      |               | Freestream Viscosity      |
        +---------------+---------------+---------------------------+

        Returns
        -------
        frozenset[str]
            Set of primary input attribute names that were changed.

        Raises
        ------
        InvalidStateError
            If inputs have incorrect values.

        Notes
        -----
        - This is similar to the fixed-pitch `Propeller.set_state(...)`,
          except that the option to directly set the collective
          incidence (`β0`) is removed and replaced with the requirement
          to specify a shaft torque (or equivalent shaft power).
        - If
        """
        changed = super().set_state(**kwargs)
        changed |= self._cs_propeller_set_state(Ps=Ps, Qs=Qs)

        # If nothing has changed, return immediately.
        if not changed:
            return changed

        self._run_cs()
        return changed | {'β0'}

    # -- Private Methods -----------------------------------------------

    # Future Work: This whole function needs a spruce-up. It only really
    #              converges once you get two good analysis points in a
    #              row.  Something a bit more robust that remembers all
    #              the previous attempts would be good. Could be
    #              generic... 'robust_newton' or similar.  Should track
    #              'good_x', 'good_y', 'bad_x', and so on.
    
    def _run_cs(self):
        """
        Adjust β0 to converge make `Qs` = `Qs_target`.
        """
        ps_print_1 = partial(self.pstyles.print, 'CSPropeller_1')
        ps_print_2 = partial(self.pstyles.print, 'CSPropeller_2')

        ps_print_1(f"Constant Speed Analysis -> V0 = {val2str(self.V0)}, "
                   f"RPM = {val2str(self.RPM)}, "
                   f"Qs = {val2str(self._Qs_target)}:")

        # Note: self._run_cs() is always called when the object state is
        # valid, so the first analysis point is already computed.
        β0_last = self.β0
        Qs_last = self.Qs
        dQs_dβ0 = None

        # Note: The CS convergence algorithm below took an awful lot of
        # time to get right and is now quite bulletproof.  Change with
        # caution.

        β0_tried = np.array([self.β0])
        β0_close = np.deg2rad(0.05)  # Skip pts if closer than this.
        Δβ0_search = (self.β0_range[1] - self.β0_range[0]) / 50
        # Note: Δβ0_search is used if previous points did not converge /
        # do not exist.

        # Define function that checks if a potential trial point is
        # within range and sufficiently different from previous
        # attempts.
        def _original_β0(β0_check: float) -> bool:
            if ((self.β0_range[0] <= β0_check <= self.β0_range[1])
                    and np.all(np.abs(β0_tried - β0_check) > β0_close)):
                return True
            else:
                return False

        # -- Outer Loop - Converge β0 ----------------------------------

        for it in range(self.maxits_cs):

            # -- Check Convergence -------------------------------------

            error = np.abs(self.Qs - self._Qs_target)
            if self._Qs_target != 0:
                # Use relative error if Qs_target != 0.
                error /= self._Qs_target

            if error < self.tol_cs:
                ps_print_1(f"-> Converged: β0 = {rad2str(self.β0)}, "
                           f"T = {val2str(self.T)}, "
                           f"Qs = {val2str(self.Qs)}")
                return

            # -- Generate New β0 ---------------------------------------

            β0_next = None

            if dQs_dβ0 is not None:
                # Case: Good last step and gradient computed. The next
                # β0 value is computed using one step of Newton's method
                # from the current point, clipped to limits.

                β0_next = self.β0 + (self._Qs_target - self.Qs) / dQs_dβ0
                β0_next = np.clip(β0_next, *self.β0_range)

            else:
                # Case: All others.  Spiral away from the present
                # position and try to generate a new point that isn't
                # near a previously tried point.
                for i_next in range(1, 100):
                    # Loop starts at one giving the nearest trial point
                    # adjacent to the current 'β0'.

                    # Try positive side then negative side.
                    for side in (+1, -1):
                        β0_try = self.β0 + side * Δβ0_search
                        if _original_β0(β0_try):
                            β0_next = β0_try
                            break

            if β0_next is None:
                # Not a single trial point could be generated... highly
                # unusual.  Exit the 'for' loop to raise exception
                # signalling failure.
                break

            # -- Inner Loop - Solve Flow for β0 ------------------------

            β0_tried = np.append(β0_tried, β0_next)
            try:
                # Run fixed-pitch analysis.
                super().set_state(β0=β0_next)
                Δβ0 = self.β0 - β0_last
                ΔQs = self.Qs - Qs_last

                # Update the gradient.  If a previous estimate exists,
                # use relaxation = 0.5.  This significantly improves
                # stability of the CS convergence particularly where
                # there are some discontinuities / other nonlinearities.

                if dQs_dβ0 is not None:
                    dQs_dβ0 = 0.5 * dQs_dβ0 + 0.5 * ΔQs / Δβ0
                else:
                    dQs_dβ0 = ΔQs / Δβ0

                # Update progress.
                β0_last = self.β0
                Qs_last = self.Qs

                ps_print_2(
                    f"Iteration {it + 1:3d} of {self.maxits_cs:3d}: "
                    f"β0 = {rad2str(self.β0)} (Δβ0 = {rad2str(Δβ0)}), "
                    f"Qs = {val2str(self.Qs)} (ΔQs = {val2str(ΔQs)})")

            except SolverError:
                # If we fail to converge we no longer have a sequence
                # of good steps.  Reset slope and continue to try new
                # points.
                dQs_dβ0 = None
                ps_print_2(
                    f"Iteration {it + 1:3d} of {self.maxits_cs:3d}: "
                    f"β0 = {rad2str(self.β0)} - Convergence failed.")

        # --------------------------------------------------------------

        raise SolverError(f"Failed to converge β0 for Qs = "
                          f"{val2str(self._Qs_target)}")

    def _cs_propeller_set_state(
            self, *,  # No defaults checks for missing keywords.
            Ps: float | None, Qs: float | None,
            **kwargs) -> frozenset[str]:
        # Internal implementation of `set_state` that sets, checks and
        # actions all state variables for only this level in the class
        # hiearchy.
        changed = frozenset()

        if 'β0' in kwargs:
            raise ValueError("β0 can't be set directly for CSPropeller.")

        if Ps is not None:
            # Set torque based on power and Ω.
            if Qs is not None:
                raise InvalidStateError("Can't set both 'Ps' and 'Qs'.")
            if self.stopped:
                raise InvalidStateError("Can't set 'Ps' for a stopped "
                                        "propeller.")
            Qs = Ps / self.Ω

        if (Qs is not None) and (Qs != self._Qs_target):
            changed |= {'Qs'}
            self._Qs_target = Qs

        if not self.stopped and np.isnan(self._Qs_target):
            raise InvalidStateError("'Qs' undefined.")

        return changed

# ----------------------------------------------------------------------
