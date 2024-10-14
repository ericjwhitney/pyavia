from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
import warnings
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson

from pyavia.aerodynamics import Foil2DAero, Blend2DAero
from pyavia.numeric.function_1d import PCHIP1D
from pyavia.ops import find_bracket
from pyavia.numeric.math_ext import check_sclarray
from pyavia.propulsion import Propeller, const_pitch_β
from ._propeller import _SclArray
from pyavia.numeric.solve import SolverError

from pyavia.numeric._dev._post_init import PostInitMeta
from pyavia.util.print_styles import (
    PrintStylesMixin, AddTabStyle, val2str)

# Written by Eric J. Whitney, January 2023.

usval2str = partial(val2str, signed=False)  # Unsigned shorthand.


# ======================================================================

class BEPropeller(PrintStylesMixin, Propeller, metaclass=PostInitMeta):
    """
    Abstract type extending `Propeller` that defines the characteristics
    shared by all types of blade element based propeller models with a
    finite number of blades.

    Parameters
    ----------
    B : int
        Number of blades (finite).

    βr : array_like, shape (N,) or float
        Local incidence variation [radians] along the blade.

        - If a list/array is provided this represents the local
          incidence at each radial station, i.e. [`βr_0`, `βr_1`, ...].

        - A single value can be provided to specify a constant-pitch
          blade design.  In this case the value is assumed to
          be the *local* incidence angle added to `β0` at *r = 0.75R*.

        .. note::This value is added to the initial `β0` value to
           give the total incidence angle at `r = 0.75R`.  The total
           incidence for each element is then computed to give constant
           pitch along the blade.  The local variation `βr` for each
           element is finally computed by subtracting `β0` again.

        .. note::See `Propeller.β` for the definintion of the total
           vs. local incidence angle.

    foils : Sequence[Foil2DAero] or Foil2DAero
        Foil/s performance model along the blade. Either:

        - A sequence of performance models is supplied corresponding to
          each radial station `r` i.e. [Model0, Model1, ...] with
          ``len(foils) == len(r)``.

          .. note::  If is this option is used in combination with
             points redistribution e.g. ``redist == 20``, the
             redistribution does *not* apply to the foils, which
             continue to apply at the radii originally specified.


        - A single model is supplied which applies at all radii.

        Foil objects are deep copied for internal storage and use.

    maxits_flow : int, default = 10
        Maximum number of iterations allowed for convergence of flow.

    tol_flow : float, default = 1e-6
        The flow is considered converged when the maximum change in the
        flow variables (method dependent) is less than this value.

    TODO MORE

    TODO disp 0, 1, 2

    Notes
    -----
    - Interpolation is used to determine intermediate positions of
      various quantities (e.g. `c`, `β`, `cl`, `cd`, etc) - Deflections
      of the blade are not considered.
    """

    # -- Inner Classes -------------------------------------------------

    class DefPoints:
        """
        Inner class for `BEPropeller` for logical layout of data
        corresponding to defining points on the geometry.
        """
        def __init__(self, prop: BEPropeller, r: npt.ArrayLike,
                     c: npt.ArrayLike, βr: npt.ArrayLike,
                     redist: int | None):
            self._prop = prop

            # -- Setup Radii (r) ---------------------------------------

            self._r, _ = check_sclarray(r, dtype=float, copy=True)
            if self.n < 2:
                raise ValueError(f"Require >= 2 radii, got {self.n}.")

            if np.any(np.diff(self._r) <= 0):
                # TODO change ^^^ to 'strict_increase` once that is improved.
                raise ValueError("'r' must be strictly increasing.")

            # -- Chord (c) Interpolator --------------------------------

            # Make the chord (c) interpolator.  If a scalar was given,
            # apply a constant chord to all stations.
            c, single_c = check_sclarray(c, dtype=float)  # No copy.
            if single_c:
                c = np.full(self.n, c[0])

            elif len(c) != self.n:
                raise ValueError(f"Chord 'c' must be scalar or have "
                                 f"shape ({self.n},).")

            self._c_interp = PCHIP1D(self._r, c)  # Makes copies.

            # -- Incidence Angle (β) Interpolator ----------------------

            # If a scalar was given, it corresponds to βr @ x=0.75.
            # Calculate βr to then given give constant *total* pitch (β)
            # along the blade.
            βr, single_βr = check_sclarray(βr, dtype=float)  # No copy.
            if single_βr:
                x = self._r / self._r[-1]  # x @ elements.
                β_x75 = self._prop.β0 + βr[0]  # Total β @ 0.75R.
                βr = const_pitch_β(β_x75, x) - βr[0]  # βr @ elements.

            elif len(βr) != self.n:
                raise ValueError(f"'βr' must be scalar or have shape "
                                 f"({self.n},).")

            self._βr_interp = PCHIP1D(self._r, βr)  # Makes copies.

            # -- Redistribute ------------------------------------------

            # If requested redistribute radii using a sine distribution.
            # The interpolators themselves are not changed.
            if redist is not None:
                if redist < 5:
                    raise ValueError(f"Require redist >= 5, got {redist}.")

                R_root, R = self._r[0], self._r[-1]
                t_root = np.arcsin(R_root / R)
                self._r = R * np.sin(np.linspace(t_root, 0.5 * np.pi,
                                                 num=redist))

        # -- Public Methods --------------------------------------------

        def c(self, r: npt.ArrayLike) -> _SclArray[float]:
            return self._c_interp(r)

        @property
        def n(self) -> int:
            """
            Number of radial stations that define the blade geometry.
            """
            return len(self._r)

        @property
        def r(self) -> npt.NDArray[float]:
            """
            Radii of stations that define the blade geometry as
            `ndarray` of shape `(n,)`, e.g. `[r_0, r_1, ..., r_n-1]`.
            """
            return self._r

        def βr(self, r: npt.ArrayLike) -> _SclArray[float]:
            return self._βr_interp(r)

    # ------------------------------------------------------------------

    class Elements(ABC):
        """
        Inner class for `BEPropeller` for for logical layout of data
        corresponding to blade elements.
        """
        def __init__(self, prop: BEPropeller, r_foils: npt.ArrayLike,
                     foils: Sequence[Foil2DAero] | Foil2DAero):
            """
            Initialise the blade elements. `prop` must have constructed
            `def_points` prior to call.
            """
            self._prop = prop

            # -- Setup Elements ----------------------------------------

            # Chords and incidence angles are determined using
            # interpolating functions from the parent propeller (this
            # is equivalent to using the midpoint rule for integration).
            r_def = self._prop.def_points.r
            self._r = 0.5 * (r_def[:-1] + r_def[1:])
            self._Δr = r_def[1:] - r_def[:-1]
            self._c = self._prop.c(self._r)
            self._βr = self._prop.βr(self._r)

            if self.n < 10:
                warnings.warn(f"Low number of blade elements in use "
                              f"({self.n}).  Accuracy may be affected.")

            # -- Setup Aerofoils ---------------------------------------

            r_foils, _ = check_sclarray(r_foils, dtype=float, copy=True)
            n_foils = len(r_foils)
            if isinstance(foils, Sequence):
                # Multiple aerofoils provided. These should align with
                # radial stations.
                if len(foils) != n_foils:
                    raise ValueError(f"Expected single or {n_foils} "
                                     f"foils, got {len(foils)}.")

                # Assign each element its own aerofoil.  Make copies
                # where required.
                self._foils = []
                for r_i in self._r:
                    # Check if this element radius lines up with any of
                    # the given aerofoil radii.
                    is_near = np.isclose(r_i, r_foils, atol=1e-9, rtol=1e-6)
                    if np.any(is_near):
                        # Lines up: Use a direct deepcopy.
                        near_idx = np.argmax(is_near)
                        self._foils.append(deepcopy(foils[near_idx]))

                    else:
                        # Make a separate aerofoil model by blending
                        # aerofoils on each side
                        idx_a, idx_b = find_bracket(r_foils, r_i)
                        r_a = r_foils[idx_a]
                        r_b = r_foils[idx_b]
                        frac = (r_b - r_i) / (r_b - r_a)
                        self._foils.append(
                            Blend2DAero(foils[idx_a], foils[idx_b], frac,
                                        deep_copy=True))

            else:
                # Single aerofoil provided.  Make an individual deep
                # copy for each blade element.
                # self._foils = [deepcopy(foils) for _ in range(self.n)]
                self._foils = [deepcopy(foils)] * self.n

            # -- Setup Aerodynamic Values ------------------------------

            # Element-wise values.
            self._cl = np.full(self.n, np.nan)
            self._cd = np.full(self.n, np.nan)
            self._dT_dr = np.full(self.n, np.nan)
            self._dQs_dr = np.full(self.n, np.nan)
            self._η_local = np.full(self.n, np.nan)

        # -- Public Methods --------------------------------------------

        @property
        def c(self) -> npt.NDArray[float]:
            """
            Blade element chords (`c`) as `ndarray` of shape `(n,)`.
            """
            return self._c

        @property
        def cd(self) -> npt.NDArray[float]:
            """
            Blade element drag coefficient (`cd`) as `ndarray` of shape
            `(n,)`.
            """
            return self._cd

        @property
        def cl(self) -> npt.NDArray[float]:
            """
            Blade element lift coefficient (`cl`) as `ndarray` of shape
            `(n,)`.
            """
            return self._cl

        @property
        def dT_dr(self) -> npt.NDArray[float]:
            """
            Blade element radial thrust gradient (math:`dT/dr`) as
            `ndarray` of shape `(n,)`.
            """
            return self._dT_dr

        @property
        def dQs_dr(self) -> npt.NDArray[float]:
            """
            Blade element radial torque gradient (math:`dQs/dr`) as
            `ndarray` of shape `(n,)`.
            """
            return self._dQs_dr

        @property
        def foils(self) -> list[Foil2DAero]:
            """
            Blade element `Foil2DAero` models as a list of length
            `n_elements`.
            """
            return self._foils

        @property
        def n(self) -> int:
            """The number of blade elements (`n`)."""
            return len(self._r)

        @property
        def r(self) -> npt.NDArray[float]:
            """
            Radii of blade element centroids as `ndarray` of shape `(n,)`.
            """
            return self._r

        @property
        def x(self) -> npt.NDArray[float]:
            """
            Radial fractions :math:`x_i=r_i/R` of blade element
            centroids as `ndarray` of shape `(n,)`.
            """
            return self._r / self._prop.R

        @property
        def β(self) -> npt.NDArray[float]:
            """
            Blade element total incidence (`β`) [radians] as `ndarray`
            of shape `(n,)`, where :math:`β = β_0 + β_r(r)`.
            """
            return self._prop.β0 + self._βr

        @property
        def Δr(self) -> npt.NDArray[float]:
            """
            Blade element width (`Δr`) as `ndarray` of shape `(n,)`,
            where :math:`Δr_i = r_{i+1} - r_i`.
            """
            return self._Δr

        # -- Protected Methods -----------------------------------------

        def _update_local_forces(self, Vp_mag: npt.NDArray[float],
                                 ϕ: npt.NDArray[float]):
            """
            Combines the current flow state given by |Vp| and ϕ along
            with pre-computed aerodynamic / atmospheric values to
            update / store the following at each blade element:
                - Thrust gradient:
                    dT/dr = 0.5.ρ.V^2.B.c.(cl.cos(ϕ) - cd.sin(ϕ))
                - Torque gradient:
                    dQ/dr = 0.5.ρ.V^2.B.c.r.(cd.cos(ϕ) + cl.sin(ϕ))
                - Local efficiency of each blade element:
                    η_local = (V/Ω) (dT/dQ)

            Parameters
            ----------
            Vp_mag : (n,) ndarray
                Local flow speed.
            ϕ : (n,) ndarray
                Local flow helix angle to disc.
            """
            # Precompute some terms to save effort.
            fac = (0.5 * self._prop.ρ0 * Vp_mag ** 2 * self._prop.B *
                   self._c)
            T_term = self._cl * np.cos(ϕ) - self._cd * np.sin(ϕ)
            Qs_term = self._r * (self._cd * np.cos(ϕ) + self._cl * np.sin(ϕ))

            # Local element gradients and local efficiency.
            self._dT_dr = fac * T_term
            self._dQs_dr = fac * Qs_term
            self._η_local = ((self._prop.V0 / self._prop.Ω) *
                             (T_term / Qs_term))

        @abstractmethod
        def _reset(self):
            """
            Reset state-dependent blade element values to NaN.

            *Derived classes must override and chain up with this
            method to ensure any additional attributes are cleared.*
            """
            self._cl[:] = np.nan
            self._cd[:] = np.nan
            self._dT_dr[:] = np.nan
            self._dQs_dr[:] = np.nan
            self._η_local[:] = np.nan

    # -- Main / Outer Class --------------------------------------------

    def __init__(self, *,
                 # Constants.
                 B: float, r: npt.ArrayLike, c: npt.ArrayLike,
                 βr: npt.ArrayLike,
                 foils: Sequence[Foil2DAero] | Foil2DAero,
                 # Options.
                 redist: int | None = 20,
                 maxits_flow: int = 10, tol_flow: float = 1e-6,
                 display_level: int = 0,
                 **kwargs):
        """
        .. note::At the end of `BEPropeller.__init__` if the propeller
           is in a valid state a flow solution will be commenced
           (unless the propeller is stopped).  Otherwise an
           `InvalidStateError` will be raised.
        """
        super().__init__(B=B, display_level=display_level, **kwargs)

        # Setup base propeller.
        if not np.isfinite(B):
            raise ValueError("Only finite 'B' values permitted.")

        # Setup inner class instances.
        self._def_points = self.DefPoints(self, r, c, βr, redist)
        self._elements = self.Elements(self, r, foils)

        # Setup solver options.
        self.maxits_flow = maxits_flow
        self.tol_flow = tol_flow

        # Setup formatting for _run_fixed() printed output.  This will
        # be the root print style.
        self.pstyles.add('BEPropeller', AddTabStyle())

        # Setup overall performance and convergence tracking.
        self._T = np.nan  # Total thrust.
        self._Qs = np.nan  # Total torque.

        # These internal values are retained to determine if we can
        # re-use previous run flow state or if we need to restart.
        self._last_J = np.nan
        self._last_β0 = np.nan

        # Set and check state inputs at this level.
        self._be_propeller_set_state()

    def __post_init__(self):
        """
        Runs after all `__init__` methods in the class hierarchy have
        completed, this method calls `self.set_state()` with no
        arguments.  This is designed to trigger a flow solution after
        the object is completely setup (if required).
        """
        self.set_state()

    # -- Public Methods ------------------------------------------------

    @property
    def AF(self) -> float:
        """
        Blade Activity Factor (AF) for one blade, see `Propeller.AF`
        for definition.

        Notes
        -----
        Integration is done via Simpson's Rule which requires an odd
        number of radial stations.  If an even number is present, then
        trapezoidal integration is done at the blade root, which allows
        for more accurate results at the tip where this is desirable.
        """
        c = self.c(self._def_points.r)
        x = self._def_points.r / self.R
        return (100000 / 16) * simpson((c / self.D) * x ** 3,  x, even='last')

    def c(self, r: npt.ArrayLike) -> _SclArray[float]:
        return self._def_points.c(r)

    @property
    def def_points(self) -> DefPoints:
        return self._def_points

    # @property
    # def display_level(self) -> int:
    #     return self._format.display_level
    #
    # @display_level.setter
    # def display_level(self, value: int):
    #     self._format.display_level = value

    @property
    def elements(self) -> Elements:
        return self._elements

    @property
    def Ps(self) -> float:
        """Shaft power :math:`P_s = Q_s Ω`."""
        return self.Qs * self.Ω

    @property
    def Qs(self) -> float:
        return self._Qs

    @property
    def R(self) -> float:
        return float(self._def_points.r[-1])

    @property
    def R_root(self) -> float:
        return float(self._def_points.r[0])

    def set_state(self, **kwargs) -> frozenset[str]:
        """
        Set propeller operating state using groups of related input
        values.  Only a single value from each state group (primary or
        optional) may be selected as shown below:

        +----------------------+---------------------------+
        |      State Group     |                           |
        +---------+------------+          Affects          |
        | Primary |  Optional  |                           |
        +=========+============+===========================+
        |  `V0`   |    `J`     | Freestream Airspeed       |
        +---------+------------+---------------------------+
        |   `Ω`   | `n`, `RPM` | Rotational Speed          |
        +---------+------------+---------------------------+
        |  `β0`   |            | Collective Incidence      |
        +---------+------------+---------------------------+
        |  `ρ0`   |            | Freetream Density         |
        +---------+------------+---------------------------+
        |  `a0`   |            | Freestream Speed of Sound |
        +---------+------------+---------------------------+
        |  `μ0`   |            | Freestream Viscosity      |
        +---------+------------+---------------------------+

        .. note::If any states are changed, the propeller flow is
           recalculated (if running) or zeroed (if stopped).

        Returns
        -------
        frozenset[str]
            Set of primary input attribute names that were changed.

        Raises
        ------
        InvalidStateError
            If inputs have incorrect values.
        """
        changed = super().set_state(**kwargs)
        changed |= self._be_propeller_set_state()

        # If nothing has changed, return immediately.
        if not changed:
            return changed

        # Calculate flow - 'T' and 'Qs' will always change.
        self._run_fixed()
        return changed | {'T', 'Qs'}

    @property
    def T(self) -> float:
        return self._T

    def βr(self, r: npt.ArrayLike) -> _SclArray[float]:
        return self._def_points.βr(r)

    # TODO SIGMA
    """
    Integration is done via Simpson's Rule which requires an odd number
    of points.  If the propeller has an even number then trapezoidal
    integration is done at the blade root, which allows for more
    accurate at the tip where this is desirable.
    """

    # -- Protected Methods ---------------------------------------------

    # noinspection PyMethodMayBeStatic
    def _be_propeller_set_state(self) -> frozenset[str]:
        # Internal implementation of `set_state` that sets, checks and
        # actions all state variables for only this level in the class
        # hiearchy.

        # Note: Presently no new state inputs at this level.
        return frozenset()

    def _run_fixed(self):
        """
        Sets up a flow calculation for the current state and fixed β0,
        then call the relevant solver.
        """
        ps_print = partial(self.pstyles.print, 'BEPropeller')

        # If the propeller is stopped, no flow calculation is required.
        # Set internal values to reflect the stopped condition and
        # return.
        if self.stopped:

            # Future Work: Setting 'stopped' condition could allow for
            # calculation of blade drag / feathered condition.

            # Invalidate all element values (including derived types).
            # Expressly set T, Qs to zero
            self._reset()  # Sets T, Qs = NaN.
            self._T = 0.0
            self._Qs = 0.0
            return

        # Converge a flow solution for the changed state.
        try:
            ps_print(f"{self.__class__.__name__} Analysis -> "
                     f"V0 ={val2str(self.V0)}, "
                     f"RPM = {usval2str(self.RPM)}:")

            self._solve_flow()

            ps_print(f"-> Flow converged: T = {val2str(self.T)}, "
                     f"Qs = {val2str(self.Qs)}")

            # Save state for possible next run.
            self._last_J = self.J
            self._last_β0 = self.β0

        except SolverError as e:
            # Reset all state-dependent data and re-raise.
            self._reset()
            raise e

    def _reset(self):
        """
        Reset all state-dependent data (e.g. overall performance and
        blade element values) to NaN to prevent accidentally accessing
        invalid data.  This resets overall propeller data in the main
        class and also calls `_reset()` on blade elements.

        .. note::Derived classes should override and chain up with this
           method to add additional attributes as required.
        """
        self._T = np.nan
        self._Qs = np.nan
        # noinspection PyProtectedMember
        self._elements._reset()  # Blade element values.

    @abstractmethod
    def _solve_flow(self):
        """
        Derived classes must provide the method for calculating
        performance at the current state (fixed `β0`), by converging
        flow variables at each blade element.

        Raises
        ------
        SolverError
            If the solver fails to converge.
        """
        raise NotImplementedError

    # noinspection PyProtectedMember
    def _update_forces(self, Vp_mag: npt.NDArray[float],
                       ϕ: npt.NDArray[float]):
        """
        The local element forces are first updated based on local flow
        conditions |Vp| and ϕ.  Then the total thrust and torque are
        computed:
            - Total thrust: T = Σ(dT_dr.Δr)
            - Total torque: Qs = Σ(dQ_dr.Δr)
        """
        # Update element local forces.
        elems = self._elements
        elems._update_local_forces(Vp_mag, ϕ)

        # Total thrust and torque.
        self._T = np.sum(elems.dT_dr * elems.Δr)
        self._Qs = np.sum(elems.dQs_dr * elems.Δr)

# ----------------------------------------------------------------------


# ======================================================================
