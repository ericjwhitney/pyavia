from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson

from ._propeller2 import BladedPropeller
from pyavia.aerodynamics import Foil2DAero, Blend2DAero
from pyavia.data import PCHIPModel1D, LineXYModel1D
from pyavia.iter import find_bracket

# Written by Eric J. Whitney, January 2023.

d2r, r2d = np.deg2rad, np.rad2deg


# ============================================================================

class GenericBEFixedProp2(BladedPropeller, ABC):
    """
    Base class with common features of all Blade Element Methods.
        - Any units can be used, provided they are consistent.
        - Interpolation is used to determine intermediate positions of
          various quantities (e.g. `c`, `β`, `cl`, `cd`, etc) - Deflections
          of the blade are not  considered.
        - Some attributes such as thrust (`T`), torque (`Qs`), etc require
          the propeller to be have a converged flowfield before they
          are useful.  Attempting to access these when not converged returns
          `NaN`.
        - This generic class assumes the propeller to be fixed pitch,
          so that calls to `_run` at this level do not change any overall
          pitch.  Variable pitch is introduced in derived classes.
    """

    def __init__(self, r: ArrayLike, c: ArrayLike, β: ArrayLike, B: int,
                 foils: list[Foil2DAero] | Foil2DAero, *,
                 redist: int | None = 20,
                 a0: float = None, μ0: float = None,
                 maxits_flow: int = 10, tol_flow: float = 1e-6,
                 disp: int | bool = False, **kwargs):
        """
        TODO
        Parameters
        ----------
        r : array_like, shape (n_stn,)
            Array of propeller radial stations i.e. [`r0`, `r1`, ...]. At
            least two stations are required and the stations must be sorted
            in order of increasing radius. These values are copied for
            internal use.

        c : array_like, shape (n_stn,) or float
            Chord length/s along the blade:
            - If a list/array is provided this represents the chord at
              each corresponding radial station, i.e. [`c0`, `c1`, ...].
            - If only a single value is given a constant chord blade is
              assumed for all stations.

        β : array_like, shape (n_stn,) or float
            Total pitch / chord angle [radians] along the blade:
            - If a list/array is provided this represents the local
              angle at each radial station, i.e. [`β_local_0`, `β_local_1`,
              ...].
            - If only a single value is given this is assumed to be the
              angle at *r = 0.75R*. The incidence of other stations is set
              to give constant pitch, ignoring β0.

            .. note:: See `BladedPropeller.β` for the definintion of the pitch /
               chord angle.



        foils : [Foil2DAero] | Foil2DAero
        # TODO Keeps references doesn't copy

            Aerofoil/s performance model along the blade. Either:
                - List of performance models corresponding to each radial
                  station `r` i.e. [Model0, Model1, ...] with ``len(foils)
                  == len(r)``.

                  .. note::  If is this option is used in combination with
                     points redistribution e.g. ``redist == 20``, the
                     redistribution does *not* apply to the aerofoils,
                     which continue to apply at the radii originally
                     specified.  Parameters are interpolated to/from these
                     stations during analysis.

                - Single model.  This is assumed to apply at all
                  analysis stations (regardless of whether points are
                  redistributed).

        β0 : float, default = 0.0 TODO move
            Collective incidence of blade [radians].  This is added to `βr`
            at each station to give total angle, i.e. `β` = `βr` + `β0`.
            If provided, must be bounded by `β0_range`.

        pitch_control : str, default = ``'fixed'`` TODO move
            Propeller pitch control / operating mode. This can be either:
                - ``'fixed'``: Fixed pitch operation.  Only states affecting
                  the overall advance ratio `J` are needed to specify the
                  operating point i.e. some combination of `V0` and `Ω` (or
                  equivalent).
                - ``'cs'``: Constant speed operation.  In addition to
                  fixed pitch operating parameters, one additional parameter
                  is required to specify the shaft torque (to find the
                  required colective incidence `β0`).  This could be `Qs`,
                  `Ps` (or equivalent).

        β0_range : (float, float), optional TODO move
            Propeller collective incidence operating range.  Operations
            that change `β0` will be restricted to this range.  The default
            range is unrestricted movement `β0` = (-π, +π).

        redist: int or None, default = 20
            - `None`: Use the analysis stations provided.
            - `int`: Produce a new distribution of radial analysis stations
              `r` according to a sine distribution with the number of points
              requested. The root and tip locations are preserved.  This
              places more points close to the tip.  The starting `r`,
              `c` and `β` values are interpolated to find the new values.



        a0: float, optional
            Freestream speed of sound.

        μ0: float, optional
            Freestream viscosity.  The default is zero (inviscid flow)
            which gives Reynolds numbers of infinity at each radial
            analysis station.

        maxits_flow : int, default = 10
            Maximum number of iterations for convergence of flow.

        tol_flow : float, default = 1e-6
            Flow is considered converged when the maximum change in the
            converged parameter is less than this value.

        maxits_cs : int, default = 20
            Maximum number of iterations for convergence of propeller blade
            angle in constant speed analyses.  Most normal points should
            only require 4-5 iterations.

        tol_cs : float, default = 1e-3
            Convergence tolerance applied to target variable (e.g. power,
            torque, thrust) in constant speed analyses

            - If the target value is non-zero this represents a relative
              error - the error divided by the target value.  For example
              if the target power is 75 hp, then ``cs_tol = 0.01`` means
              the blade angle is converged if shaft power is within 1%,
              or within the range 74.25 hp <-> 75.75 hp.
            - If the target value is zero this is applied as an absolute
              tolerance.

        **kwargs :
            For remaining arguments `V0`, `J`, `Ω`, `n`, `RPM`, `ρ0`, etc,
            see `BladedPropeller.__init__`.

        Notes
        -----
        At this level a flow calculation is not triggered at the end of
        `__init__`.  That is handled by the derived classes.
        """
        # Attribute suffix convention:
        #   - _stn: Values at (user supplied) radial analysis stations.
        #   - _el (or no suffix): Values at blade elements.

        # -- Basic Propeller & Atmosphere ------------------------------------

        super().__init__(**kwargs)
        self._a0, self._μ0 = np.nan, np.nan
        self._set_be_states(a0=a0, μ0=μ0)

        # -- Setup Initial r, c, β, B ----------------------------------------

        # Radii.
        self._r_stn = np.array(r, ndmin=1, copy=True)
        r_foils = self._r_stn.copy()  # Separately record for aerofoils.

        if self._r_stn.ndim != 1:
            raise ValueError("Radial stations 'r' must have shape (n_stn,).")
        if self.n_stn < 5:
            raise ValueError(f"Require len(r) >= 5, got {self.n_stn}.")
        if np.any(np.diff(self._r_stn) <= 0):
            raise ValueError("Radial stations 'r' must strictly increase.")

        # Chords.
        self._c_stn = np.array(c, ndmin=1, copy=True)
        if len(self._c_stn) == 1:
            # Expand single constant chord to all stations.
            self._c_stn = np.full_like(self._r_stn, self._c_stn[0])

        if len(self._c_stn) != len(self._r_stn):
            raise ValueError("Chords 'c' must have shape (n_stn,).")

        # Angles.
        self._β_stn = np.array(β, ndmin=1, copy=True)
        if len(self._β_stn) == len(self._r_stn):
            # All angles supplied.
            β075 = None

        elif len(self._β_stn) == 1:
            # Const. pitch angle supplied - Assumed to correspond to
            # 0.75R.  Expand this angle as a constant pitch at all radii.
            β075 = self._β_stn[0]
            self._set_const_pitch(β075)

        else:
            raise ValueError("'β' must be array of shape (n_stn,) or single "
                             "0.75R value.")

        # Setup interpolators for chord and pitch.
        self._make_geo_interpolators()

        # # Number of blades.
        # self._B = B
        # if self._B < 1:
        #     raise ValueError(f"Require B >= 1, got {self._B}.")

        # -- Redistribute r, c, β (If Reqd) ----------------------------------

        if redist is not None:
            if redist < 5:
                raise ValueError(f"Require redist >= 5, got {redist}.")

            # Redistribute r via sine distribution.
            t_root = np.arcsin(self._r_stn[0] / self._r_stn[-1])
            self._r_stn = self.R * np.sin(np.linspace(
                t_root, 0.5 * np.pi, num=redist))

            # Redistribute chords using current interpolator.
            self._c_stn = self._c_interp(self._r_stn)

            # Redistribute angles.
            if β075 is None:
                # Using current interpolator.
                self._β_stn = self._β_interp(self._r_stn)

            else:
                # Assuming constant pitch angle.
                self._set_const_pitch(β075)

            # Remake interpolators.
            self._make_geo_interpolators()

        # -- Setup Element Values --------------------------------------------

        # Element geometry (fixed).  Chords and pitch angles are determined
        # using an interpolating function (this is equivalent to using the
        # midpoint rule for integration).
        self._Δr_el = self._r_stn[1:] - self._r_stn[:-1]
        self._r_el = 0.5 * (self._r_stn[:-1] + self._r_stn[1:])
        self._c_el = np.asarray(self.c(self._r_el))
        self._β_el = np.asarray(self.β(self._r_el))

        # Element aerofoils.
        if isinstance(foils, Sequence):
            # Prescribed stations for each aerofoil.
            if len(foils) < 2:
                raise ValueError("Require len(foils) >= 2.")
            if len(foils) != len(r_foils):
                raise ValueError("Require len(foils) == len(r).")

            # Make blended aerofoil models for blade elements.
            self._foils_el = []
            for r_i in self._r_el:
                # Add intermediate aerofoils.
                idx_a, idx_b = find_bracket(r_foils, r_i)
                r_a, r_b = r_foils[idx_a], r_foils[idx_b]
                frac_a2b = (r_b - r_i) / (r_b - r_a)
                self._foils_el.append(
                    Blend2DAero(foils[idx_a], foils[idx_b], frac_a2b))

        else:
            # Use the same aerofoil for all elements.
            self._foils_el = [foils] * self.element_n

        # -- Setup Outputs / Other Parameters --------------------------------

        self.maxits_flow, self.tol_flow = maxits_flow, tol_flow
        self.disp = disp
        self._not_converged()  # Setup _converged and element forces.

    # -- Public Methods -----------------------------------------------------

    @property
    def a0(self) -> float:
        """Freestream speed of sound (:math:`a_0`)."""
        return self._a0

    @property
    def AF(self) -> float:
        """
        Blade Activity Factor (AF) for one blade, see `BladedPropeller.AF` for
        definition.

        Notes
        -----
        The integration is done via Simpson's rule over the known radial
        stations.  Simpson's Rule requires an odd number of points,
        so if the propeller has an even number then trapezoidal integration
        is done at the blade root.  This allows for more accurate
        integration at the tip where this is desirable.
        """
        return (100000 / 16) * simpson(
            (self._c_stn / self.D) * self._x_stn ** 3, self._x_stn,
            even='last')

    # def α(self) -> NDArray:
    #     """
    #     Returns angle of attack at each radial station.  This is computed by
    #         extrapolating the element-wise values held internally.
    #     """
    #     if self._α_el is _NOT_READY:
    #         raise NotSetError("'α' is not available.")
    #     return _change_radii(self._α_el, self._r_el, self._r_stn, widen=True)

    # @property
    # def B(self) -> int:
    #     return self._B

    def β(self, r: ArrayLike) -> ArrayLike:
        return self._β_interp(r)

    def c(self, r: ArrayLike) -> ArrayLike:
        return self._c_interp(r)

    def cd(self, r: ArrayLike) -> ArrayLike:
        """
        Returns the drag coefficient at one or more given radial station/s `r`.
        """
        return self._cd_interp(r)

    def cl(self, r: ArrayLike) -> ArrayLike:
        """
        Returns the lift coefficient at one or more given radial station/s `r`.
        """
        return self._cl_interp(r)

    @classmethod
    def def_states(cls) -> frozenset[str]:
        return super().def_states() | {'a0', 'μ0'}

    @property
    def element_β(self) -> NDArray[float]:
        """
        Array of shape (`n_el`,) giving of the overall pitch angle `β` for
        each element.
        """
        return self._β_el

    @property
    def element_c(self) -> NDArray[float]:
        """
        Array of shape (`n_el`,) giving of the chord for each element.
        """
        return self._c_el

    @property
    @abstractmethod
    def element_cd(self) -> NDArray[float]:
        """
        Array of shape (`n_el`,) giving of the drag coefficient for each
        element.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def element_cl(self) -> NDArray[float]:
        """
        Array of shape (`n_el`,) giving of the lift coefficient for each
        element.
        """
        raise NotImplementedError

    @property
    def element_foils(self) -> list[Foil2DAero]:
        """
        List of length `n_el` holding a reference to a `Foil2DAero` object
        corresponding to each blade element.
        """
        return self._foils_el

    @property
    def element_n(self) -> int:
        """Number of blade elements (= `n_stn` - 1)."""
        return len(self._r_stn) - 1

    @property
    @abstractmethod
    def element_Vp(self) -> NDArray[float]:
        """
        Array of shape (`n_el`, 2) giving of the axial and tangential
        components of flow velocity through the disc relative to each
        element [(Vp_a_0, Vp_t_0), (Vp_a_1, Vp_t_1), ...].
        """
        raise NotImplementedError

    # @property
    # @abstractmethod
    # def element_V2(self) -> NDArray:
    #     """
    #     Array of shape (`n_el`, 2) giving of the axial and tangential
    #     components of local flow velocity immediately behind each element
    #     [(V2_a_0, V2_t_0), (V2_a_1, V2_t_1), ...].
    #     """
    #     raise NotImplementedError

    @property
    def element_r(self) -> NDArray[float]:
        """
        Array of shape `(n_el,)` giving the radii for the blade elements
        from root to tip.
        """
        return self._r_el

    @property
    def element_x(self) -> NDArray[float]:
        """
        Array of radius fractions, shape `(n_el,)` giving `x = r_el/R`.
        """
        return self._r_el / self._r_stn[-1]

    @classmethod
    def input_states(cls) -> frozenset[str]:
        return super().input_states() | {'a0', 'μ0'}

    # Future work: This can probably be pulled out of the class and made
    # into a function that works 'on' the class.
    # def design(self, target: str, *, cl: ArrayLike = None,
    #            maxits_opt: int = 50, disp: int | bool = None, **kwargs):
    #     """
    #     Modify the geometry of the propeller at a specified operating
    #     condition.  Required parameters will depend on the type of
    #     modification (see `target`).  Valid combinations are:
    #
    #     .. note:: This method applies to fixed pitch propeller operation
    #        only (``pitch_control == 'fixed'``).  A constant speed propeller
    #        will automatically balance the power / torque and this would not
    #        converge to a valid solution.
    #
    #     Parameters
    #     ----------
    #     target : str
    #         Type of changes to make to propeller to accomplish the design
    #         objective:
    #         - ``'twist'``:  The local section angles along the blade `βr`
    #           are individually varied to give the loading specified by the
    #           `cl` parameter.  In this case, `Ω` (or equivalent) must be
    #           used by itself to speciy shaft conditions in `**kwargs`.
    #         - ``'twist-chord'``:  In addition to changes made by
    #           ``'twist'``, the chord ditribution is scaled by an overall
    #           factor to balance the required torque. In this case the `Ps` or
    #           `Qs` parameter must be added to ``**kwargs`` to allow the
    #           torque to be determined.
    #
    #     cl : shape (nr,), array_like
    #         Desired section cl values.
    #
    #     maxits_opt : int, default = 50.
    #         Maximum number of iterations allowed for modification.
    #
    #     disp : bool, int, optional
    #         See `run(...)` for details.
    #
    #     kwargs : {str: value}
    #         Remaining keyword arguments define the operating condition and
    #         are passed directly to ``operate(...)``.
    #
    #     Raises
    #     ------
    #     ValueError
    #         For illegal combinations or values of parameters.
    #     SolverError
    #         If maximum iterations are exceeded / the flow could not be
    #         converged.
    #     """
    #     disp_enter(disp)
    #
    #     # Rearrange arguments into required operating values.
    #     kwargs = self._reorder_run_kwargs(kwargs)
    #
    #     # Set pitch control up front for checking.
    #     self._pitch_control = kwargs.pop('pitch_control',
    #                                      self._pitch_control)
    #
    #     # Future work: Add more design types as reqd?
    #     if target in ('twist', 'twist-chord'):
    #         # For fixed-pitch propeller, blade angles are adjusted to give
    #         # lift coefficient.  Optionally chords are adjusted to give
    #         # requested torque.
    #         if self._pitch_control != 'fixed':
    #             raise ValueError(f"Modification '{target}' applicable only "
    #                              f"to fixed pitch propellers.")
    #
    #         if target == 'twist-chord':
    #             # For combined twist-chord, the shaft torque becomes the
    #             # target value as the propeller will be fixed pitch.
    #             try:
    #                 Q_target = kwargs.pop('Qs')
    #             except KeyError:
    #                 raise TypeError(f"Could not determine target torque to "
    #                                 f"modify blade chord.")
    #         else:
    #             Q_target = None
    #
    #         if cl is None:
    #             raise ValueError("Lift coefficient must be specified.")
    #
    #         disp_print(f"Modifying propeller '{target}':")
    #
    #         # Restrict target cl to elements.
    #         cl_target = _change_radii(cl, self._r_stn, self._r_el)
    #
    #         # Main design loop.
    #         for it in range(maxits_opt):
    #
    #             # Solve the propeller.
    #             self.run(**kwargs)  # type: ignore
    #
    #             # Check convergence.
    #             cl_err = cl_target - self._cl
    #             max_cl_err = np.max(np.abs(cl_err))
    #             if Q_target is not None:
    #                 Q_err = Q_target - self.Qs
    #             else:
    #                 Q_err = 0.0
    #
    #             # Future work: Make convergence parameters adjustable.
    #             if (max_cl_err <= 0.005) and (np.abs(Q_err) < 0.01):
    #                 # Prolong βr, c back to radial stations.
    #                 self._βr_stn = _change_radii(self._β_el, self._r_el,
    #                                              self.r(), widen=True)
    #                 self._c_stn = _change_radii(self._c_el, self._r_el,
    #                                             self.r(), widen=True)
    #
    #                 disp_print(f"\t-> Converged.  "
    #                            f"|error(cl)| = {max_cl_err:.05f}, "
    #                            f"error(Q) = {Q_err:.01f}")
    #                 disp_exit()
    #                 return
    #
    #             # Further refinement required.
    #             disp_print(f"\t-> Iteration {it + 1:3d} of {maxits_opt:3d}: "
    #                        f"|error(cl)| = {max_cl_err:8.05f}, "
    #                        f"error(Q) = {Q_err:8.01f}")
    #
    #             # Set relaxation factors to stabilise convergence.
    #             relax_β = 0.5
    #             relax_c = 0.25
    #
    #             # Adjust angle at each station using the cl error and an
    #             # assumed lift curve slope.
    #             self._β_el += relax_β * (cl_err / (2 * np.pi))
    #
    #             # If required, determine the chord scale factor as a simple
    #             # linear effect of the torque error.
    #             if Q_target is not None:
    #                 self._c_el *= (1 + relax_c * Q_err / Q_target)
    #
    #         # Future work: Reset angles and chords?
    #         raise SolverError("Reached iteration limit without converging "
    #                           "modification.")
    #
    #     else:
    #         raise ValueError(f"Unknown modification '{target}'.")

    # def M(self) -> NDArray:
    #     """
    #     Returns
    #     -------
    #     M : shape (nr,), ndarray
    #         Mach number at each radial station.  This is computed by
    #         extrapolating the element-wise values held internally.
    #     """
    #     if self._M is _NOT_READY:
    #         raise NotSetError("'M' is not available.")
    #     return _change_radii(self._M, self._r_el, self._r_stn, widen=True)

    @property
    def μ0(self) -> float:
        """Freestream viscosity (:math:`μ_0`)."""
        return self._μ0

    @property
    def n_stn(self) -> int:
        """Number of radial stations."""
        return len(self._r_stn)

    @classmethod
    def output_states(cls) -> frozenset[str]:
        return super().output_states()  # TODO add to this?

    @property
    def Ps(self) -> float:
        """ Shaft power Ps = Q_s.Ω."""
        return self.Qs * self.Ω

    @property
    def Qs(self) -> float:
        return self._Qs

    @property
    def R(self) -> float:
        return self._r_stn[-1]

    @property
    def R_root(self) -> float:
        return self._r_stn[0]

    @property
    def r_stn(self) -> NDArray[float]:
        """
        Array of shape `(n_stn,)` giving the radial analysis stations used
        for the discrete propeller, i.e. [`r0`, `r1`, ...].
        """
        return self._r_stn

    # def Re(self) -> NDArray:
    #     """
    #     Returns
    #     -------
    #     Re : shape (nr,) ndarray
    #         Reynolds number at each radial station.  This is computed by
    #         extrapolating the element-wise values held internally.
    #
    #         .. note:: Values are converted to logarithms for extrapolation
    #            to preserve the character of the Reynolds Number.
    #     """
    #     if self._Re is _NOT_READY:
    #         raise NotSetError("'Re' is not available.")
    #     Re_log = np.log10(self._Re)
    #     Re_log_stn = _change_radii(Re_log, self._r_el, self._r_stn,
    #     widen=True)
    #     return np.power(10.0, Re_log_stn)

    def set_states(self, *, V0: float = None, J: float = None,
                   Ω: float = None, n: float = None, RPM: float = None,
                   ρ0: float = None, a0: float = None, μ0: float = None,
                   **kwargs) -> frozenset[str]:
        """
        Set propeller operating state using core or optional states.  Only
        a single value from each state group (core or optional) may be
        selected as shown below:

        +-------------------+---------------------------+
        |    State Group    |                           |
        +------+------------+          Affects          |
        | Core |  Optional  |                           |
        +======+============+===========================+
        | `V0` |    `J`     | Freestream Airspeed       |
        +------+------------+---------------------------+
        | `Ω`  | `n`, `RPM` | Rotational Speed          |
        +------+------------+---------------------------+
        | `ρ0` |            | Freetream Density         |
        +------+------------+---------------------------+
        | `a0` |            | Freestream Speed of Sound |
        +------+------------+---------------------------+
        | `μ0` |            | Freestream Viscosity      |
        +------+------------+---------------------------+

        If the states have changed, a flow calculation is triggered.

        Returns
        -------
        frozenset[str]
            Set of states that were changed.

        """
        # Setting superclass states covers V0, J, Ω, n, RPM, ρ0. Remaining
        # freestream parameters are set locally.
        changed = super().set_states(V0=V0, J=J, Ω=Ω, n=n, RPM=RPM, ρ0=ρ0)
        changed |= self._set_be_states(a0=a0, μ0=μ0)

        if changed:
            if self._run():
                self._converged()
            else:
                self._not_converged()

        return changed

    @property
    def σ(self) -> float:
        """
        Propeller solidity, see `BladedPropeller.σ` for definition.  Integration
        is performed using Simpson's Rule in the same manner as
        `DiscretePropeller2.AF`.
        """
        return (self.B / self.Ap) * simpson(self._c_stn, self._r_stn,
                                            even='last')

    @property
    def T(self) -> float:
        return self._T

    # -- Private Methods -----------------------------------------------------

    def _converged(self):
        """
        This method must be called on successful exit from _run().  It sets
        status flags and computes forces and performance.  Derived types
        can chain up to also handle the flowfield.

        Forces are found by combining current flow state at each blade element
        location along with internally stored aerodynamic coefficients and
        atmospheric values to compute the following (stored internally):
            - Thrust gradient: dT_dr = 0.5.ρ.V^2.B.c.(cl.cos(ϕ) - cd.sin(ϕ))
            - Torque gradient: dQ_dr = 0.5.ρ.V^2.B.c.r.(cd.cos(ϕ) + cd.sin(ϕ))
            - Integral of gradients to get the total thrust and torque on
              the propeller.
            - Local efficienct of each blade element η_local = (V/Ω) (dT/dQ)
        """
        self._is_converged = True

        # Compute local element flow speed and angle from components.
        Vp_a_el, Vp_t_el = self.element_Vp[:, 0], self.element_Vp[:, 1]
        Vp_mag = np.sqrt(Vp_a_el ** 2 + Vp_t_el ** 2)
        ϕ = np.arctan2(Vp_a_el, Vp_t_el)

        # Precompute some terms to save effort.
        fac = 0.5 * self._ρ0 * Vp_mag ** 2 * self._B * self._c_el
        T_term = (self.element_cl * np.cos(ϕ) - self.element_cd * np.sin(ϕ))
        Qs_term = self._r_el * (self.element_cd * np.cos(ϕ) +
                                self.element_cl * np.sin(ϕ))

        # Compute element forces and overall performance.
        self._dT_dr = fac * T_term
        self._dQs_dr = fac * Qs_term
        self._T = float(np.sum(self._dT_dr * self._Δr_el))
        self._Qs = float(np.sum(self._dQs_dr * self._Δr_el))
        self._η_local = (self._V0 / self._Ω) * (T_term / Qs_term)

        # Make flow interpolators.  At root and tip these are assumed to have
        # small linear extensions from the last element value.
        # TODO alpha as well?
        self._cl_interp = PCHIPModel1D(self._r_el, self.element_cl,
                                       extrapolate='linear')
        self._cd_interp = PCHIPModel1D(self._r_el, self.element_cd,
                                       extrapolate='linear')

    @abstractmethod
    def _run(self):
        """
        Calculate flow at the blade elements (implemented by derived class).
        Returns `True` if flow converged, otherwise `False`. Note: The
        output forces and performance are not calculated within `_run`.
        """
        raise NotImplementedError

    def _make_geo_interpolators(self):
        """
        Construct c, β interpolators based on `self._r_stn`, `self._c_stn`,
        `self._β_stn`.
        """
        self._c_interp = PCHIPModel1D(self._r_stn, self._c_stn,
                                      extrapolate=None)
        self._β_interp = PCHIPModel1D(self._r_stn, self._β_stn,
                                      extrapolate=None)

    # def _run_cs(self, target_attr: str, target_val: float,
    #             disp: int | bool = None):
    #     """
    #     Varies β0 to converge the given target attribute of the
    #     propeller to the value requested.
    #     """
    #     disp_enter(disp)
    #     disp_print(f"-> Converging {target_attr} = {target_val:+.1f}:")
    #
    #     failed = False
    #     y, dydβ0 = None, None
    #     β0_last, y_last = None, None,
    #
    #     # Note: The CS convergence algorithm below took an awful lot of
    #     time to get right and is now quite bulletproof.  Change with
    #     caution.
    #
    #     close_β_tol = d2r(0.05)
    #     β0_attempts = np.array([])
    #     Δβ0_search = (self._β0_range[1] - self._β0_range[0]) / 50
    #     # Note: Δβ0_search is used if previous points did not converge / do
    #     # not exist.
    #
    #     for it in range(self.maxits_cs):
    #
    #         # -- Generate New β0 --------------------------------------------
    #
    #         if ((not failed) and
    #                 (dydβ0 is not None) and
    #                 (self._β0_range[0] < self._β0 < self._β0_range[1])):
    #             # Case: Good last step, inside bounds, known gradient.
    #             # The next β0 value is computed using one step of Newton's
    #             # method from the current point, clipped to limits.
    #             self._β0 += (target_val - y) / dydβ0
    #             self._β0 = np.clip(self._β0, *self._β0_range)
    #
    #         else:
    #             # Case: All others.  Spiral away from the present position
    #             # and try to generate a new point that isn't near a
    #             # previously tried point.
    #             β0_next = None
    #             for i_next in range(0, 100):
    #                 # Loop starts at zero so that the very first point can be
    #                 # self._β0 itself.
    #
    #                 # Positive side.
    #                 β0_pos = self._β0 + i_next * Δβ0_search
    #                 if ((self._β0_range[0] <= β0_pos <= self._β0_range[1])
    #                         and np.all(np.abs(β0_attempts - β0_pos) >
    #                                    close_β_tol)):
    #                     β0_next = β0_pos
    #                     break
    #
    #                 # Negative side.
    #                 β0_neg = self._β0 - i_next * Δβ0_search
    #                 if ((self._β0_range[0] <= β0_neg <= self._β0_range[1])
    #                         and np.all(np.abs(β0_attempts - β0_neg) >
    #                                    close_β_tol)):
    #                     β0_next = β0_neg
    #                     break
    #
    #             if β0_next is not None:
    #                 self._β0 = β0_next
    #             else:
    #                 # Not a single trial point could be generated... highly
    #                 # unusual.  This is a clear failure.
    #                 break
    #
    #         # -- Run New β0 -------------------------------------------------
    #
    #         β0_attempts = np.append(β0_attempts, self._β0)
    #         failed = False
    #         try:
    #             self._run_fixed()
    #             y = getattr(self, target_attr)
    #             disp_print(f"\t... Iteration {it + 1:3d} of "
    #                        f"{self.maxits_cs:3d}: "
    #                        f"β0 = {r2d(self._β0):+6.02f}°, "
    #                        f"{target_attr} = {y:+11.1f}")
    #
    #         except SolverError:
    #             disp_print(f"\t... Iteration {it + 1:3d} of "
    #                        f"{self.maxits_cs:3d}: "
    #                        f"β0 = {r2d(self._β0):+6.02f}°, "
    #                        f"Convergence failed.")
    #             failed = True
    #             continue
    #
    #         # -- Update Gradient / Check Convergence ------------------------
    #
    #         if β0_last is not None:
    #             # Update the gradient.  If a previous estimate exists,
    #             # use relaxation = 0.5.  This significantly improves
    #             # stability of the CS convergence particularly where there
    #             # are some discontinuities / other nonlinearities.
    #             Δβ0 = self._β0 - β0_last
    #             Δy = y - y_last
    #             if dydβ0 is not None:
    #                 dydβ0 = 0.5 * dydβ0 + 0.5 * Δy / Δβ0  # Relax
    #
    #             else:
    #                 dydβ0 = Δy / Δβ0  # No relaxation on first step.
    #
    #             # Check convergence.
    #             error = np.abs(Δy)
    #             if target_val != 0:
    #                 error /= target_val
    #
    #             if error < self.tol_cs:
    #                 disp_print(f"\t-> Converged.")
    #                 disp_exit()
    #                 return
    #
    #         # Update last 'good' point.
    #         β0_last, y_last = self._β0, y
    #
    #     # -------------------------------------------------------------------
    #
    #     disp_exit()
    #     raise SolverError(f"Failed to converge constant speed operation "
    #                       f"for {target_attr} = {target_val:+.1f}")

    def _not_converged(self):
        """
        This method must be called on exit from `_run` if the method did not
        converge (it is also called by GenericBEFixedProp during
        `__init__`). This sets up / resets blade element and overall forces.
        Derived types can chain up to also reset the flowfield.
        """
        self._is_converged = False

        # Set forces to NaN.
        self._dT_dr = np.full(self.element_n, np.nan)
        self._dQs_dr = np.full(self.element_n, np.nan)
        self._Qs, self._T = np.nan, np.nan
        self._η_local = np.full(self.element_n, np.nan)

        # Set flow interpolators to NaN.
        # TODO alpha as well?
        all_nan = LineXYModel1D([self.R_root, self.R], [np.nan, np.nan],
                                extrapolate=None)
        self._cl_interp = all_nan
        self._cd_interp = all_nan

    def _set_be_states(self, a0: float | None,
                       μ0: float | None) -> frozenset[str]:
        """
        Common code for `__init__` and `set_states`.  See `set_states` for
        details.
        """
        changed = frozenset()
        if a0 is not None:
            self._a0 = a0
            changed |= {'a0'}

        if self._a0 <= 0.0:
            raise ValueError(f"Require a0 > 0, got: {self._a0}")

        if μ0 is not None:
            self._μ0 = μ0
            changed |= {'μ0'}

        if self._μ0 < 0.0:
            raise ValueError(f"Require μ0 >= 0, got: {self._μ0}")

        return changed

    def _set_const_pitch(self, β075: float):
        """
        Set all `self._β_stn` along the blade to give constant pitch angle
        using the β value given for 0.75R.
        """
        self._β_stn = np.arctan(np.tan(β075) * 0.75 / self._x_stn)

    @property
    def _x_stn(self) -> NDArray[float]:
        """
        Array of radius fractions, shape `(n_stn,)` giving `x = r_stn/R`.
        """
        return self._r_stn / self._r_stn[-1]

# ============================================================================
