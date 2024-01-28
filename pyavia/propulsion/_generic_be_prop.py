from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from pyavia.containers import WriteOnceDict
from pyavia.propulsion import DiscreteProp
from pyavia.type_ext import make_sentinel
from pyavia.solve.exception import SolverError
from pyavia.util import disp_enter, disp_print, disp_exit

d2r, r2d = np.deg2rad, np.rad2deg

_NOT_READY = make_sentinel('_NOT_READY')


# Future work: Add a plot function for plotting properties along the blade.

# ============================================================================

# Future work:  Make this a `States` object.

class GenericBEProp(DiscreteProp, ABC):
    """
    Base class with common features of Blade Element Methods for propellers:
        - Any units can be used, provided they are consistent.
        - Results are given at the specified radial stations. Interpolation
          is used to determine intermediate positions of various quantities
          (e.g. c, β, c_l, c_d, etc) - Deflections of the blade are not
          considered.
        - Some attributes such as thrust (`T`), torque (`Qs`), etc require
          the propeller to be run before they are available.  Attempting to
          access these beforehand causes a `RuntimeError` to be raised.

    Attribute suffix conventions:

        - `_stn`: Values at (user supplied) radial analysis stations.
        - No suffix or `_el`: Values at blade elements.
        - `_af`: Values at aerofoil section stations.
    """

    def __init__(self, r: ArrayLike, c: ArrayLike, βr: ArrayLike, B: int,
                 foils: [FoilAero] | FoilAero, β0: float = 0.0,
                 pitch_control: str = 'fixed', *,
                 β0_range: (float, float) = (-np.pi, +np.pi),
                 redist: int | None = 20,
                 ρ0: float = None, a0: float = None, μ0: float = 0,
                 maxits_flow: int = 10, tol_flow: float = 1e-6,
                 maxits_cs: int = 20, tol_cs: float = 1e-3,
                 **kwargs):
        """
        Parameters
        ----------
        r : shape (nr,) array_like
            1-D array of propeller radial stations of the i.e. [r0, r1,
            ...]. At least two stations are required and the stations must
            be sorted in order of increasing radius. These values are
            copied for internal use.

        c : shape (nr,) array_like | float
            Chord length/s along the blade:
                - If list/array is provided this represents the chord at
                  each corresponding radial station, i.e. [c0, c1, ...].
                - If only a single value is given a constant chord
                  blade is assumed for all stations.

        βr : array_like , optional
            Component of pitch angle [radians] that varies locally along
            the blade, in addition to the overall collective angle β0:
                - If a list/array is provided this represents the local
                  component of incidence at each radial station,
                  i.e. [β_local_0, β_local_1, ...].
                - If only a single value is given this is assumed to be the
                  incidence at *r = 0.75R*. The incidence of other stations
                  is set to give constant pitch, ignoring β0.

            .. note:: See method ``β(...)`` for the definintion of
               the pitch angle.

        B : int
            Number of blades (> 0).

        foils : [Foil2DBasic] | Foil2DBasic
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

        β0 : float, default = 0.0
            Collective incidence of blade [radians].  This is added to `βr`
            at each station to give total angle, i.e. `β` = `βr` + `β0`.
            If provided, must be bounded by `β0_range`.

        pitch_control : str, default = ``'fixed'``
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

        β0_range : (float, float), optional
            Propeller collective incidence operating range.  Operations
            that change `β0` will be restricted to this range.  The default
            range is unrestricted movement `β0` = (-π, +π).

        redist: int or None, default = 20
            - `None`: Use the analysis stations provided.
            - `int`: Using the given number of points, produce a new
              distribution of radial analysis stations `r` according to a
              sine distribution with thte number of points provided. The
              root and tip locations are preserved.  This places more points
              close to the tip.  The starting `r`, `c` and `βr` values are
              interpolated to find the new values.

        ρ0: float, optional
            Freestream air density.

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
        """
        super().__init__(**kwargs)

        # Setup radial stations.
        self._r_stn = np.array(r, ndmin=1, copy=True)
        if self._r_stn.ndim != 1:
            raise ValueError("Radial stations 'r' must have shape (nr,).")
        if len(self._r_stn) < 2:
            raise ValueError("At least two radial stations 'r' required.")
        if np.any(np.diff(self._r_stn) <= 0):
            raise ValueError("Radial stations 'r' must strictly increase.")

        # Set chords at radial stations.
        self._c_stn = np.array(c, ndmin=1, copy=True)
        if len(self._c_stn) == 1:
            # Expand single constant chord to all stations.
            self._c_stn = np.full_like(self._r_stn, self._c_stn[0])

        if len(self._c_stn) != len(self._r_stn):
            raise ValueError("Chords 'c' must have shape (nr,).")

        # Set local incidence angles at radial stations.
        if isinstance(βr, Sequence):
            self._βr_stn = np.array(βr, ndmin=1, copy=True)

        else:
            # If only one βr value was given it is assumed to
            # correspond to 0.75R.  Expand this angle as a constant pitch
            # at all radii.
            self._βr_stn = np.arctan(np.tan(βr) * 0.75 / self.x())

        if len(self._βr_stn) != len(self._r_stn):
            raise ValueError("Local incidence 'βr' must have shape (nr,).")

        # Redistribute r, c and βr if requested.  New chords and angles are
        # generated by interpolating within the existing range.
        if redist is not None:
            t_root = np.arcsin(self._r_stn[0] / self._r_stn[-1])
            new_stn = self.R * np.sin(np.linspace(t_root, 0.5 * np.pi,
                                                  num=redist))
            self._c_stn = _change_radii(self._c_stn, self._r_stn, new_stn)
            self._βr_stn = _change_radii(self._βr_stn, self._r_stn, new_stn)
            self._r_stn = new_stn

        if len(self._r_stn) < 5:
            raise ValueError("At least five radial stations 'r' required "
                             "for BEMPropeller.")

        # Set number of blades.
        self._B = B
        if self._B <= 0:
            raise ValueError("One or more blades 'B' required.")

        # Setup aerofoils.
        if isinstance(foils, Sequence):
            # Prescribed stations for each aerofoil.  Note 'r' used here is
            # the original radii as these have been copied and not
            # redistributed.
            self._foil_af = foils
            self._r_af = np.array(r, ndmin=1, copy=True)

            if len(self._foil_af) != len(self._r_af):
                raise ValueError("Aerofoil array must have shape (nr,).")

        else:
            # Same aerofoil at all analysis stations.
            self._foil_af = [foils] * len(self._r_stn)
            self._r_af = self.r().copy()

        # Setup collective pitch angle.
        self._β0_range = (min(β0_range), max(β0_range))
        self._β0 = β0
        if not (self._β0_range[0] <= self._β0 <= self._β0_range[1]):
            raise ValueError("β0 outside operating range.")

        # Setup pitch control.
        self._pitch_control = pitch_control
        if self._pitch_control not in ('fixed', 'cs'):
            raise ValueError(f"Invalid pitch control "
                             f"'{self._pitch_control}'.")

        # Setup atmosphere.
        self._ρ0, self._a0, self._μ0 = ρ0, a0, μ0

        # Setup solver parameters.
        self.maxits_flow, self.tol_flow = maxits_flow,  tol_flow
        self.maxits_cs, self.tol_cs = maxits_cs, tol_cs

        # The primary operating state variables are always V, Ω for BE
        # propellers.
        self._V0, self._Ω = _NOT_READY, _NOT_READY
        self._Qs, self._T = _NOT_READY, _NOT_READY

        # Setup element-wise values.  Note:
        #   - Chords are set by averaging to preserve area.
        #   - Pitch angles are determined by interpolation to preserve
        #     local trend.
        #   - Remainder are set to None for safety (and to happify linter).
        self._Δr = self._r_stn[1:] - self._r_stn[:-1]
        self._r = 0.5 * (self._r_stn[:-1] + self._r_stn[1:])
        self._c = 0.5 * (self._c_stn[:-1] + self._c_stn[1:])
        self._βr = _change_radii(self._βr_stn, self._r_stn, self._r)

        self._M, self._Re = _NOT_READY, _NOT_READY
        self._α, self._cl, self._cd = _NOT_READY, _NOT_READY, _NOT_READY,
        self._dT_dr, self._dQs_dr = _NOT_READY, _NOT_READY
        self._η_local = _NOT_READY
        self._V2_a, self._V2_t = _NOT_READY, _NOT_READY  # Just behind disc.

        # Setup aerofoil values.
        self._M_af, self._Re_af = _NOT_READY, _NOT_READY
        self._α_af, self._cl_af, self._cd_af = (_NOT_READY,) * 3

    # -- Public Methods -----------------------------------------------------

    def α(self) -> NDArray:
        """
        Returns
        -------
        a : (nr,) ndarray
            Angle of Attack at each radial station.  This is computed by
            extrapolating the element-wise values held internally.
        """
        if self._α is _NOT_READY:
            raise NotSetError("'α' is not available.")
        return _change_radii(self._α, self._r, self._r_stn, widen=True)

    @property
    def B(self) -> int:
        return self._B

    # Future work: Simplfiy this to make it more compatible with States
    # object. Move 'if' clauses to separate method?
    def β(self, *, r: [float] = None, x: [float] = None) -> NDArray:
        if x is None:
            if r is None:
                # No arguments:
                return self._β0 + self._βr_stn
            else:
                return self._β0 + self.βr(r=r)
        else:
            return super().β(r=r, x=x)

    @property
    def β0(self) -> float:
        """Collective blade incidence angle [radians]."""
        return self._β0

    @property
    def β0_range(self) -> (float, float):
        """Collective blade incidence angle range (min, max) [radians]."""
        return self._β0_range

    def βr(self, *, r: [float] = None, x: [float] = None) -> NDArray:
        """
        Returns
        -------
        βr : ndarray
            Result depends on keyword arguments:

            - No arguments:  Returns an `ndarray, shape (nr,)` the local
              pitch angle `βr` at all radial stations.

            - `r=[r0, r1, ...]`: Returns an `ndarray` giving the
              local pitch angle `βr` at the given radial stations [βr1,
              βr2, ...].

            - `x=[x0, x1, ...]: Returns an `ndarray` giving the
              local pitch angles `βr` at the requested radius fractions.
              Implemented as ``r=np.asarray(x)*self.R)``.
        """
        if x is None:
            if r is None:
                # No arguments.
                return self._βr_stn
            else:
                # 'r' only argument.
                return _change_radii(self._βr_stn, self._r_stn, np.asarray(r))

        else:
            if r is None:
                # 'x' only argument.
                return self.βr(r=np.asarray(x) * self.R)
            else:
                # Both arguments.
                raise TypeError("'x=' and 'r=' cannot be used together.")

    def c(self) -> NDArray:
        return self._c_stn

    def cl(self) -> NDArray:
        """
        Returns
        -------
        cl : (nr,) ndarray
            Lift coefficient at each radial station.  This is computed by
            extrapolating the element-wise values held internally.
        """
        if self._cl is _NOT_READY:
            raise NotSetError("'cl' is not available.")
        return _change_radii(self._cl, self._r, self._r_stn, widen=True)

    def cd(self) -> NDArray:
        """
        Returns
        -------
        cd : (nr,) ndarray
            Drag coefficient at each radial station.  This is computed by
            extrapolating the element-wise values held internally.
        """
        if self._cd is _NOT_READY:
            raise NotSetError("'cd' is not available.")
        return _change_radii(self._cd, self._r, self._r_stn, widen=True)

    @property
    def ρ0(self) -> float:
        if self._ρ0 is _NOT_READY:
            raise NotSetError("'ρ0' is not available.")
        return self._ρ0

    # Future work: This can probably be pulled out of the class and made
    # into a function that works 'on' the class.
    def design(self, target: str, *, cl: ArrayLike = None,
               maxits_opt: int = 50, disp: int | bool = None, **kwargs):
        """
        Modify the geometry of the propeller at a specified operating
        condition.  Required parameters will depend on the type of
        modification (see `target`).  Valid combinations are:

        .. note:: This method applies to fixed pitch propeller operation
           only (``pitch_control == 'fixed'``).  A constant speed propeller
           will automatically balance the power / torque and this would not
           converge to a valid solution.

        Parameters
        ----------
        target : str
            Type of changes to make to propeller to accomplish the design
            objective:
            - ``'twist'``:  The local section angles along the blade `βr`
              are individually varied to give the loading specified by the
              `cl` parameter.  In this case, `Ω` (or equivalent) must be
              used by itself to speciy shaft conditions in `**kwargs`.
            - ``'twist-chord'``:  In addition to changes made by
              ``'twist'``, the chord ditribution is scaled by an overall
              factor to balance the required torque. In this case the `Ps` or
              `Qs` parameter must be added to ``**kwargs`` to allow the
              torque to be determined.

        cl : shape (nr,), array_like
            Desired section cl values.

        maxits_opt : int, default = 50.
            Maximum number of iterations allowed for modification.

        disp : bool, int, optional
            See `run(...)` for details.

        kwargs : {str: value}
            Remaining keyword arguments define the operating condition and
            are passed directly to ``operate(...)``.

        Raises
        ------
        ValueError
            For illegal combinations or values of parameters.
        SolverError
            If maximum iterations are exceeded / the flow could not be
            converged.
        """
        disp_enter(disp)

        # Rearrange arguments into required operating values.
        kwargs = self._reorder_run_kwargs(kwargs)

        # Set pitch control up front for checking.
        self._pitch_control = kwargs.pop('pitch_control', self._pitch_control)

        # Future work: Add more design types as reqd?
        if target in ('twist', 'twist-chord'):
            # For fixed-pitch propeller, blade angles are adjusted to give
            # lift coefficient.  Optionally chords are adjusted to give
            # requested torque.
            if self._pitch_control != 'fixed':
                raise ValueError(f"Modification '{target}' applicable only "
                                 f"to fixed pitch propellers.")

            if target == 'twist-chord':
                # For combined twist-chord, the shaft torque becomes the
                # target value as the propeller will be fixed pitch.
                try:
                    Q_target = kwargs.pop('Qs')
                except KeyError:
                    raise TypeError(f"Could not determine target torque to "
                                    f"modify blade chord.")
            else:
                Q_target = None

            if cl is None:
                raise ValueError("Lift coefficient must be specified.")

            disp_print(f"Modifying propeller '{target}':")

            # Restrict target cl to elements.
            cl_target = _change_radii(cl, self._r_stn, self._r)

            # Main design loop.
            for it in range(maxits_opt):

                # Solve the propeller.
                self.run(**kwargs)  # type: ignore

                # Check convergence.
                cl_err = cl_target - self._cl
                max_cl_err = np.max(np.abs(cl_err))
                if Q_target is not None:
                    Q_err = Q_target - self.Qs
                else:
                    Q_err = 0.0

                # Future work: Make convergence parameters adjustable.
                if (max_cl_err <= 0.005) and (np.abs(Q_err) < 0.01):
                    # Prolong βr, c back to radial stations.
                    self._βr_stn = _change_radii(self._βr, self._r,
                                                 self.r(), widen=True)
                    self._c_stn = _change_radii(self._c, self._r,
                                                self.r(), widen=True)

                    disp_print(f"\t-> Converged.  "
                               f"|error(cl)| = {max_cl_err:.05f}, "
                               f"error(Q) = {Q_err:.01f}")
                    disp_exit()
                    return

                # Further refinement required.
                disp_print(f"\t-> Iteration {it + 1:3d} of {maxits_opt:3d}: "
                           f"|error(cl)| = {max_cl_err:8.05f}, "
                           f"error(Q) = {Q_err:8.01f}")

                # Set relaxation factors to stabilise convergence.
                relax_β = 0.5
                relax_c = 0.25

                # Adjust angle at each station using the cl error and an
                # assumed lift curve slope.
                self._βr += relax_β * (cl_err / (2 * np.pi))

                # If required, determine the chord scale factor as a simple
                # linear effect of the torque error.
                if Q_target is not None:
                    self._c *= (1 + relax_c * Q_err / Q_target)

            # Future work: Reset angles and chords?
            raise SolverError("Reached iteration limit without converging "
                              "modification.")

        else:
            raise ValueError(f"Unknown modification '{target}'.")

    def M(self) -> NDArray:
        """
        Returns
        -------
        M : shape (nr,), ndarray
            Mach number at each radial station.  This is computed by
            extrapolating the element-wise values held internally.
        """
        if self._M is _NOT_READY:
            raise NotSetError("'M' is not available.")
        return _change_radii(self._M, self._r, self._r_stn, widen=True)

    @property
    def nr(self) -> int:
        return len(self._r_stn)

    @property
    def Ω(self) -> float:
        if self._Ω is _NOT_READY:
            raise NotSetError("'Ω' is not available.")
        return self._Ω

    @property
    def Ps(self) -> float:
        """
        Shaft power Ps = Q_s.Ω.
        """
        return self.Qs * self.Ω

    @property
    def pitch_control(self) -> str:
        """
        Returns the current propeller pitch control mode.  Refer to
        `__init__` for available modes.
        """
        return self._pitch_control

    @property
    def Qs(self) -> float:
        if self._Qs is _NOT_READY:
            raise NotSetError("'Qs' is not available.")
        return self._Qs

    def r(self) -> NDArray:
        return self._r_stn

    def Re(self) -> NDArray:
        """
        Returns
        -------
        Re : shape (nr,) ndarray
            Reynolds number at each radial station.  This is computed by
            extrapolating the element-wise values held internally.

            .. note:: Values are converted to logarithms for extrapolation
               to preserve the character of the Reynolds Number.
        """
        if self._Re is _NOT_READY:
            raise NotSetError("'Re' is not available.")
        Re_log = np.log10(self._Re)
        Re_log_stn = _change_radii(Re_log, self._r, self._r_stn, widen=True)
        return np.power(10.0, Re_log_stn)

    def run(self, *, disp: int | bool = None, **kwargs):
        """
        Set the current operating state based on arguments provided. Required
        arguments will depend on the type of operation.  Valid combinations
        are:

        - For ``pitch_control == 'fixed'`` the collective angle `β0` remains
          constant.  Parameters `V`, `Ω` must be supplied.

        - For ``pitch_control == 'cs'`` the collective angle `β0` varies tp
          acheive torque equilibrium.  Valid parameter combinations are:

            - `V`, `Ω`, `Qs`.
            - `V`, `Ω`, `Ps`.
            - `V`, `Ω`, `T`.

          If the solution does not converge, `β0` is reset to its starting
          value before the `SolverError` exception is raised.

        Parameters
        ----------
        kwargs : {str: value}
            Dict of optional parameters which must form a complete
            operating condition for the propeller (all are `float` type):
            - `V0`: Freestream airspeed.
            - `Ω`: Rotational speed.  Note: Units should be *rad/s*
              to provide compatible values of `n` etc.
            - `Ps`: Shaft power, positive when the propeller is
              being driven.
            - `Qs`: Shaft torque, positive when the propeller is being driven.
            - `T`: Propeller thrust, positive when the propeller is
              accelerating the airflow.
            - `ρ0`: Freestream air density.
            - `a0`: Freestream speed of sound.
            - `μ0`: Freestream air viscosity.  Note that a value of zero
              provides for inviscid flow which gives corresponding Reynolds
              numbers of infinity at each radial analysis station.
        disp : int or bool, default = None
            Display information during calculation. See
            ``pyavia.util.disp_enter(...)`` for more details.

        Raises
        ------
        ValueError | TypeError
            For illegal combinations or values of parameters.
        SolverError
            If maximum iterations are exceeded / the flow could not be
            converged.
        """
        disp_enter(disp)

        # Rearrange arguments into required operating values.
        kwargs = self._reorder_run_kwargs(kwargs)

        # Set optional parameters.
        self._ρ0 = kwargs.pop('ρ0', self._ρ0)
        self._a0 = kwargs.pop('a0', self._a0)
        self._μ0 = kwargs.pop('μ0', self._μ0)
        self._pitch_control = kwargs.pop('pitch_control', self._pitch_control)
        self._β0 = kwargs.pop('β0', self._β0)
        if not (self._β0_range[0] <= self._β0 <= self._β0_range[1]):
            raise ValueError("β0 outside operating range.")

        # Set common operating parameters.
        try:
            self._V0 = kwargs.pop('V0')
            self._Ω = kwargs.pop('Ω')

        except KeyError:
            raise TypeError(f"Could not derive operating state V, Ω from "
                            f"parameters: {', '.join(kwargs.keys())}")

        if self._pitch_control == 'fixed':
            # Should be no further parameters present (T, P or Q).
            if kwargs:
                raise TypeError(
                    f"Fixed-pitch operating state over-specified by "
                    f"parameters: {', '.join(kwargs.keys())}")

            disp_print(f"Blade Element Analysis - Fixed Pitch - V0 = "
                       f"{self.V0:+.1f}, Ω = {self.Ω:+.1f}")
            self._run_fixed()

        elif self._pitch_control == 'cs':
            # Should be one parameter remaining (T, P or Q).
            if len(kwargs) != 1:
                raise TypeError(f"Unexpected operating state arguments "
                                f"remaining: {', '.join(kwargs.keys())}")

            target_attr, target_val = kwargs.popitem()

            disp_print(f"Blade Element Analysis - Constant Speed - V0 = "
                       f"{self.V0:+.1f}, Ω = {self.Ω:+.1f}, "
                       f"{target_attr} = {target_val:+.1f}")
            self._run_cs(target_attr, target_val)

        else:
            raise ValueError(
                f"Invalid pitch control '{self._pitch_control}'.")

        disp_exit()

    @property
    def T(self) -> float:
        if self._T is _NOT_READY:
            raise NotSetError("'T' is not available.")
        return self._T

    @property
    def V0(self) -> float:
        if self._V0 is _NOT_READY:
            raise NotSetError("'V0' is not available.")
        return self._V0

    @property
    def V2_a(self) -> NDArray:
        """
        Axial flow velocity component, immediately
        behind the propeller disc (all radial stations).
        """
        if self._V2_a is _NOT_READY:
            raise NotSetError("'V2_a' is not available.")
        return _change_radii(self._V2_a, self._r, self._r_stn, widen=True)

    @property
    def V2_t(self) -> NDArray:
        """
        Tangential flow velocity component, immediately behind the
        propeller disc (all radial stations).
        """
        if self._V2_t is _NOT_READY:
            raise NotSetError("'V2_t' is not available.")
        return _change_radii(self._V2_t, self._r, self._r_stn, widen=True)

    # -- Private Methods ----------------------------------------------------

    def _calc_aero(self, coeffs: [str], foil_err: bool = True) -> [NDArray]:
        """
        Does the following:
            - Computes and stores `self._Re_af`, `self._M_af` and
              `self._α_af` at aerofoil stations.
            - Runs the aerofoil section at each station.
            - Records the aerofoil attributes listed in `coeffs`.
              Also permitted are:
                * `cd_α0`: Aerofoil is run again to get `cd` at α = 0.
            - Extrapolates these values back to the blade element radii.

        Requires `self._α`, `self._Re` and `self._M` to have already been
        computed.

        If ``foil_err=True`` (the default), raises `SolverError` if any
        foil properties return `NaN`.

        Returns
        -------
        [NDArray]
            List of NDArrays where each element is an ndarray of the values
            along the blade element radii.
        """
        # Interpolate α, M, Re at aerofoil stations.  Change Reynolds
        # number to log basis to avoid any hiccups when interpolating.
        Re_log = np.log10(self._Re)
        self._α_af = _change_radii(self._α, self._r, self._r_af, widen=True)
        self._M_af = _change_radii(self._M, self._r, self._r_af, widen=True)
        Re_log_af = _change_radii(Re_log, self._r, self._r_af, widen=True)
        self._Re_af = np.power(10.0, Re_log_af)

        # Compute aerodynamic coefficients at each aerofoil station.
        res_af = [[] for _ in range(len(coeffs))]
        for i, foil in enumerate(self._foil_af):

            # Loop over each required attribute.
            last_α = None
            for j, coeff in enumerate(coeffs):
                if coeff == 'cd_α0':
                    # Special case, drag at α = 0.
                    α = 0.0
                    attr = 'cd'
                else:
                    # Most cases.
                    α = self._α_af[i]
                    attr = coeff

                # Run aerofoil if α has changed.
                if α != last_α:
                    foil.set_states(α=α, Re=self._Re_af[i], M=self._M_af[i])
                    last_α = α

                # Add station value to each attribute.
                res_af[j].append(getattr(foil, attr))

        # Check for NaN which can occur if aerofoil state is outside the
        # allowable area.
        if foil_err:

            def arr_print(x, log: bool = False) -> str:
                if not log:
                    fmt = {'float_kind': lambda x_: f"{x_:8.03f}"}
                else:
                    fmt = {'float_kind': lambda x_: f"{x_:8.02E}"}
                return np.array2string(np.asarray(x), max_line_width=200,
                                       suppress_small=True, separator=', ',
                                       formatter=fmt)

            err_str = ""
            for j, coeff in enumerate(coeffs):
                if np.any(np.isnan(res_af[j])):
                    err_str += f"\t{coeff:>5s} = {arr_print(res_af[j])}\n"

            if err_str:
                err_str = (f"Invalid foil properties along blade:\n" +
                           err_str +
                           f"Blade flow conditions:\n"
                           f"\t{'r':>5s} = {arr_print(self._r_af)}\n"
                           f"\t{'α°':>5s} = {arr_print(r2d(self._α_af))}\n"
                           f"\t{'Re':>5s} = {arr_print(self._Re_af, True)}\n"
                           f"\t{'M':>5s} = {arr_print(self._M_af)}\n")

                raise SolverError(err_str)

        # Expand coefficients to element mid-points.
        res = []
        for res_af_j in res_af:
            res.append(_change_radii(res_af_j, self._r_af, self._r))

        return res

    def _calc_forces(self, Vp_mag: NDArray, ϕ: NDArray):
        """
        Combines current flow state at each blade element location along
        with internally stored aerodynamic coefficients and atmospheric
        values to compute the following (stored internally):
            - Thrust gradient: dT_dr = 0.5.ρ.V^2.B.c.(cl.cos(ϕ) - cd.sin(ϕ))
            - Torque gradient: dQ_dr = 0.5.ρ.V^2.B.c.r.(cd.cos(ϕ) + cd.sin(ϕ))
            - Integral of gradients to get the total thrust and torque on
              the propeller.
            - Local efficienct of each blade element η_local = (V/Ω) (dT/dQ)

        Parameters
        ----------
        Vp_mag : (n,) ndarray
            Local flow speed.
        ϕ : (n,) ndarray
            Local flow helix angle to disc.
        """
        # Precompute fac = 0.5.ρ.V^2.B.c to save effort.
        fac = 0.5 * self._ρ0 * Vp_mag ** 2 * self._B * self._c
        T_term = (self._cl * np.cos(ϕ) - self._cd * np.sin(ϕ))
        Qs_term = self._r * (self._cd * np.cos(ϕ) + self._cl * np.sin(ϕ))

        self._dT_dr = fac * T_term
        self._dQs_dr = fac * Qs_term
        self._T = float(np.sum(self._dT_dr * self._Δr))
        self._Qs = float(np.sum(self._dQs_dr * self._Δr))
        self._η_local = (self._V0 / self._Ω) * (T_term / Qs_term)

    @staticmethod
    def _reorder_run_kwargs(kwargs) -> {}:
        """
        Rearrange / derive the required operating values of V0, Ω, Ps,
        Qs, T.  Note that power / torque / thrust are only required for
        constant speed operation. Raises `TypeError` if the operating
        state is over-specified.
        """
        # A 'write once' dict is used to detect redundant parameters.
        unique_kwargs = WriteOnceDict(kwargs)

        try:
            if 'Ps' in unique_kwargs:
                # Remove Ps when present.
                if 'Qs' in unique_kwargs:
                    # Ps, Qs -> Ω, Qs.
                    unique_kwargs['Ω'] = (unique_kwargs['Ps'] /
                                          unique_kwargs['Qs'])

                elif 'Ω' in unique_kwargs:
                    # Ps, Ω -> Ω, Qs.
                    unique_kwargs['Qs'] = (unique_kwargs['Ps'] /
                                           unique_kwargs['Ω'])

                del unique_kwargs['Ps']

        except KeyError:
            raise TypeError(f"Operating state over-specified by parameters: "
                            f"{', '.join(unique_kwargs.keys())}")

        return unique_kwargs

    def _run_cs(self, target_attr: str, target_val: float,
                disp: int | bool = None):
        """
        Varies β0 to converge the given target attribute of the
        propeller to the value requested.
        """
        disp_enter(disp)
        disp_print(f"-> Converging {target_attr} = {target_val:+.1f}:")

        failed = False
        y, dydβ0 = None, None
        β0_last, y_last = None, None,

        # Note: The CS convergence algorithm below took an awful lot of
        # timeto get right and is now quite bulletproof.  Change with caution.

        close_β_tol = d2r(0.05)
        β0_attempts = np.array([])
        Δβ0_search = (self._β0_range[1] - self._β0_range[0]) / 50
        # Note: Δβ0_search is used if previous points did not converge / do
        # not exist.

        for it in range(self.maxits_cs):

            # -- Generate New β0 --------------------------------------------

            if ((not failed) and
                    (dydβ0 is not None) and
                    (self._β0_range[0] < self._β0 < self._β0_range[1])):
                # Case: Good last step, inside bounds, known gradient.
                # The next β0 value is computed using one step of Newton's
                # method from the current point, clipped to limits.
                self._β0 += (target_val - y) / dydβ0
                self._β0 = np.clip(self._β0, *self._β0_range)

            else:
                # Case: All others.  Spiral away from the present position
                # and try to generate a new point that isn't near a
                # previously tried point.
                β0_next = None
                for i_next in range(0, 100):
                    # Loop starts at zero so that the very first point can be
                    # self._β0 itself.

                    # Positive side.
                    β0_pos = self._β0 + i_next * Δβ0_search
                    if ((self._β0_range[0] <= β0_pos <= self._β0_range[1])
                            and np.all(np.abs(β0_attempts - β0_pos) >
                                       close_β_tol)):
                        β0_next = β0_pos
                        break

                    # Negative side.
                    β0_neg = self._β0 - i_next * Δβ0_search
                    if ((self._β0_range[0] <= β0_neg <= self._β0_range[1])
                            and np.all(np.abs(β0_attempts - β0_neg) >
                                       close_β_tol)):
                        β0_next = β0_neg
                        break

                if β0_next is not None:
                    self._β0 = β0_next
                else:
                    # Not a single trial point could be generated... highly
                    # unusual.  This is a clear failure.
                    break

            # -- Run New β0 -------------------------------------------------

            β0_attempts = np.append(β0_attempts, self._β0)
            failed = False
            try:
                self._run_fixed()
                y = getattr(self, target_attr)
                disp_print(f"\t... Iteration {it + 1:3d} of "
                           f"{self.maxits_cs:3d}: "
                           f"β0 = {r2d(self._β0):+6.02f}°, "
                           f"{target_attr} = {y:+11.1f}")

            except SolverError:
                disp_print(f"\t... Iteration {it + 1:3d} of "
                           f"{self.maxits_cs:3d}: "
                           f"β0 = {r2d(self._β0):+6.02f}°, "
                           f"Convergence failed.")
                failed = True
                continue

            # -- Update Gradient / Check Convergence ------------------------

            if β0_last is not None:
                # Update the gradient.  If a previous estimate exists,
                # use relaxation = 0.5.  This significantly improves
                # stability of the CS convergence particularly where there
                # are some discontinuities / other nonlinearities.
                Δβ0 = self._β0 - β0_last
                Δy = y - y_last
                if dydβ0 is not None:
                    dydβ0 = 0.5 * dydβ0 + 0.5 * Δy / Δβ0  # Relax

                else:
                    dydβ0 = Δy / Δβ0  # No relaxation on first step.

                # Check convergence.
                error = np.abs(Δy)
                if target_val != 0:
                    error /= target_val

                if error < self.tol_cs:
                    disp_print(f"\t-> Converged.")
                    disp_exit()
                    return

            # Update last 'good' point.
            β0_last, y_last = self._β0, y

        # -------------------------------------------------------------------

        disp_exit()
        raise SolverError(f"Failed to converge constant speed operation "
                          f"for {target_attr} = {target_val:+.1f}")

    @abstractmethod
    def _run_fixed(self, disp: int | bool = 0):
        """
        Calculate performance assuming fixed pitch (i.e. holding β0
        constant), using the stored flow conditions.  This is the main
        performance calculation function and also forms the inner loop for
        constant speed analyses.  Specific implementations override this
        method.

        Parameters
        ----------
        disp : int or bool, optional
           See `run(...)` for details.

        Raises
        ------
        ValueError
           If invalid operating conditions are specified.
        SolverError
           If the flow could not be converged.
        """
        raise NotImplementedError


# ============================================================================


def _change_radii(y: ArrayLike, from_r: ArrayLike, to_r: ArrayLike,
                  widen: bool = False) -> np.ndarray:
    """
    Interpolate / extrapolate quantity `y` originally aligned with `from_r`
    to be aligned with `to_r`.  PCHIP is used unless otherwise stipulated.
    If ``widen = True``, extrapolation (prolongation) is allowed if the
    `to_r` radii are outside the limits of `from_r`.
    """
    return PchipInterpolator(from_r, y,  # type: ignore
                             extrapolate=True if widen else False)(to_r)

# ============================================================================
