from collections import defaultdict
import warnings

import numpy as np
from numpy.typing import ArrayLike

from .base import Foil2DAero, std_α
from .stall import PostStall2DMixin
from pyavia.numeric.solve import SolverError
from pyavia.numeric.filter import savgol_variable
from pyavia.numeric.function_1d import Line1D, PCHIP1D

# Written by Eric J. Whitney, January 2023.

d2r, r2d = np.deg2rad, np.rad2deg


# ======================================================================

class Polar2DAero(Foil2DAero):
    """
    Aerodynamic model that interpolates a polar / performance data set
    supplied for a single Reynolds and Mach Number.

    Notes
    -----
    - All properties return `NaN` outside the available `α` range for
      that particular property.
    - If a property is defined by a single value (i.e. is constant)
      then it is assumed to apply over the same `α` range as the `cl`
      property.
    """
    warn_α_range = True
    """If ``warn_α_range==True``, attempts to extrapolate for values 
    outside the `α` range for each property will produce a warning. Note that 
    properties return a value of `NaN` in this situation."""

    def __init__(self, *, Re: float, M: float, α_data: ArrayLike,
                 cl_data: ArrayLike, cd_data: ArrayLike,
                 cm_qc_data: ArrayLike, xtr_u_data: ArrayLike = None,
                 xtr_l_data: ArrayLike = None, α: float = 0.0,
                 allow_duplicates: bool = False, smooth: bool = False,
                 **kwargs):
        r"""
        Parameters
        ----------
        Re : float
            Fixed Reynolds number.
        M : float
            Fixed Mach number.
        α_data : array_like, shape (n,)
            Angles of attack.

            .. note:: All elements in this array must be valid values.
               Individual `None` entries are not permitted.

        cl_data : array_like, shape (n,)
            Lift coefficients :math:`c_l`.  At least two non-`None` values
            are required (to give a linear trend at a minimum).
        cd_data : array_like, shape (n,)
            Drag coefficients :math:`c_d`.  At least two non-`None` values
            are required (to give a linear trend at a minimum).
        cm_qc_data : array_like, shape (n,)
            Pitching moment coefficients (quarter chord) :math:`c_{m,c/4}`.
            At least one non-`None` value is required (to give a constant
            trend at a minimum).
        xtr_u_data : array_like, shape (n,), optional
            Upper surface boundary layer transition point :math:`x_u/c`.
        xtr_l_data : array_like, shape (n,), optional
            Lower surface boundary layer transition point :math:`x_u/c`.
        α : float, default = 0.0
            Initial angle of attack.
        allow_duplicates : bool, default = False
            Allow duplicate `α` entries.  If `True` each property (`cl_data`,
            `cd_data`, etc) having duplicated values are averaged to produce
            a single internal value. If `False`, `ValueError` is raised.
        smooth : bool, default = False
            If `True`, properties are smoothed to remove numerical or
            experimental noise.

        Notes
        -----
        - Where supplied, the length of sequences `α_data`, `cl_data`,
          `cd_data`, etc must be the same. Within each sequence, individual
          `α`-entries can be omitted by replacing with the value `None`,
          (except for `α_data`).  See `Parameters` above for the minimum
          number of real values required in each sequence.
        - `Re` and `M` are fixed i.e. they cannot be used in `set_states`
          for this model.
        - If selected, properties `cl_data`, `cd_data`, etc are smoothed
          using a digital filter:
            - A Savitzky–Golay variable step filter is used. This filter is of
              third order with a window size set to capture
              :math:`α \pm 2.5°` and is applied over three passes.
            - Smoothing is only applied if there are sufficiently many
              points to run the filter.
            - To reduce noise, after deriving the the :math:`c_{lα}(α)` curve
              from :math:`c_l(α)` the same filter is applied to this data
              (same order, window size and passes).
        """
        super().__init__(**kwargs)
        if Re <= 0.0:
            raise ValueError(f"Require Reynolds number > 0, got: {Re}")
        if M < 0.0:
            raise ValueError(f"Require Mach number >= 0, got: {M}")

        self._Re, self._M = Re, M
        self._α = std_α(α)

        # -- Assemble Data ---------------------------------------------------

        # Put arguments into single dict for easier processing.
        prop_data = {'cl': cl_data, 'cd': cd_data, 'cm_qc': cm_qc_data,
                     'xtr_u': xtr_u_data, 'xtr_l': xtr_l_data}

        # Check matching lengths of all property fields.
        if (n_rows := len(α_data)) < 2:
            raise ValueError("At least two α values required.")

        for prop, data in prop_data.items():
            if data is not None and len(data) != n_rows:
                raise ValueError(f"Length of '{prop}' ({len(data)}) != "
                                 f"length of 'α' ({n_rows}).")

        # -- Clean-Up Data ---------------------------------------------------

        filter_order = 3
        filter_width = 0.04363323  # 2.5° -> [rad].

        prop_α_vals = {}
        for prop, data in prop_data.items():
            if data is None:
                # Store as empty α-value table.
                prop_α_vals[prop] = np.array([])
                continue

            # For each attribute we first build dict of {α_i: [y1, y2...]}
            # which allows for possible duplicates.
            α_vals = defaultdict(list)
            for α_i, val in zip(α_data, data, strict=True):
                if α_i is None:
                    raise ValueError("α=None not allowed.")

                if val is not None:
                    α_vals[α_i].append(val)

            # Check for duplicate α values and do averaging if permitted.
            for α_i, vals in α_vals.items():
                if (n_local := len(vals)) == 1:
                    α_vals[α_i] = vals[0]  # Convert to scalar.

                elif n_local > 1:
                    if allow_duplicates:
                        # Take average of values.
                        α_vals[α_i] = sum(vals) / n_local

                    else:
                        raise ValueError(f"Got duplicate {prop} values "
                                         f"at α = {r2d(α_i):.2f}°.")

            # Convert to [(α, val), ...] for sorting.
            α_vals = [(α_i, val) for α_i, val in α_vals.items()]
            α_vals.sort(key=lambda x: x[0])
            α_vals = np.array(α_vals)  # Conver to ndarray.

            while smooth:  # 'while' masquerading as 'if'.
                # Apply Savitzky–Golay filter (if possible).  Get average Δα
                # for data set and compute window width (odd number reqd) to
                # give average width.
                Δα_mean = np.diff(α_vals[:, 0]).mean()
                window = int(((filter_width / Δα_mean) // 2)) * 2 + 1

                if filter_order >= window:
                    warnings.warn(
                        f"Skipping filter for '{prop}': 'α' spacing too wide. "
                        f"\n\t-> Avg Δα = {r2d(Δα_mean):.02f}°, filter width "
                        f"= {r2d(filter_width):.02f}°"
                        f"\n\t-> Filter window = {window} < order "
                        f"{filter_order}.")
                    break

                if len(α_vals) < window + 1:
                    warnings.warn(f"Skipping filter for '{prop}': too few "
                                  f"datapoints. Required {window + 1} points "
                                  f"got {len(α_vals)} points.")
                    break

                # Split tuples into lists, apply filter and remerge.
                αs, vals = zip(*α_vals)
                smoothed = savgol_variable(αs, vals, window=window,
                                           order=filter_order, passes=3)
                α_vals = np.array(list(zip(αs, smoothed)))
                break  # One trip 'while' loop.

            prop_α_vals[prop] = α_vals

        # -- Build α-Property Interpolators ----------------------------------

        α_min = np.min(prop_α_vals['cl'][:, 0])  # Used to extend 'single' ...
        α_max = np.max(prop_α_vals['cl'][:, 0])  # ... or 'no' value curves.
        n_reqd = {  # Min. points required for each type.
            'cl': 2, 'cd': 2, 'cm_qc': 1, 'xtr_u': 0, 'xtr_l': 0}
        max_α_gap = d2r(5.0)  # Max gap in multipoint curves.
        self._α_interps = {}

        for prop, α_vals in prop_α_vals.items():
            # Check number of points.
            if (n_pts := α_vals.shape[0]) < n_reqd[prop]:
                raise ValueError(f"Require {n_reqd[prop]} point/s min for "
                                 f"(α, {prop}) curve.")

            # Generate interpolators.
            if n_pts > 1:
                # Multipoint case: Generate PCHIP (linear up to cubic). If
                # multiple points are mandatory, check for excessive gaps.
                if (n_reqd[prop] > 1 and
                        np.any(np.diff(α_vals[:, 0]) > max_α_gap)):
                    raise ValueError(f"Gap in (α, {prop}) curve > "
                                     f"{r2d(max_α_gap):.2f}°.")

                self._α_interps[prop] = PCHIP1D(α_vals[:, 0], α_vals[:, 1],
                                                ext_lo=np.nan, ext_hi=np.nan)

            elif n_pts == 1:
                # Single value case: Assume constant across entire α range.
                self._α_interps[prop] = Line1D(
                    x=α_vals[0, 0], y=α_vals[0, 1], slope=0,
                    x_domain=[α_min, α_max], ext_lo=np.nan, ext_hi=np.nan)

            else:
                # No value case: Return NaN everywhere.
                self._α_interps[prop] = Line1D(slope=0, intercept=np.nan)

        # -- Build α-clα Interpolator ----------------------------------------

        # Generate a α-clα interpolator based on the α-cl model.  This
        # allows for smoothing to be applied to the α-cl curve prior to
        # computing its derivative.
        α_vals = self._α_interps['cl'].x
        clα_vals = self._α_interps['cl'].derivative(α_vals)

        while smooth:  # Another 'while' masquerading as 'if'.
            Δα_mean = np.diff(α_vals).mean()
            window = int(((filter_width / Δα_mean) // 2)) * 2 + 1  # Odd.

            if filter_order >= window:
                warnings.warn(
                    f"Skipping filter for 'clα': 'α' spacing too wide. "
                    f"\n\t-> Avg Δα = {r2d(Δα_mean):.02f}°, filter width "
                    f"= {r2d(filter_width):.02f}°"
                    f"\n\t-> Filter window = {window} < order "
                    f"{filter_order}.")
                break

            if len(α_vals) < window + 1:
                warnings.warn(f"Skipping filter for 'clα': too few "
                              f"datapoints. Required {window + 1} points "
                              f"got {len(α_vals)} points.")
                break

            clα_smooth = savgol_variable(α_vals, clα_vals, window=window,
                                         order=filter_order, passes=3)
            clα_vals = clα_smooth
            break  # One trip 'while' loop.

        self._α_interps['clα'] = PCHIP1D(α_vals, clα_vals,
                                         ext_lo=np.nan, ext_hi=np.nan)

        # -- Setup Live Values ----------------------------------------------

        self._α0 = None  # Fixed after first calculation.
        self._reset_α_states()  # Setup states that vary with 'α'.

    # -- Public Methods -----------------------------------------------------

    @property
    def α(self) -> float:
        return self._α

    @property
    def α0(self) -> float:
        """
        Returns the Zero Lift Angle (ZLA) `α0`, computed by finding a valid
        root of the the lift interpolation function :math:`c_l = f(α)`.

        Notes
        -----
        - If the given data range of `cl` does not intersect the axis i.e.
          there is no `a` giving :math:`c_l(a) = 0`, `NaN` is returned.

        - If the `α`-`cl` curve has multiple axis intersections, this
          returns the first value having a positive corresponding `clα`
          value. Raises `SolverError` if no valid axis intersection
          could be found in this case, as this may represent a pathologic
          problem with the input data.
        """
        if self._α0 is None:
            # Get α at all axis intersections (i.e. cl = 0).
            α0 = self._α_interps['cl'].solve()

            if not α0:
                # Case: No intersection point.
                self._α0 = np.nan

            elif len(α0) == 1:
                # Case: Single intersection point.
                self._α0 = α0[0]

            else:
                # Case: Multiple intersection points.  First, eliminate any
                # that correspond to a negative or weak positive slope.
                α0 = [α0_i for α0_i in α0
                      if self._α_interps['clα'](α0_i) > 0.5]
                if not α0:
                    raise SolverError("Failed to compute a0.",
                                      details="No α-cl axis intersection "
                                              "found with clα > 0.5.")

                # Finally, save the value closest to zero.
                self._α0 = α0[np.argmin(np.abs(α0))]

        return self._α0

    @property
    def α_stall_neg(self) -> float:
        """
        Angle of attack corresponding to lowest :math:`c_l` available in the
        input data.  See ``cl_stall_pos()`` for implementation details.
        """
        α_cl = self._α_interps['cl']
        return α_cl.x[np.argmin(α_cl.y)]

    @property
    def α_stall_pos(self) -> float:
        """
        Angle of attack corresponding to highest :math:`c_l` available in the
        input data.  See ``cl_stall_pos()`` for implementation details.
        """
        α_cl = self._α_interps['cl']
        return α_cl.x[np.argmax(α_cl.y)]

    @property
    def cd(self) -> float:
        return self._interp_prop('cd')

    @property
    def cl(self) -> float:
        return self._interp_prop('cl')

    @property
    def clα(self) -> float:
        return self._interp_prop('clα')

    @property
    def cl_stall_neg(self) -> float:
        """
        Lowest :math:`c_l` available in the input data.  See `cl_stall_pos`
        for implementation details.
        """
        return np.min(self._α_interps['cl'].y)

    @property
    def cl_stall_pos(self) -> float:
        """
        Highest :math:`c_l` available in the input data.

        Notes
        -----
        No checks are made to determine if this is actually a local maximum
        (i.e. that :math:`c_l` values are actually lower on either side),
        or if it is the first or only maximum (i.e. an initial stall or
        overall maximum lift, etc).
        """
        return np.max(self._α_interps['cl'].y)

    @property
    def cm_qc(self) -> float:
        return self._interp_prop('cm_qc')

    @classmethod
    def input_states(cls) -> frozenset[str]:
        return super().input_states() | {'α'}

    @property
    def M(self) -> float:
        return self._M

    @classmethod
    def output_states(cls) -> frozenset[str]:
        return super().output_states() | {'xtr_l', 'xtr_u'}

    @property
    def Re(self) -> float:
        return self._Re

    def set_state(self, *, α: float = None) -> frozenset[str]:
        """
        `α` is the only available parameter (`Re` and `M` are fixed).
        """
        if α is not None:
            α = std_α(α)
            if self._α != α:
                self._α = α
                self._reset_α_states()
                return frozenset(['α'])

        return frozenset()

    @property
    def valid_state(self) -> bool:
        """A `Polar2DAero` object always returns `True`."""
        return True

    @property
    def xtr_l(self) -> float:
        """Lower surface boundary layer transition point."""
        return self._interp_prop('xtr_l')

    @property
    def xtr_u(self) -> float:
        """Upper surface boundary layer transition point."""
        return self._interp_prop('xtr_u')

    # -- Private Methods -----------------------------------------------------

    def _interp_prop(self, prop: str) -> float:
        """
        Interpolate property `prop` at the current `α`, cache and return
        value.  If `deriv_of` is supplied, computed as the derivative of
        the given property at the current `α`. Returns `NaN` if out of
        valid range / unavailable.
        """
        # Return if already set.
        if (val := getattr(self, '_' + prop)) is not None:
            return val

        # Get interpolator and check we are within bounds.
        interp = self._α_interps[prop]
        interp_x_min, interp_x_max = interp.x_domain
        if (self.warn_α_range and
                not (interp_x_min <= self.α <= interp_x_max)):
            warnings.warn(
                f"α = {r2d(self.α):+.03f}° out of range getting {prop}:\n"
                f"\t{'α_range':>7s} = "
                f"{r2d(interp_x_min):+.03f}° -> {r2d(interp_x_max):+.03f}°\n"
                f"\t{'Re':>7s} = {self.Re:.03E}\n"
                f"\t{'M':>7s} = {self.M:.03f}\n")

        # Interpolate and cache for re-use.
        val = interp(self.α)
        setattr(self, '_' + prop, val)
        return val

    def _reset_α_states(self):
        """Initialise / reset all state outputs that change with 'α'."""
        self._cl, self._cd, self._cm_qc = None, None, None
        self._xtr_u, self._xtr_l = None, None
        self._clα = None


# ===========================================================================


class Polar2DAeroPostStall(PostStall2DMixin, Polar2DAero):
    """
    `Polar2DAeroPostStall` extends `Polar2DAero` class by adding the
    `PostStall2DMixin` mixin to give estimated properties at large
    positive and negative (post-stall) angles of attack.

    Default post-stall model behaviour is to replace base model `NaN`
    values only, regardless of lift (unless overridden).
    """

    def __init__(self, **kwargs):
        super().__init__(**({'post_nan': True, 'post_crossover': False} |
                            kwargs))


# ============================================================================
